"""CPU 冒烟演示:不下载任何模型/数据,用迷你模型走完 M1 量化核心逻辑链。

目的:在无 GPU、无 transformers/datasets 的机器上,快速理解并验证
      "量化数学(RTN/GPTQ/AWQ) → 校准统计收集 → 就地替换 Linear →
       压缩统计 → 伪量化前后输出对比" 这条主流程。
运行:python scripts/cpu_smoke.py(乱码时 PYTHONUTF8=1 python scripts/cpu_smoke.py)
"""
import os
import sys

# 把项目根目录加入 sys.path,使从 scripts/ 下直接运行也能 import lowbitsparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from lowbitsparse.quant import (
    QuantConfig, apply_quantization, compression_report, FakeQuantLinear,
    gptq_quantize_weight, awq_quantize_weight,
    collect_calib_stats, target_linear_names)
from lowbitsparse.quant.rtn import rtn_quantize_weight
from lowbitsparse.quant.primitives import fake_quant_groupwise


class TinyLM(nn.Module):
    """一个迷你"类 Transformer":几个 Linear 模拟 attention/MLP 投影 + lm_head。"""

    def __init__(self, d=256, vocab=1000):
        super().__init__()
        self.q_proj = nn.Linear(d, d)      # 模拟注意力投影
        self.o_proj = nn.Linear(d, d)
        self.up_proj = nn.Linear(d, 4 * d)  # 模拟 MLP
        self.down_proj = nn.Linear(4 * d, d)
        self.lm_head = nn.Linear(d, vocab)  # 输出层,量化时默认 skip

    def forward(self, x):
        x = self.o_proj(self.q_proj(x))
        x = self.down_proj(torch.relu(self.up_proj(x)))
        return self.lm_head(x)


def step1_rtn_math():
    """步骤1:单看 RTN 量化数学 —— 位宽越低,反量化误差越大。"""
    print("\n[步骤1] RTN 量化数学:一个权重矩阵在不同 bit 下的相对误差")
    torch.manual_seed(0)
    w = torch.randn(128, 512)   # CPU 上用 float32
    for bits in (8, 4, 3):
        cfg = QuantConfig(n_bits=bits, group_size=128, symmetric=False)
        w_dq = rtn_quantize_weight(w, cfg)      # 量化-反量化
        rel = (w - w_dq).norm().item() / w.norm().item()
        print(f"  INT{bits}: 相对误差 = {rel:.4%}  (shape 保持 {tuple(w_dq.shape)})")


def step2_apply_and_report():
    """步骤2:就地替换整个模型的 Linear,并统计压缩比。"""
    print("\n[步骤2] 就地替换 Linear -> FakeQuantLinear + 压缩统计")
    model = TinyLM()                              # float32,CPU 友好
    print("  替换前 q_proj 类型:", type(model.q_proj).__name__)

    cfg = QuantConfig(n_bits=4, group_size=128, symmetric=False, skip=("lm_head",))
    model, n = apply_quantization(model, cfg)     # 就地替换,返回同一个 model
    print(f"  替换了 {n} 个 Linear(lm_head 被 skip)")
    print("  替换后 q_proj 类型:", type(model.q_proj).__name__)
    print("  lm_head 是否仍为 Linear:", isinstance(model.lm_head, nn.Linear))

    rep = compression_report(model)
    print(f"  压缩统计: 等效 {rep['effective_bits']} bit, 体积 {rep['size_mb']} MB, "
          f"量化权重数 {rep['quant_weights']}")
    return model


def step3_forward_compare():
    """步骤3:同一输入,量化前 vs 量化后输出的差异(伪量化只影响数值,不改结构)。"""
    print("\n[步骤3] 伪量化前后前向输出对比")
    torch.manual_seed(1)
    model = TinyLM()
    x = torch.randn(2, 256)
    y_fp = model(x).detach()                      # 量化前输出

    apply_quantization(model, QuantConfig(n_bits=4, group_size=128))
    y_q = model(x).detach()                       # 量化后输出(同一 model,已就地改)
    diff = (y_fp - y_q).norm().item() / y_fp.norm().item()
    print(f"  输出 shape 一致: {tuple(y_fp.shape)} == {tuple(y_q.shape)}")
    print(f"  输出相对差异: {diff:.4%}  (量化误差传导到最终输出,符合预期)")


def _corr_input(n_tok, in_f):
    """构造通道相关的合成激活,使 Hessian 明显非对角(GPTQ 才有发挥空间)。"""
    base = torch.randn(n_tok, in_f)
    return base + 0.8 * torch.roll(base, 1, dims=1)


def step4_gptq_vs_rtn():
    """步骤4:GPTQ 用 Hessian 补偿误差,相关输入下输出误差低于 RTN。"""
    print("\n[步骤4] GPTQ(Hessian 校准 + 误差补偿) vs RTN")
    torch.manual_seed(0)
    W = torch.randn(96, 128)
    X = _corr_input(512, 128)             # 相关激活
    H = X.t() @ X                          # 校准 Hessian
    cfg = QuantConfig(n_bits=3, group_size=128, symmetric=False, method="gptq")

    W_gptq = gptq_quantize_weight(W, H, cfg)
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    e_g = (X @ (W - W_gptq).t()).norm().item()
    e_r = (X @ (W - W_rtn).t()).norm().item()
    print(f"  INT3 输出误差(越低越好): RTN={e_r:.3f}  GPTQ={e_g:.3f}")
    print(f"  GPTQ 相对 RTN 降低: {(1 - e_g / e_r):.2%}")


def step5_awq_vs_rtn():
    """步骤5:AWQ 按激活幅度保护重要通道,激活悬殊时加权误差低于 RTN。"""
    print("\n[步骤5] AWQ(激活感知逐通道缩放) vs RTN")
    torch.manual_seed(1)
    W = torch.randn(64, 256)
    act = torch.rand(256) + 0.05
    act[:4] *= 50.0                        # 4 个超大激活通道
    cfg = QuantConfig(n_bits=3, group_size=256, symmetric=False, method="awq")

    W_awq = awq_quantize_weight(W, act, cfg)
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    wa = ((W - W_awq) * act.unsqueeze(0)).pow(2).sum().item()
    wr = ((W - W_rtn) * act.unsqueeze(0)).pow(2).sum().item()
    print(f"  INT3 激活加权误差(越低越好): RTN={wr:.1f}  AWQ={wa:.1f}")
    print(f"  AWQ 相对 RTN 降低: {(1 - wa / wr):.2%}")


def step6_group_sweep():
    """步骤6:group_size 扫描——组越小误差越低,per-channel 最差。"""
    print("\n[步骤6] group_size 扫描(RTN INT4 相对误差)")
    torch.manual_seed(2)
    w = torch.randn(256, 512)
    for gs in (64, 128, 256, -1):
        w_dq = fake_quant_groupwise(w, 4, gs, False)
        rel = (w - w_dq).norm().item() / w.norm().item()
        tag = "per-channel" if gs == -1 else f"group={gs}"
        print(f"  {tag:>12}: 相对误差 = {rel:.4%}")


def step7_calib_pipeline():
    """步骤7:端到端校准流水线(GPTQ/AWQ 真实代码路径,非仅数学函数)。

    覆盖生产链路:collect_calib_stats(hook 收集逐层 H/激活)→
    apply_quantization(按 method 路由到 gptq/awq)→ compression_report。
    用合成"激活"当校准输入(TinyLM 直接吃 float 向量),CPU 秒级、无下载。
    """
    print("\n[步骤7] 端到端校准流水线:collect_calib_stats -> apply_quantization")
    torch.manual_seed(3)
    x = torch.randn(2, 256)                        # 评测输入,量化前后对比用

    for method in ("gptq", "awq"):
        model = TinyLM()                            # 每种方法用全新模型
        y_fp = model(x).detach()                    # 量化前输出
        cfg = QuantConfig(n_bits=4, group_size=128, symmetric=False,
                          method=method, skip=("lm_head",))

        # 1) 定位待量化层名(与 apply 内部同一套 skip 逻辑)
        names = target_linear_names(model, cfg)
        # 2) 用合成校准输入跑一次前向,hook 增量累积每层 H / 激活幅度
        calib = torch.randn(16, 256)                # 16 条"激活样本"
        stats = collect_calib_stats(model, calib, names, batch_size=8)
        # 3) 按 method 路由:gptq 用 H,awq 用 act_scales,就地替换 Linear
        model, n = apply_quantization(model, cfg, calib_stats=stats)
        rep = compression_report(model)

        y_q = model(x).detach()
        diff = (y_fp - y_q).norm().item() / y_fp.norm().item()
        print(f"  [{method}] 校准层数={len(stats)}, 替换={n}, "
              f"等效{rep['effective_bits']}bit, 输出相对差异={diff:.4%}")
        # 抽一层确认统计维度正确(H 为 in×in,act 为 in)
        h = stats["down_proj"]["H"]
        a = stats["down_proj"]["act_scales"]
        print(f"       down_proj: H{tuple(h.shape)} act{tuple(a.shape)} "
              f"-> {type(model.down_proj).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("LowBitSparse CPU 冒烟演示(无需 GPU / 模型下载)")
    print("=" * 60)
    step1_rtn_math()
    step2_apply_and_report()
    step3_forward_compare()
    step4_gptq_vs_rtn()
    step5_awq_vs_rtn()
    step6_group_sweep()
    step7_calib_pipeline()
    print("\n[OK] 全部跑通 —— RTN/GPTQ/AWQ 数学与校准流水线、"
          "就地替换、压缩统计、group 扫描。")
