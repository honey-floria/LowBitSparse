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
    collect_calib_stats, target_linear_names, free_calib_stats)
from lowbitsparse.quant.rtn import rtn_quantize_weight
from lowbitsparse.quant.primitives import fake_quant_groupwise
from lowbitsparse.sparse import (
    SparseConfig, build_sparse_attention_mask, install_sparse_attention,
    sparse_density)


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
    """步骤5:AWQ 激活感知缩放 vs RTN(裁剪搜索默认关,见下方说明)。"""
    print("\n[步骤5] AWQ(激活感知逐通道缩放) vs RTN")
    torch.manual_seed(1)
    W = torch.randn(64, 256)
    W[:, ::16] *= 6.0                      # 注入权重离群
    act = torch.rand(256) + 0.05
    act[:4] *= 50.0                        # 4 个超大激活通道
    cfg = QuantConfig(n_bits=3, group_size=256, symmetric=False, method="awq")

    def werr(W_dq):
        return ((W - W_dq) * act.unsqueeze(0)).pow(2).sum().item()

    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    W_scale = awq_quantize_weight(W, act, cfg)            # 默认:仅缩放
    W_clip = awq_quantize_weight(W, act, cfg, clip_search=True)  # 裁剪(默认关)
    wr, ws, wc = werr(W_rtn), werr(W_scale), werr(W_clip)
    print(f"  INT3 激活加权误差(越低越好): RTN={wr:.1f}  AWQ缩放={ws:.1f}")
    print(f"  缩放 vs RTN 降低: {(1 - ws / wr):.2%}")
    # 裁剪:合成权重 MSE 上"看似"更优,但 M1-h A100 实测真实 PPL 全面恶化,故默认关
    print(f"  [裁剪搜索:默认关] 合成加权误差={wc:.1f}(权重空间代理下略降,"
          f"但真实 PPL 恶化——代理不忠实,见 OPTIMIZATION M1-h)")


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


class TiedTinyLM(nn.Module):
    """迷你 LM,lm_head 与 embed_tokens 绑定(模拟 Qwen2.5-0.5B 的 tied embedding)。"""

    def __init__(self, vocab=1000, d=256):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight   # 绑定:同一张量

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, m):
        self.embed_tokens = m

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed_tokens(ids)))


class TinySelfAttention(nn.Module):
    """最小可运行自注意力,用于稀疏 mask 冒烟。"""

    def __init__(self, d=128, n_heads=4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, seq_len, self.d)
        return self.o_proj(out)


class TinySparseLM(nn.Module):
    """带 `_update_causal_mask` 的最小 causal LM,用于 hook 冒烟。"""

    def __init__(self, d=128, vocab=256, n_heads=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.attn = TinySelfAttention(d=d, n_heads=n_heads)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def _update_causal_mask(self, attention_mask, inputs_embeds,
                            cache_position=None, past_key_values=None,
                            output_attentions=False):
        q_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        mask = torch.zeros((1, 1, q_len, q_len), device=device, dtype=dtype)
        future = torch.triu(torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1)
        return mask.masked_fill(future.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        mask = self._update_causal_mask(None, x)
        x = self.attn(x, attention_mask=mask)
        return self.lm_head(x)


def step8_embedding_quant():
    """步骤8:embedding 量化消融——绑定权重量化一次,embed/lm_head 共享,压缩比升。

    演示突破"embedding FP16 地板":真实 0.5B 上 embedding 占量化后体积 ~42%,
    量化它是唯一能把压缩比从 ~2.1x 推到 3x+ 的方向。绑定检测确保不拆散共享矩阵。
    """
    from lowbitsparse.quant import FakeQuantEmbedding

    print("\n[步骤8] embedding 量化消融(绑定权重,embed/lm_head 共享)")
    torch.manual_seed(4)
    ids = torch.randint(0, 1000, (2, 16))

    # 基线:只量化 Linear,embedding 保持 FP16(skip lm_head)
    m0 = TiedTinyLM()
    apply_quantization(m0, QuantConfig(n_bits=4, group_size=128, skip=("lm_head",)))
    r0 = compression_report(m0)

    # 消融:量化 embedding INT8(绑定的 lm_head 一并,共享同一反量化权重)
    m1 = TiedTinyLM()
    apply_quantization(m1, QuantConfig(n_bits=4, group_size=128, skip=("lm_head",),
                                       quant_embedding=True, embedding_bits=8))
    r1 = compression_report(m1)

    shared = m1.embed_tokens.weight is m1.lm_head.weight
    print(f"  基线(emb FP16):  等效 {r0['effective_bits']} bit, 体积 {r0['size_mb']} MB")
    print(f"  消融(emb INT8):  等效 {r1['effective_bits']} bit, 体积 {r1['size_mb']} MB")
    print(f"  embed 类型: {type(m1.embed_tokens).__name__}, "
          f"lm_head 类型: {type(m1.lm_head).__name__}")
    print(f"  绑定保持(embed.weight is lm_head.weight): {shared}")
    out = m1(ids)
    print(f"  量化后 forward 输出 shape: {tuple(out.shape)}, 数值有限: "
          f"{bool(torch.isfinite(out).all())}")


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
        # 4) 量化后释放校准统计:H 占显存大头,评测前清掉压低峰值。
        freed = free_calib_stats(stats)
        print(f"       释放校准统计 {freed} MB(真实 0.5B 上每层 H 可达 ~95MB," 
              f"累计数 GB);清空后 stats 层数={len(stats)}")


def step9_sparse_attention():
    """步骤9:稀疏注意力 mask / hook 演示。"""
    print("\n[步骤9] 稀疏注意力(mask + hook) vs 全因果")
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (2, 8))

    base = TinySparseLM()
    y_full = base(ids).detach()

    modes = [
        ("sliding_window", SparseConfig(mode="sliding_window", window_size=4)),
        ("streaming_llm", SparseConfig(mode="streaming_llm", window_size=4, sink_size=2)),
        ("block_sparse", SparseConfig(mode="block_sparse", sink_size=0, block_size=2, block_lookback=1)),
    ]
    for name, cfg in modes:
        density = sparse_density(8, 8, cfg)
        mask = build_sparse_attention_mask(8, 8, cfg, dtype=torch.float32)
        model = TinySparseLM()
        model.load_state_dict(base.state_dict())
        patch = install_sparse_attention(model, cfg)
        y_sparse = model(ids).detach()
        patch.restore()
        diff = (y_full - y_sparse).norm().item() / y_full.norm().item()
        print(f"  {name:>14}: density={density['density']:.3f}, sparsity={density['sparsity']:.3f}, "
              f"mask shape={tuple(mask.shape)}, 输出差异={diff:.4%}")


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
    step8_embedding_quant()
    step9_sparse_attention()
    print("\n[OK] 全部跑通 —— RTN/GPTQ/AWQ 数学与校准流水线、就地替换、"
          "压缩统计、group 扫描、embedding 量化消融、稀疏注意力 mask/hook。")
