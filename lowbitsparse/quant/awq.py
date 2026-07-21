"""AWQ 激活感知权重量化(Activation-aware Weight Quantization)。

思路(Lin et al. 2023):权重量化的精度损失在各输入通道间分布极不均匀——
激活幅度大的通道量化误差影响更大。AWQ 对每个输入通道搜索一个"保护缩放"s,
量化前把权重 W 该通道乘以 s、激活除以 s,量化后再还原:
    伪量化权重 = Q(W · diag(s)) · diag(1/s)
这样重要通道(大激活)的权重范围缩小,量化粒度更细,精度损失更低。
s 按激活幅度 a(per-input-channel 均值绝对值)的 ratio 次幂定义:
    s = (a / mean(a))^ratio, ratio ∈ [0,1]
ratio=0 → s≡1 → 退化为 RTN,ratio=1 → 完全按激活幅度缩放;
在 grid 上搜索令"加权量化误差最小"的 ratio,从而自适应选出最佳保护力度。
"""
import torch

from .primitives import fake_quant_groupwise


def awq_quantize_weight(weight: torch.Tensor, act_scales: torch.Tensor,
                        cfg, n_grid: int = 20) -> torch.Tensor:
    """用 AWQ 对单层 2D 权重做伪量化。

    参数:
        weight:     [out_features, in_features] 的 Linear 权重。
        act_scales: [in_features] 每条输入通道在校准集上的激活均值绝对值。
        cfg:        QuantConfig(用 n_bits/group_size/symmetric)。
        n_grid:     ratio 网格点数,默认 20(ratio ∈ {0, 0.05, …, 1.0})。
    返回:
        反量化后的权重(与 weight 同形状同 dtype)。
    """
    orig_dtype = weight.dtype
    W = weight.detach().float().clone()
    a = act_scales.float().to(W.device).clamp(min=1e-6)   # [in_features]
    a_norm = a / a.mean()          # 归一化,使"1"对应平均通道

    # 参考基准:纯 RTN 的加权 Frobenius 误差 ||a·(W - Q(W))||_F²
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    best_err = float("inf")
    best_W_dq = W_rtn

    for k in range(n_grid + 1):
        ratio = k / n_grid          # ratio ∈ [0.0, …, 1.0]
        s = a_norm.pow(ratio)       # [in_features] 保护缩放向量
        s_inv = 1.0 / s

        # 缩放权重后量化,再还原(等价于"量化后乘以 diag(1/s)")
        Ws = W * s.unsqueeze(0)                # 每列乘以对应通道 s
        Ws_dq = fake_quant_groupwise(
            Ws, cfg.n_bits, cfg.group_size, cfg.symmetric)
        W_dq = Ws_dq * s_inv.unsqueeze(0)     # 还原:每列除以 s

        # 加权误差:按激活幅度加权的量化残差,近似输出 MSE 的代理指标
        err = ((W - W_dq) * a.unsqueeze(0)).pow(2).sum().item()
        if err < best_err:
            best_err = err
            best_W_dq = W_dq

    return best_W_dq.to(orig_dtype)
