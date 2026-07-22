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

from .primitives import fake_quant_groupwise, fake_quant_groupwise_autoclip


def awq_quantize_weight(weight: torch.Tensor, act_scales: torch.Tensor,
                        cfg, n_grid: int = 20, clip_search: bool = True,
                        n_clip: int = 20) -> torch.Tensor:
    """用 AWQ 对单层 2D 权重做伪量化。

    两阶段(对齐 AWQ 论文):
      1) 缩放搜索:按激活幅度搜每通道保护缩放 s,最小化加权误差(保护大激活通道);
      2) 裁剪搜索(auto_clip):在最佳缩放后的权重上,逐组搜范围收缩系数 α,
         牺牲少数离群权重换 bulk 网格分辨率。两阶段而非联合网格,省开销。

    参数:
        weight:      [out_features, in_features] 的 Linear 权重。
        act_scales:  [in_features] 每条输入通道在校准集上的激活均值绝对值。
        cfg:         QuantConfig(用 n_bits/group_size/symmetric)。
        n_grid:      缩放 ratio 网格点数,默认 20(ratio ∈ {0, 0.05, …, 1.0})。
        clip_search: 是否启用第 2 阶段裁剪搜索(默认 True;False 退回纯缩放 AWQ)。
        n_clip:      裁剪 α 网格点数,默认 20。
    返回:
        反量化后的权重(与 weight 同形状同 dtype)。
    """
    orig_dtype = weight.dtype
    W = weight.detach().float().clone()
    a = act_scales.float().to(W.device).clamp(min=1e-6)   # [in_features]
    a_norm = a / a.mean()          # 归一化,使"1"对应平均通道

    # === 阶段 1:缩放搜索 ===
    # 参考基准:纯 RTN 的加权 Frobenius 误差 ||a·(W - Q(W))||_F²
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    best_err = float("inf")
    best_s = torch.ones_like(a)    # 记录最佳缩放,供阶段 2 复用
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
            best_s = s
            best_W_dq = W_dq

    if not clip_search:
        return best_W_dq.to(orig_dtype)

    # === 阶段 2:在最佳缩放后的权重上逐组搜裁剪 α ===
    s = best_s
    s_inv = 1.0 / s
    Ws = W * s.unsqueeze(0)
    Ws_dq = fake_quant_groupwise_autoclip(
        Ws, cfg.n_bits, cfg.group_size, cfg.symmetric, n_clip=n_clip)
    W_dq_clip = Ws_dq * s_inv.unsqueeze(0)

    # 裁剪 grid 含 α=1,理论上不劣;仍以加权误差为准做一次保底比较,取更优者
    err_clip = ((W - W_dq_clip) * a.unsqueeze(0)).pow(2).sum().item()
    if err_clip < best_err:
        best_W_dq = W_dq_clip
    return best_W_dq.to(orig_dtype)
