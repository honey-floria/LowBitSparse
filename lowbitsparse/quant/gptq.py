"""GPTQ 权重量化(Hessian 校准 + 逐列误差补偿)。

思路(Frantar et al. 2022):RTN 独立地就近取整每个权重,忽略了"量化某列会
牵动输出、而这一偏差本可由后续列补偿"这一事实。GPTQ 用校准集统计的二阶信息
H = XiXi^T(Xi 为该层输入激活),按列依次量化,每量化一列就把它的量化误差
按 H^-1 加权地补偿到尚未处理的列上,从而在相同位宽下显著降低输出误差。

本实现为"伪量化":返回反量化后的权重(FP16),便于纯 PyTorch 评测精度,
与 RTN/AWQ 共用 primitives 里的定点数学。块大小取 group_size,保证
分组边界与分块边界对齐,每组 scale 在该组起始处按当前(已补偿)权重计算。
"""
import torch

from .primitives import find_qparams, quant_dequant


def _cholesky_inverse_upper(H: torch.Tensor, percdamp: float):
    """对 Hessian 做阻尼后求其逆的上三角 Cholesky 因子 Hinv。

    阻尼:对角线加 percdamp·mean(diag),压制病态/近奇异,保证可分解。
    若仍失败则逐步加大阻尼重试(最多几次),换取数值稳健。
    返回上三角 Hinv,使 Hinv[i,i] 恰为"量化第 i 列的误差归一化因子"。
    """
    in_f = H.shape[0]
    diag = torch.arange(in_f, device=H.device)
    damp = percdamp * torch.mean(torch.diag(H)).clamp(min=1e-8)
    for _ in range(5):
        Hd = H.clone()
        Hd[diag, diag] += damp
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True)
        except Exception:
            damp *= 10          # 病态:加大阻尼再试
    raise RuntimeError("GPTQ: Cholesky 反复失败,Hessian 病态严重")


def gptq_quantize_weight(weight: torch.Tensor, H: torch.Tensor, cfg,
                         percdamp: float = 0.01) -> torch.Tensor:
    """用 GPTQ 对单层 2D 权重做伪量化。

    参数:
        weight: [out_features, in_features] 的 nn.Linear.weight。
        H:      [in_features, in_features] 校准 Hessian(≈ Σ x xᵀ),来自 calibration。
        cfg:    QuantConfig(用 n_bits/group_size/symmetric)。
        percdamp: 对角阻尼比例。
    返回:
        反量化后的权重(与 weight 同形状同 dtype)。
    """
    orig_dtype = weight.dtype
    W = weight.detach().float().clone()
    out_f, in_f = W.shape
    gs = in_f if cfg.group_size in (-1, None) else cfg.group_size
    H = H.detach().float().clone()

    # 死通道:对应输入恒 0(H 对角为 0),量化后置 0,并把 H 对角设 1 免奇异
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    Hinv = _cholesky_inverse_upper(H, percdamp)   # 上三角误差补偿因子
    Q = torch.zeros_like(W)                        # 存反量化结果

    # 逐列处理,按 group 边界刷新 scale/zero;误差沿 Hinv 补偿到右侧未处理列
    scale = zero = qmax = None
    for i in range(in_f):
        if i % gs == 0:      # 进入新组:用当前(已补偿)权重的该组求 qparams
            j = min(i + gs, in_f)
            g = W[:, i:j]
            scale, zero, qmax = find_qparams(g, cfg.n_bits, cfg.symmetric)
        w_col = W[:, i:i + 1]
        q_col = quant_dequant(w_col, scale, zero, qmax, cfg.symmetric)
        Q[:, i:i + 1] = q_col
        d = Hinv[i, i]
        err = (w_col - q_col) / d                 # 归一化误差
        # 把误差按 Hinv 行加权补偿到第 i+1.. 列(仅右侧,保证因果顺序)
        W[:, i + 1:] -= err * Hinv[i, i + 1:].unsqueeze(0)

    Q[:, dead] = 0.0
    return Q.to(orig_dtype)
