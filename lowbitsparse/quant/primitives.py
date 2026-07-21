"""量化原语:被 RTN / GPTQ / AWQ 复用的分组量化数学。

把"求 scale/zero → 量化到整数网格 → 反量化"这三件事抽成小函数,
让三种量化算法只在"如何决定要量化哪块权重、按什么顺序、要不要缩放"上有差异,
底层的定点数学保持单一实现,避免重复与不一致。

约定:
- 权重形状统一为 [out_features, in_features],沿 in_features(列)分组。
- 全程用 float32 计算,round/除法数值更稳,末尾再转回原 dtype。
"""
import torch


def find_qparams(x: torch.Tensor, n_bits: int, symmetric: bool):
    """对最后一维求量化参数 scale/zero(qmax 一并返回,便于复用)。

    参数:
        x:         任意形状张量,沿最后一维求 min/max(即"组内")。
        n_bits:    位宽。
        symmetric: True=对称(zero=0,范围 ±(2^(b-1)-1));False=非对称(范围 [0,2^b-1])。
    返回:
        (scale, zero, qmax):scale/zero 形状为 x 去掉最后一维再 keepdim;
        对称时 zero 为全 0 张量(占位,保持接口一致)。
    """
    if symmetric:
        qmax = 2 ** (n_bits - 1) - 1
        # 组内绝对值最大定尺度;clamp 防全 0 组除零
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        zero = torch.zeros_like(scale)          # 对称无 zero,给全 0 占位
        return scale, zero, qmax
    qmax = 2 ** n_bits - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp(min=1e-8)   # 每单位整数的浮点步长
    zero = torch.round(-x_min / scale)                 # 浮点 0 对应的整数码
    return scale, zero, qmax


def quant_dequant(x: torch.Tensor, scale, zero, qmax, symmetric: bool):
    """用给定 scale/zero 把 x 量化到整数网格再反量化(伪量化一趟)。

    对称:q=clamp(round(x/scale), -qmax, qmax),x_dq=q*scale。
    非对称:q=clamp(round(x/scale)+zero, 0, qmax),x_dq=(q-zero)*scale。
    scale/zero 会按广播规则作用到 x 的最后一维(组内共享)。
    """
    if symmetric:
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return q * scale
    q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
    return (q - zero) * scale


def fake_quant_groupwise(w: torch.Tensor, n_bits: int, group_size: int,
                         symmetric: bool) -> torch.Tensor:
    """对 2D 权重按行内分组做 RTN 伪量化(供 RTN/AWQ 复用)。

    参数:
        w:          [out_features, in_features]。
        n_bits:     位宽。
        group_size: 组大小,沿 in_features 分组;-1/None 表示整行一组(per-channel)。
        symmetric:  是否对称量化。
    返回:
        反量化后的权重(与 w 同形状同 dtype)。

    非整除时右侧 pad 到 group_size 整数倍(pad 值置 0,反量化后切掉),全程向量化。
    """
    out_features, in_features = w.shape
    orig_dtype = w.dtype
    w = w.float()
    gs = in_features if group_size in (-1, None) else group_size

    pad = (gs - in_features % gs) % gs
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    n_group = w.shape[1] // gs
    wg = w.reshape(out_features, n_group, gs)          # [out, n_group, gs]

    scale, zero, qmax = find_qparams(wg, n_bits, symmetric)
    wg_dq = quant_dequant(wg, scale, zero, qmax, symmetric)

    w_dq = wg_dq.reshape(out_features, -1)[:, :in_features]   # 复原并切掉 padding
    return w_dq.to(orig_dtype)
