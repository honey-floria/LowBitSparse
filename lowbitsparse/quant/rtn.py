"""RTN(Round-To-Nearest)权重量化的核心数学。

对每个分组独立求量化参数(scale/zero),把浮点权重映射到低位整数网格再映射回来,
即"伪量化(fake-quant)"——权重仍以 FP16 存,但数值已被量化误差污染,
从而能真实反映低 bit 对精度的影响,同时便于纯 PyTorch 实现、无需 INT kernel。
"""
import torch


def _quantize_groupwise(w: torch.Tensor, n_bits: int, group_size: int,
                        symmetric: bool):
    """对权重矩阵按行内分组做 RTN 量化-反量化。

    参数:
        w:          权重张量,形状 [out_features, in_features]。
        n_bits:     位宽。
        group_size: 组大小,沿 in_features 分组;-1 表示整行一组(per-channel)。
        symmetric:  是否对称量化。
    返回:
        w_dq: 反量化后的权重(与 w 同形状同 dtype),即被量化误差污染的权重。

    实现要点:沿输入维把每行切成若干组,每组独立算 scale/zero,
    用 padding 处理 in_features 不能被 group_size 整除的情况,全程向量化。
    """
    out_features, in_features = w.shape
    orig_dtype = w.dtype
    w = w.float()                          # 升 float32 保证 round/除法数值稳定
    gs = in_features if group_size in (-1, None) else group_size

    # 若不能整除,右侧 pad 到 gs 的整数倍(pad 值先置 0,后面会被切掉)
    pad = (gs - in_features % gs) % gs
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    n_group = w.shape[1] // gs
    # reshape 成 [out_features, n_group, gs],最后一维即"组内"
    wg = w.reshape(out_features, n_group, gs)

    if symmetric:
        # 对称:量化范围 [-(2^(b-1)-1), 2^(b-1)-1],zero=0
        qmax = 2 ** (n_bits - 1) - 1
        # 每组取绝对值最大做尺度,避免溢出;clamp 防止全 0 组除零
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wg / scale), -qmax, qmax)   # 量化到整数网格
        wg_dq = q * scale                                       # 反量化
    else:
        # 非对称:范围 [0, 2^b-1],用组内 min/max 定 scale 和 zero
        qmax = 2 ** n_bits - 1
        w_min = wg.amin(dim=-1, keepdim=True)
        w_max = wg.amax(dim=-1, keepdim=True)
        scale = ((w_max - w_min) / qmax).clamp(min=1e-8)        # 每单位整数对应的浮点步长
        zero = torch.round(-w_min / scale)                      # 浮点 0 对应的整数码
        q = torch.clamp(torch.round(wg / scale) + zero, 0, qmax)
        wg_dq = (q - zero) * scale                              # 反量化回浮点

    # 复原形状并切掉 padding 列,转回原 dtype
    w_dq = wg_dq.reshape(out_features, -1)[:, :in_features]
    return w_dq.to(orig_dtype)


def rtn_quantize_weight(weight: torch.Tensor, cfg) -> torch.Tensor:
    """RTN 量化对外入口:按 QuantConfig 对 2D 权重做伪量化。

    参数:
        weight: nn.Linear.weight,形状 [out_features, in_features]。
        cfg:    QuantConfig,用其 n_bits / group_size / symmetric。
    返回:
        反量化后的权重张量(伪量化),可直接回填进 Linear 层。
    """
    return _quantize_groupwise(
        weight, n_bits=cfg.n_bits, group_size=cfg.group_size,
        symmetric=cfg.symmetric,
    )
