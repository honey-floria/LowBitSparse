"""RTN(Round-To-Nearest)权重量化。

对每个分组独立求量化参数(scale/zero),把浮点权重映射到低位整数网格再映射回来,
即"伪量化(fake-quant)"——权重仍以 FP16 存,但数值已被量化误差污染,
从而能真实反映低 bit 对精度的影响,同时便于纯 PyTorch 实现、无需 INT kernel。

RTN 是最简 baseline:只看权重自身分布,不用任何校准数据。核心分组数学已抽到
`primitives.py`,本模块只保留对外入口,与 GPTQ/AWQ 共享同一套定点实现。
"""
import torch

from .primitives import fake_quant_groupwise


def _quantize_groupwise(w: torch.Tensor, n_bits: int, group_size: int,
                        symmetric: bool) -> torch.Tensor:
    """按行内分组做 RTN 量化-反量化(薄封装,实现见 primitives)。"""
    return fake_quant_groupwise(w, n_bits, group_size, symmetric)


def rtn_quantize_weight(weight: torch.Tensor, cfg) -> torch.Tensor:
    """RTN 量化对外入口:按 QuantConfig 对 2D 权重做伪量化。

    参数:
        weight: nn.Linear.weight,形状 [out_features, in_features]。
        cfg:    QuantConfig,用其 n_bits / group_size / symmetric。
    返回:
        反量化后的权重张量(伪量化),可直接回填进 Linear 层。
    """
    return fake_quant_groupwise(
        weight, n_bits=cfg.n_bits, group_size=cfg.group_size,
        symmetric=cfg.symmetric,
    )
