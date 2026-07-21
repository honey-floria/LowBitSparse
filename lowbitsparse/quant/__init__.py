"""权重量化模块(M1: RTN / GPTQ / AWQ)。"""
from .config import QuantConfig
from .rtn import rtn_quantize_weight
from .gptq import gptq_quantize_weight
from .awq import awq_quantize_weight
from .fake_linear import FakeQuantLinear
from .apply import (
    apply_quantization, compression_report, target_linear_names)
from .calibration import get_calib_inputs, collect_calib_stats

__all__ = [
    "QuantConfig",
    "rtn_quantize_weight",
    "gptq_quantize_weight",
    "awq_quantize_weight",
    "FakeQuantLinear",
    "apply_quantization",
    "compression_report",
    "target_linear_names",
    "get_calib_inputs",
    "collect_calib_stats",
]
