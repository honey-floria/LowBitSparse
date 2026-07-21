"""权重量化模块(M1: RTN;后续 GPTQ / AWQ)。"""
from .config import QuantConfig
from .rtn import rtn_quantize_weight
from .fake_linear import FakeQuantLinear
from .apply import apply_quantization, compression_report

__all__ = [
    "QuantConfig",
    "rtn_quantize_weight",
    "FakeQuantLinear",
    "apply_quantization",
    "compression_report",
]
