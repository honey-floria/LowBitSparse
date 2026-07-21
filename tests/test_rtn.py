"""RTN 量化数学的单元测试。

验证三条性质:
1) 量化-反量化后 shape/dtype 不变;
2) 量化误差在合理容差内(round-trip);
3) 位宽越高,误差越小(单调性)——这是量化正确性的关键信号。

依赖 torch,在 Colab 上 `pytest tests/ -v` 运行。
"""
import pytest

torch = pytest.importorskip("torch")   # 无 torch 环境自动跳过,不算失败

from lowbitsparse.quant.config import QuantConfig
from lowbitsparse.quant.rtn import rtn_quantize_weight


def _rel_err(w, w_dq):
    """相对误差 = ||w - w_dq|| / ||w||,衡量量化损伤。"""
    return (w - w_dq).norm().item() / w.norm().item()


def test_shape_dtype_preserved():
    """反量化结果与输入同形状、同 dtype。"""
    w = torch.randn(64, 256, dtype=torch.float16)
    cfg = QuantConfig(n_bits=4, group_size=128, symmetric=False)
    w_dq = rtn_quantize_weight(w, cfg)
    assert w_dq.shape == w.shape
    assert w_dq.dtype == w.dtype


def test_higher_bits_lower_error():
    """位宽越高误差越小:INT8 < INT4 < INT3。"""
    torch.manual_seed(0)
    w = torch.randn(128, 512)
    errs = {}
    for b in (3, 4, 8):
        cfg = QuantConfig(n_bits=b, group_size=128, symmetric=False)
        errs[b] = _rel_err(w, rtn_quantize_weight(w, cfg))
    assert errs[8] < errs[4] < errs[3]


def test_int8_error_small():
    """INT8 量化误差应很小(健全性),相对误差 < 1%。"""
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    cfg = QuantConfig(n_bits=8, group_size=128, symmetric=False)
    assert _rel_err(w, rtn_quantize_weight(w, cfg)) < 0.01


def test_non_divisible_group():
    """in_features 不能被 group_size 整除时,padding 路径应正常工作。"""
    w = torch.randn(32, 200)   # 200 不是 128 的整数倍
    cfg = QuantConfig(n_bits=4, group_size=128, symmetric=False)
    w_dq = rtn_quantize_weight(w, cfg)
    assert w_dq.shape == w.shape
    assert torch.isfinite(w_dq).all()


def test_symmetric_runs():
    """对称量化路径可运行且误差有限。"""
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    cfg = QuantConfig(n_bits=4, group_size=64, symmetric=True)
    w_dq = rtn_quantize_weight(w, cfg)
    assert torch.isfinite(w_dq).all()
    assert _rel_err(w, w_dq) < 0.3
