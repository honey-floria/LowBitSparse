"""group_size 与 per-channel vs per-group 的性质测试(CPU、秒级)。

验证量化粒度与误差的关系:
- 组越小(scale 越局部),量化误差越低;per-channel(整行一组)误差最大。
这是 group_size 扫描实验背后的核心直觉,也为验收表格提供正确性锚点。
"""
import pytest

torch = pytest.importorskip("torch")

from lowbitsparse.quant.primitives import fake_quant_groupwise


def _rel_err(w, w_dq):
    return (w - w_dq).norm().item() / w.norm().item()


def test_smaller_group_lower_error():
    """group_size 越小,RTN 量化误差越小:64 < 128 < 256。"""
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    errs = {}
    for gs in (64, 128, 256):
        errs[gs] = _rel_err(w, fake_quant_groupwise(w, 4, gs, False))
    assert errs[64] < errs[128] < errs[256]


def test_per_channel_worst():
    """per-channel(group=-1,整行一组)误差应不低于 per-group=256。"""
    torch.manual_seed(1)
    w = torch.randn(128, 1024)
    e_pc = _rel_err(w, fake_quant_groupwise(w, 4, -1, False))
    e_pg = _rel_err(w, fake_quant_groupwise(w, 4, 256, False))
    assert e_pc >= e_pg


def test_group_matches_channel_when_equal():
    """group_size == in_features 时应等价于 per-channel(数值一致)。"""
    torch.manual_seed(2)
    w = torch.randn(32, 128)
    a = fake_quant_groupwise(w, 4, 128, False)
    b = fake_quant_groupwise(w, 4, -1, False)   # group=-1 → 整行一组 = 128
    assert torch.allclose(a, b, atol=1e-6)
