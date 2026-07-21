"""AWQ 量化的单元测试(CPU、合成数据,秒级)。

验证:
1) 输出 shape/dtype 不变、数值有限;
2) AWQ 的激活加权误差 ≤ RTN:因为 ratio=0 的网格点恰好退化为 RTN,
   AWQ 在网格上取最优,故必不劣于 RTN(这是 AWQ 的构造性保证);
3) 激活幅度悬殊时 AWQ 相对 RTN 有明显改善。
"""
import pytest

torch = pytest.importorskip("torch")

from lowbitsparse.quant.config import QuantConfig
from lowbitsparse.quant.awq import awq_quantize_weight
from lowbitsparse.quant.primitives import fake_quant_groupwise


def _weighted_err(W, W_dq, act):
    """激活加权量化误差 ||a·(W - W_dq)||_F²(AWQ 的优化目标代理)。"""
    return ((W - W_dq) * act.unsqueeze(0)).pow(2).sum().item()


def test_shape_dtype_finite():
    torch.manual_seed(0)
    W = torch.randn(64, 128, dtype=torch.float16)
    act = torch.rand(128) + 0.1
    cfg = QuantConfig(n_bits=4, group_size=64, symmetric=False, method="awq")
    W_dq = awq_quantize_weight(W, act, cfg)
    assert W_dq.shape == W.shape and W_dq.dtype == W.dtype
    assert torch.isfinite(W_dq).all()


def test_awq_not_worse_than_rtn():
    """AWQ 加权误差不劣于 RTN(ratio=0 网格点即 RTN)。"""
    torch.manual_seed(1)
    W = torch.randn(96, 256)
    # 激活幅度悬殊:少数通道极大,凸显 AWQ 保护效果
    act = torch.rand(256) + 0.05
    act[:8] *= 30.0
    cfg = QuantConfig(n_bits=3, group_size=128, symmetric=False, method="awq")

    W_awq = awq_quantize_weight(W, act, cfg)
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)
    assert _weighted_err(W, W_awq, act) <= _weighted_err(W, W_rtn, act) + 1e-6


def test_awq_helps_on_skewed_act():
    """激活极不均衡时,AWQ 应带来可见改善(严格更优)。"""
    torch.manual_seed(2)
    W = torch.randn(64, 256)
    act = torch.rand(256) + 0.05
    act[:4] *= 50.0                    # 4 个超大激活通道
    cfg = QuantConfig(n_bits=3, group_size=256, symmetric=False, method="awq")

    e_awq = _weighted_err(W, awq_quantize_weight(W, act, cfg), act)
    e_rtn = _weighted_err(
        W, fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric), act)
    assert e_awq < e_rtn               # 悬殊激活下应严格更优
