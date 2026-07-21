"""GPTQ 量化的单元测试(CPU、合成数据,秒级)。

验证:
1) 输出 shape/dtype 不变、数值有限;
2) 在"相关输入"下,GPTQ 的 Hessian 加权误差 ≤ RTN——这是 GPTQ 的核心收益,
   因为它直接最小化 H 加权的量化残差;随机不相关输入下 H≈对角,二者会接近。
"""
import pytest

torch = pytest.importorskip("torch")

from lowbitsparse.quant.config import QuantConfig
from lowbitsparse.quant.gptq import gptq_quantize_weight
from lowbitsparse.quant.primitives import fake_quant_groupwise


def _corr_hessian(in_f, n_tok=512, seed=0):
    """构造相关输入的 Hessian H=XᵀX(通道间相关 → H 非对角)。"""
    torch.manual_seed(seed)
    base = torch.randn(n_tok, in_f)
    # 加入列间相关:每列混入相邻列,制造非对角 H
    mix = base + 0.8 * torch.roll(base, 1, dims=1)
    return mix.t() @ mix, mix


def test_shape_dtype_finite():
    """GPTQ 输出与输入同形状/同 dtype,且数值有限。"""
    torch.manual_seed(0)
    W = torch.randn(64, 128, dtype=torch.float16)
    H, _ = _corr_hessian(128)
    cfg = QuantConfig(n_bits=4, group_size=64, symmetric=False, method="gptq")
    W_dq = gptq_quantize_weight(W, H, cfg)
    assert W_dq.shape == W.shape and W_dq.dtype == W.dtype
    assert torch.isfinite(W_dq).all()


def test_gptq_beats_rtn_weighted():
    """相关输入下,GPTQ 的 H 加权误差应不劣于 RTN(通常更优)。"""
    in_f, out_f = 128, 96
    H, X = _corr_hessian(in_f, seed=1)
    torch.manual_seed(2)
    W = torch.randn(out_f, in_f)
    cfg = QuantConfig(n_bits=3, group_size=128, symmetric=False, method="gptq")

    W_gptq = gptq_quantize_weight(W, H, cfg)
    W_rtn = fake_quant_groupwise(W, cfg.n_bits, cfg.group_size, cfg.symmetric)

    # 用真实输出误差衡量:||X Wᵀ - X W_qᵀ||_F
    err_gptq = ((X @ (W - W_gptq).t()).norm()).item()
    err_rtn = ((X @ (W - W_rtn).t()).norm()).item()
    assert err_gptq <= err_rtn * 1.02   # 容 2% 抖动,应更优或持平


def test_dead_channel_zeroed():
    """恒 0 输入通道(H 对角为 0)对应权重列应被安全置 0、不产生 NaN。"""
    in_f, out_f = 64, 32
    H, _ = _corr_hessian(in_f, seed=3)
    H[10, :] = 0.0
    H[:, 10] = 0.0                     # 第 10 通道死亡
    W = torch.randn(out_f, in_f)
    cfg = QuantConfig(n_bits=4, group_size=64, symmetric=False, method="gptq")
    W_dq = gptq_quantize_weight(W, H, cfg)
    assert torch.isfinite(W_dq).all()
    assert torch.allclose(W_dq[:, 10], torch.zeros(out_f), atol=1e-5)
