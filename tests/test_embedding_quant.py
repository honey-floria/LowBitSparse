"""embedding 量化消融的单元测试。

验证:
1) FakeQuantEmbedding 量化-反量化 round-trip:shape/dtype 保持,误差有限;
2) 绑定(tied)模型:量化后 embed 与 lm_head 仍共享同一 buffer(未拆散);
3) compression_report 对绑定共享矩阵去重,且量化 embedding 后压缩比提升;
4) forward 输出 shape 正确、数值有限。
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lowbitsparse.quant import (
    QuantConfig, apply_quantization, compression_report, FakeQuantEmbedding,
    FakeQuantLinear)
from lowbitsparse.quant.fake_embedding import FakeQuantEmbedding as FQE


class TiedTinyLM(nn.Module):
    """迷你 LM,lm_head 与 embed_tokens 绑定(模拟 Qwen2.5-0.5B)。"""

    def __init__(self, vocab=320, d=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight   # 绑定

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, m):
        self.embed_tokens = m

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed_tokens(ids)))


def test_fake_quant_embedding_roundtrip():
    """FakeQuantEmbedding 保持 shape/dtype,INT8 误差小。"""
    emb = nn.Embedding(256, 128)
    fq = FQE(emb, n_bits=8, group_size=64, symmetric=False)
    assert fq.weight.shape == emb.weight.shape
    assert fq.weight.dtype == emb.weight.dtype
    rel = (emb.weight.data - fq.weight).norm() / emb.weight.data.norm()
    assert rel < 0.01


def test_tied_weight_still_shared_after_quant():
    """绑定模型量化后,embed 与 lm_head 仍指向同一 buffer 对象。"""
    torch.manual_seed(0)
    model = TiedTinyLM()
    cfg = QuantConfig(n_bits=4, group_size=64, quant_embedding=True,
                      skip=("lm_head",))
    model, n = apply_quantization(model, cfg)
    assert isinstance(model.embed_tokens, FakeQuantEmbedding)
    assert isinstance(model.lm_head, FakeQuantLinear)
    # 关键:同一 buffer,绑定未被拆散
    assert model.embed_tokens.weight is model.lm_head.weight


def test_quant_embedding_improves_compression():
    """量化 embedding 后压缩比高于仅量化 Linear(embedding 不再是 FP16 地板)。"""
    torch.manual_seed(0)
    base = QuantConfig(n_bits=4, group_size=64, skip=("lm_head",))
    m1 = TiedTinyLM()
    apply_quantization(m1, base)
    r1 = compression_report(m1)

    m2 = TiedTinyLM()
    apply_quantization(m2, QuantConfig(n_bits=4, group_size=64,
                                       quant_embedding=True, skip=("lm_head",)))
    r2 = compression_report(m2)
    assert r2["size_mb"] < r1["size_mb"]        # 量化 embedding 更小
    assert r2["quant_weights"] > r1["quant_weights"]


def test_quant_embedding_forward_ok():
    """量化 embedding 后 forward 输出 shape 正确、数值有限。"""
    torch.manual_seed(1)
    model = TiedTinyLM()
    apply_quantization(model, QuantConfig(n_bits=8, group_size=64,
                                          quant_embedding=True, skip=("lm_head",)))
    ids = torch.randint(0, 320, (2, 16))
    out = model(ids)
    assert out.shape == (2, 16, 320)
    assert torch.isfinite(out).all()
