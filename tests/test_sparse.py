"""稀疏注意力 mask 与 hook 的单元测试(CPU、纯 torch)。"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lowbitsparse.sparse import (
    SparseConfig, build_sparse_attention_mask, install_sparse_attention,
    sparse_density, sparse_visibility)


def test_sliding_window_visibility():
    cfg = SparseConfig(mode="sliding_window", window_size=3)
    vis = sparse_visibility(4, 6, cfg)
    expected = torch.tensor([
        [True, True, True, False, False, False],
        [False, True, True, True, False, False],
        [False, False, True, True, True, False],
        [False, False, False, True, True, True],
    ])
    assert torch.equal(vis, expected)


def test_streaming_sink_kept():
    cfg = SparseConfig(mode="streaming_llm", window_size=3, sink_size=2)
    vis = sparse_visibility(4, 6, cfg)
    assert bool(vis[-1, 0])
    assert bool(vis[-1, 1])
    assert bool(vis[-1, 3])
    assert bool(vis[-1, 4])
    assert bool(vis[-1, 5])
    assert not bool(vis[-1, 2])


def test_block_sparse_has_density():
    cfg = SparseConfig(mode="block_sparse", block_size=2, block_lookback=1)
    density = sparse_density(8, 8, cfg)
    assert 0 < density["density"] < 1
    assert density["sparsity"] > 0


class TinySparseLM(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def _update_causal_mask(self, attention_mask, inputs_embeds,
                            cache_position=None, past_key_values=None,
                            output_attentions=False):
        q_len = inputs_embeds.shape[1]
        mask = torch.zeros((1, 1, q_len, q_len), device=inputs_embeds.device,
                           dtype=inputs_embeds.dtype)
        future = torch.triu(torch.ones(q_len, q_len, device=inputs_embeds.device,
                                       dtype=torch.bool), diagonal=1)
        return mask.masked_fill(future.unsqueeze(0).unsqueeze(0),
                                torch.finfo(inputs_embeds.dtype).min)

    def forward(self, x):
        return self._update_causal_mask(None, x)


def test_install_sparse_attention_patches_and_restores():
    model = TinySparseLM()
    x = torch.randn(2, 6, 32)
    base = model._update_causal_mask(None, x)

    cfg = SparseConfig(mode="sliding_window", window_size=3)
    handle = install_sparse_attention(model, cfg)
    try:
        patched = model._update_causal_mask(None, x)
        expected = build_sparse_attention_mask(6, 6, cfg, dtype=x.dtype)
        assert torch.equal(patched, expected)
    finally:
        handle.restore()

    restored = model._update_causal_mask(None, x)
    assert torch.equal(restored, base)
