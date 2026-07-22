"""稀疏注意力 mask 与 hook 的单元测试(CPU、纯 torch)。"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lowbitsparse.sparse import (
    CachePruneStats, SparseConfig, build_sparse_attention_mask,
    install_sparse_attention, install_streaming_kv_pruning,
    prune_streaming_past_key_values,
    sparse_density, sparse_visibility, streaming_keep_indices)


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


def test_streaming_keep_indices_keeps_sink_and_window():
    idx = streaming_keep_indices(seq_len=10, sink_size=2, window_size=3)
    assert torch.equal(idx, torch.tensor([0, 1, 7, 8, 9]))


def test_streaming_keep_indices_no_prune_when_short():
    idx = streaming_keep_indices(seq_len=4, sink_size=2, window_size=3)
    assert torch.equal(idx, torch.tensor([0, 1, 2, 3]))


def test_prune_tuple_past_key_values():
    cfg = SparseConfig(mode="streaming_llm", window_size=3, sink_size=2)
    key = torch.randn(1, 4, 10, 8)
    value = torch.randn(1, 4, 10, 8)
    past = ((key, value),)

    pruned, stats = prune_streaming_past_key_values(past, cfg)
    assert isinstance(stats, CachePruneStats)
    assert stats.applied
    assert stats.original_len == 10
    assert stats.kept_len == 5
    assert stats.pruned == 5
    assert pruned[0][0].shape[-2] == 5
    assert pruned[0][1].shape[-2] == 5


def test_prune_tuple_short_cache_noop():
    cfg = SparseConfig(mode="streaming_llm", window_size=3, sink_size=2)
    key = torch.randn(1, 4, 4, 8)
    value = torch.randn(1, 4, 4, 8)
    past = ((key, value),)

    pruned, stats = prune_streaming_past_key_values(past, cfg)
    assert not stats.applied
    assert stats.reason == "within_budget"
    assert pruned[0][0].shape[-2] == 4


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


class ForwardOnlyLM(nn.Module):
    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)
        self.last_attention_mask = None

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False):
        self.last_attention_mask = attention_mask
        return self.proj(self.embed(input_ids))


class DummyOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class KVForwardLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.last_input = None

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        self.last_input = input_ids
        key = torch.randn(1, 2, 10, 4)
        value = torch.randn(1, 2, 10, 4)
        return DummyOutput(self.proj(torch.randn(1, 1, 8)), ((key, value),))


def test_install_sparse_attention_forward_fallback():
    model = ForwardOnlyLM()
    ids = torch.randint(0, 32, (2, 6))
    assert model.forward.__self__ is model

    cfg = SparseConfig(mode="sliding_window", window_size=3)
    handle = install_sparse_attention(model, cfg)
    try:
        model(ids)
        injected = model.last_attention_mask
        expected = build_sparse_attention_mask(6, 6, cfg, dtype=next(model.parameters()).dtype)
        assert injected.shape == (1, 1, 6, 6)
        assert torch.equal(injected, expected)
    finally:
        handle.restore()

    model(ids)
    assert model.last_attention_mask is None


def test_install_streaming_kv_pruning_forward_wrapper():
    model = KVForwardLM()
    cfg = SparseConfig(mode="streaming_llm", window_size=3, sink_size=2)
    handle = install_streaming_kv_pruning(model, cfg)
    try:
        out = model(torch.randint(0, 8, (1, 1)))
        assert out.past_key_values[0][0].shape[-2] == 5
        assert handle.last_stats is not None
        assert handle.last_stats.applied
    finally:
        handle.restore()

    restored = model(torch.randint(0, 8, (1, 1)))
    assert restored.past_key_values[0][0].shape[-2] == 10
