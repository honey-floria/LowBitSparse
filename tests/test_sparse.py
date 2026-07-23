"""稀疏注意力 mask 与 hook 的单元测试(CPU、纯 torch)。"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lowbitsparse.sparse import (
    CachePruneStats, SparseConfig, build_sparse_attention_mask,
    install_sparse_attention, install_streaming_kv_pruning,
    prune_streaming_past_key_values,
    sparse_density, sparse_visibility, streaming_keep_indices,
    RingKVCache, build_ring_graph_decode)


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


class FakeCacheLayer:
    def __init__(self, seq_len=10):
        self.keys = torch.randn(1, 2, seq_len, 4)
        self.values = torch.randn(1, 2, seq_len, 4)
        self.cumulative_length = seq_len

    def get_seq_length(self):
        return int(self.keys.shape[-2])


class FakeLayerCache:
    def __init__(self, seq_len=10, layers=1):
        self.layers = [FakeCacheLayer(seq_len=seq_len) for _ in range(layers)]
        self._seen_tokens = seq_len

    def get_seq_length(self):
        return self.layers[0].get_seq_length() if self.layers else 0


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


def test_install_streaming_kv_pruning_layers_cache():
    class LayerCacheLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, input_ids, past_key_values=None, use_cache=False):
            return DummyOutput(self.proj(torch.randn(1, 1, 8)), FakeLayerCache(seq_len=10))

    model = LayerCacheLM()
    cfg = SparseConfig(mode="streaming_llm", window_size=3, sink_size=2)
    handle = install_streaming_kv_pruning(model, cfg)
    try:
        out = model(torch.randint(0, 8, (1, 1)))
        assert out.past_key_values.layers[0].keys.shape[-2] == 5
        assert out.past_key_values.layers[0].values.shape[-2] == 5
        assert out.past_key_values.layers[0].cumulative_length == 5
        assert handle.last_stats is not None
        assert handle.last_stats.applied
    finally:
        handle.restore()


# ---- M2-e RingKVCache(回绕 KV cache)----


def test_sparse_package_exports_m2e_api():
    assert RingKVCache is not None
    assert callable(build_ring_graph_decode)


def test_ring_cache_prefill_returns_full_kv():
    """prefill 必须返回真实完整 K/V(形状匹配 attention),不能返回截短 buffer。"""
    cache = RingKVCache(sink_size=2, window_size=3)
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn(1, 2, 10, 4)
    rk, rv = cache.update(k, v, layer_idx=0)
    assert rk.shape[-2] == 10 and rv.shape[-2] == 10
    assert torch.equal(rk, k)
    # 内部 buffer 恒定为 sink+window=5。
    assert cache.key_buf[0].shape[-2] == 5


def test_ring_cache_decode_returns_constant_shape():
    """decode 每步返回恒定形状 buffer(CUDA graph replay 的前提)。"""
    cache = RingKVCache(sink_size=2, window_size=3)
    cache.update(torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4), 0)
    shapes = set()
    for _ in range(20):
        rk, rv = cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), 0)
        shapes.add(tuple(rk.shape))
    # 20 步 decode 形状全程不变,且等于 sink+window。
    assert shapes == {(1, 2, 5, 4)}


def test_ring_cache_sink_preserved_window_wraps():
    """sink 段永久保留;window 段循环覆盖最老槽位。"""
    cache = RingKVCache(sink_size=2, window_size=3)
    # prefill 用可辨识的常量填充:每个 token 位置 t 的值全为 t。
    k = torch.stack([torch.full((1, 2, 4), float(t)) for t in range(6)], dim=2)
    v = k.clone()
    cache.update(k, v, 0)
    kbuf = cache.key_buf[0]
    # sink = 前 2 个 token(值 0,1)。
    assert kbuf[0, 0, 0, 0].item() == 0.0
    assert kbuf[0, 0, 1, 0].item() == 1.0
    # window 初始 = 最后 3 个 token(值 3,4,5),写指针回到 0。
    assert kbuf[0, 0, 2, 0].item() == 3.0
    assert kbuf[0, 0, 4, 0].item() == 5.0
    assert cache._wptr[0] == 0

    # decode 一个值为 99 的 token:覆盖 window 槽位 0(buffer index sink+0=2)。
    cache.update(torch.full((1, 2, 1, 4), 99.0), torch.full((1, 2, 1, 4), 99.0), 0)
    assert kbuf[0, 0, 2, 0].item() == 99.0
    # sink 段不受影响。
    assert kbuf[0, 0, 0, 0].item() == 0.0
    assert kbuf[0, 0, 1, 0].item() == 1.0
    assert cache._wptr[0] == 1


def test_ring_cache_seq_length_capped():
    """get_seq_length 不超过 sink+window(cache 大小恒定,不随 decode 增长)。"""
    cache = RingKVCache(sink_size=2, window_size=3)
    cache.update(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4), 0)
    for _ in range(50):
        cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), 0)
    assert cache.get_seq_length(0) == 5


def test_ring_cache_reset_clears():
    cache = RingKVCache(sink_size=2, window_size=3)
    cache.update(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4), 0)
    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache._wptr[0] == 0
    assert torch.count_nonzero(cache.key_buf[0]) == 0
