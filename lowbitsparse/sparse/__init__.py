"""稀疏注意力模块(M2 实现:滑窗 / StreamingLLM / 块稀疏)。"""
from .config import SparseConfig
from .masks import build_sparse_attention_mask, sparse_visibility, sparse_density
from .apply import install_sparse_attention, SparsePatchHandle
from .cache import (
    CachePruneStats,
    StreamingKVPruneHandle,
    install_streaming_kv_pruning,
    prune_streaming_past_key_values,
    prune_tensor_cache,
    streaming_keep_indices,
)

__all__ = [
    "SparseConfig",
    "build_sparse_attention_mask",
    "sparse_visibility",
    "sparse_density",
    "install_sparse_attention",
    "SparsePatchHandle",
    "CachePruneStats",
    "StreamingKVPruneHandle",
    "install_streaming_kv_pruning",
    "prune_streaming_past_key_values",
    "prune_tensor_cache",
    "streaming_keep_indices",
]
