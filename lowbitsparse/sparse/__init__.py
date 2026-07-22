"""稀疏注意力模块(M2 实现:滑窗 / StreamingLLM / 块稀疏)。"""
from .config import SparseConfig
from .masks import build_sparse_attention_mask, sparse_visibility, sparse_density
from .apply import install_sparse_attention, SparsePatchHandle

__all__ = [
    "SparseConfig",
    "build_sparse_attention_mask",
    "sparse_visibility",
    "sparse_density",
    "install_sparse_attention",
    "SparsePatchHandle",
]
