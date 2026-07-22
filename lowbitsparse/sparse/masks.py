"""稀疏注意力 mask 构造。

这里的职责和 quant/primitives 类似:把“数学上怎么定义稀疏可见性”单独抽出来,
让后面的 hook / benchmark 只消费一个统一的 mask 接口。

核心约定:
- `sparse_visibility` 输出 bool 矩阵,表示哪些 key 对哪些 query 可见。
- `build_sparse_attention_mask` 把 bool 视图转成 additive mask,可直接喂给 attention。
"""
from __future__ import annotations

import torch

from .config import SparseConfig


def _to_float_dtype(dtype):
    """把可能传入的 dtype 规范成可用于 mask 的浮点 dtype。

    某些调用路径可能给的是非浮点 dtype 或 None,这里统一兜底成 float32,
    避免后面 `torch.finfo` / `masked_fill` 出现类型问题。
    """
    if dtype is None:
        return torch.float32
    try:
        probe = torch.empty((), dtype=dtype)
    except Exception:
        return torch.float32
    return dtype if torch.is_floating_point(probe) else torch.float32


def sparse_visibility(q_len: int, kv_len: int, cfg: SparseConfig,
                      device=None) -> torch.Tensor:
    """返回 bool 形式的可见性矩阵 [q_len, kv_len]。

    行表示 query 位置,列表示 key 位置;True 表示这个 key 可以参与当前 query 的注意力。
    这里先构造全局 causal 约束,再叠加不同稀疏模式的局部规则。
    """
    if q_len <= 0 or kv_len <= 0:
        raise ValueError("q_len 和 kv_len 必须为正")
    if q_len > kv_len:
        raise ValueError("q_len 不能大于 kv_len")

    mode = cfg.mode
    # query 位置只覆盖最后 q_len 个 token,与 HF causal mask 的真实用法一致。
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    # 基础因果约束:未来 token 永远不可见。
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

    if mode == "sliding_window":
        # 滑窗:只保留每个 query 左侧最近 window_size 个 token。
        start = (q_pos - max(cfg.window_size, 1) + 1).unsqueeze(1)
        visible = causal & (k_pos.unsqueeze(0) >= start)
    elif mode == "streaming_llm":
        # StreamingLLM:前 sink_size 个 token 永久在线,后面再叠加局部滑窗。
        sink = k_pos.unsqueeze(0) < max(cfg.sink_size, 0)
        start = (q_pos - max(cfg.window_size, 1) + 1).unsqueeze(1)
        local = k_pos.unsqueeze(0) >= start
        visible = causal & (sink | local)
    elif mode == "block_sparse":
        # 块稀疏:按 block_size 分块,当前块只看若干个历史块 + sink 区域。
        block_size = max(cfg.block_size, 1)
        lookback = max(cfg.block_lookback, 0)
        sink = k_pos.unsqueeze(0) < max(cfg.sink_size, 0)
        q_block = q_pos // block_size
        k_block = k_pos // block_size
        local = (q_block.unsqueeze(1) - k_block.unsqueeze(0))
        visible = causal & (sink | ((local >= 0) & (local <= lookback)))
    else:
        raise ValueError(f"未知 sparse mode: {cfg.mode}")

    return visible


def build_sparse_attention_mask(q_len: int, kv_len: int, cfg: SparseConfig,
                                device=None, dtype=None) -> torch.Tensor:
    """返回 additive mask [1, 1, q_len, kv_len]。

    这里用 additive mask 而不是 bool mask,是为了兼容 HF / SDPA / 传统 attention
    里最常见的“加上一个很大的负数再 softmax”的实现路径。
    """
    dtype = _to_float_dtype(dtype)
    visible = sparse_visibility(q_len, kv_len, cfg, device=device)
    mask = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=dtype)
    # 不可见位置填充极小值,softmax 后近似 0。
    neg_inf = torch.finfo(dtype).min
    mask = mask.masked_fill(~visible.unsqueeze(0).unsqueeze(0), neg_inf)
    return mask


def sparse_density(q_len: int, kv_len: int, cfg: SparseConfig, device=None) -> dict:
    """返回可见 token 占比与稀疏率。

    这个函数不是核心算法,但对 benchmark 很有用:一眼能看出当前 mask 有多稀疏。
    """
    visible = sparse_visibility(q_len, kv_len, cfg, device=device)
    allowed = int(visible.sum().item())
    total = int(visible.numel())
    density = allowed / max(total, 1)
    return {
        "allowed": allowed,
        "total": total,
        "density": round(density, 6),
        "sparsity": round(1.0 - density, 6),
    }
