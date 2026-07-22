"""StreamingLLM KV cache 裁剪工具(M2-c)。

M2-b 的 additive mask 只是在 attention 里屏蔽旧 token,KV cache 本身仍按完整
上下文增长,所以 decode 的 `kv_len` 没有变短。这里实现的是更直接的路径:
保留 attention sink + 最近窗口的 K/V,让下一步 decode 真正只看这些缓存。

注意:裁剪会让 cache 的物理长度短于原始绝对位置。对 RoPE 模型,调用侧应继续
传递真实递增的 `cache_position` / `position_ids`;否则模型可能按裁剪后的短长度
推断位置,生成质量会受影响。`eval.profiler.profile_latency` 已做这层兼容。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

import torch

from .config import SparseConfig


@dataclass
class CachePruneStats:
    """一次 KV cache 裁剪的统计信息,方便 benchmark 写入 json。"""

    applied: bool
    original_len: int
    kept_len: int
    pruned: int
    layers: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StreamingKVPruneHandle:
    """保存一次 forward KV 裁剪 patch,支持 generate/smoke 后恢复。"""

    owner: object
    attr_name: str
    original: object
    config: SparseConfig
    last_stats: CachePruneStats | None = None

    def restore(self):
        """恢复被替换前的 forward。"""
        setattr(self.owner, self.attr_name, self.original)


def streaming_keep_indices(seq_len: int, sink_size: int, window_size: int,
                           device=None) -> torch.Tensor:
    """返回 StreamingLLM 应保留的 cache 下标。

    规则是 `[0:sink_size)` 的 attention sink 加上最近 `window_size` 个 token。
    两段发生重叠时自然合并,输出保持升序,便于 `index_select` 一次完成裁剪。
    """
    seq_len = max(int(seq_len), 0)
    sink_size = max(int(sink_size), 0)
    window_size = max(int(window_size), 0)
    sink_end = min(sink_size, seq_len)
    recent_start = max(sink_end, seq_len - window_size)

    parts = []
    if sink_end > 0:
        parts.append(torch.arange(0, sink_end, device=device, dtype=torch.long))
    if recent_start < seq_len:
        parts.append(torch.arange(recent_start, seq_len, device=device, dtype=torch.long))
    if not parts:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(parts, dim=0)


def prune_tensor_cache(tensor: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    """沿 K/V 张量的序列维(`-2`)裁剪。

    HF causal LM 常见 K/V 形状是 `[batch, heads, seq, head_dim]`;
    GQA/MQA 也是同一序列维。若张量维度不足,保持原样。
    """
    if not torch.is_tensor(tensor) or tensor.dim() < 2:
        return tensor
    if keep_indices.numel() == tensor.shape[-2]:
        return tensor
    return tensor.index_select(dim=-2, index=keep_indices.to(tensor.device))


def _tensor_seq_len(tensor: Any) -> int:
    if torch.is_tensor(tensor) and tensor.dim() >= 2:
        return int(tensor.shape[-2])
    return 0


def _legacy_seq_len(past_key_values) -> int:
    if not isinstance(past_key_values, (tuple, list)) or not past_key_values:
        return 0
    first = past_key_values[0]
    if isinstance(first, (tuple, list)) and first:
        first = first[0]
    return _tensor_seq_len(first)


def _set_attr_if_possible(obj, name: str, value) -> bool:
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _prune_layer_like(layer, keep_indices: torch.Tensor, original_len: int):
    """裁剪单个 cache layer 或 legacy (key, value) 对。"""
    if layer is None:
        return layer, 0

    if isinstance(layer, (tuple, list)):
        if len(layer) >= 2 and torch.is_tensor(layer[0]) and torch.is_tensor(layer[1]):
            key, value, *rest = layer
            if _tensor_seq_len(key) != original_len or _tensor_seq_len(value) != original_len:
                return layer, 0
            key = prune_tensor_cache(key, keep_indices)
            value = prune_tensor_cache(value, keep_indices)
            return type(layer)((key, value, *rest)), 1
        changed = 0
        items = []
        for item in layer:
            new_item, item_changed = _prune_layer_like(item, keep_indices, original_len)
            changed += item_changed
            items.append(new_item)
        return type(layer)(items), changed

    if torch.is_tensor(layer):
        return layer, 0

    direct_pairs = (
        ("keys", "values"),
        ("key", "value"),
        ("k_cache", "v_cache"),
    )
    for key_attr, value_attr in direct_pairs:
        key = getattr(layer, key_attr, None)
        value = getattr(layer, value_attr, None)
        if not (torch.is_tensor(key) and torch.is_tensor(value)):
            continue
        if _tensor_seq_len(key) != original_len or _tensor_seq_len(value) != original_len:
            continue
        new_key = prune_tensor_cache(key, keep_indices)
        new_value = prune_tensor_cache(value, keep_indices)
        if _set_attr_if_possible(layer, key_attr, new_key) and _set_attr_if_possible(layer, value_attr, new_value):
            for extra_attr in ("cumulative_length", "seq_length", "seq_len", "_seen_tokens"):
                if hasattr(layer, extra_attr):
                    _set_attr_if_possible(layer, extra_attr, int(keep_indices.numel()))
            return layer, 1

    return layer, 0


def _prune_cache_container(past_key_values, keep_indices: torch.Tensor,
                           original_len: int):
    """递归裁剪新版 HF cache 容器。"""
    # 一些实现把真正的 layer 列表挂在 `.layers` 上。
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, list):
        layer_count = 0
        for i, layer in enumerate(layers):
            new_layer, changed = _prune_layer_like(layer, keep_indices, original_len)
            if changed:
                layers[i] = new_layer
            layer_count += changed
        return past_key_values, layer_count

    # 旧一些的容器可能分别暴露 key/value cache 列表。
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        layer_count = 0
        for i, (key, value) in enumerate(zip(key_cache, value_cache)):
            if _tensor_seq_len(key) != original_len or _tensor_seq_len(value) != original_len:
                continue
            key_cache[i] = prune_tensor_cache(key, keep_indices)
            value_cache[i] = prune_tensor_cache(value, keep_indices)
            layer_count += 1
        if layer_count and hasattr(past_key_values, "_seen_tokens"):
            _set_attr_if_possible(past_key_values, "_seen_tokens", int(keep_indices.numel()))
        return past_key_values, layer_count

    # Decoder-only 的某些实现会把 cache 再包一层,例如 self/cross attention cache。
    subattrs = ("self_attention_cache", "self_cache", "past", "cache")
    layer_count = 0
    for name in subattrs:
        sub = getattr(past_key_values, name, None)
        if sub is None:
            continue
        new_sub, changed = _prune_cache_object(sub, keep_indices, original_len)
        if changed:
            _set_attr_if_possible(past_key_values, name, new_sub)
            layer_count += changed
    if layer_count:
        return past_key_values, layer_count
    return past_key_values, 0


def _prune_cache_object(past_key_values, keep_indices: torch.Tensor,
                        original_len: int):
    """对未知形态的 cache 做递归裁剪。"""
    if isinstance(past_key_values, (tuple, list)):
        return _prune_legacy_cache(past_key_values, keep_indices, original_len)
    return _prune_cache_container(past_key_values, keep_indices, original_len)


def _prune_legacy_cache(past_key_values, keep_indices: torch.Tensor,
                        original_len: int):
    """裁剪 tuple/list 格式的 legacy `past_key_values`。"""
    pruned_layers = []
    layer_count = 0
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            pruned_layers.append(layer)
            continue
        key, value, *rest = layer
        if _tensor_seq_len(key) != original_len or _tensor_seq_len(value) != original_len:
            pruned_layers.append(layer)
            continue
        key = prune_tensor_cache(key, keep_indices)
        value = prune_tensor_cache(value, keep_indices)
        layer_count += 1
        layer_type = type(layer)
        pruned_layers.append(layer_type((key, value, *rest)))
    cache_type = type(past_key_values)
    return cache_type(pruned_layers), layer_count


def prune_streaming_past_key_values(past_key_values, cfg: SparseConfig):
    """按 StreamingLLM 策略裁剪 `past_key_values`。

    返回 `(new_past_key_values, CachePruneStats)`。当前支持:
    - legacy tuple/list cache: `((key, value), ...)`
    - HF DynamicCache 常见实现:对象上有 `key_cache` / `value_cache` list
    """
    if past_key_values is None:
        return past_key_values, CachePruneStats(
            applied=False, original_len=0, kept_len=0, pruned=0, reason="no_cache")

    if hasattr(past_key_values, "get_seq_length"):
        try:
            original_len = int(past_key_values.get_seq_length())
        except Exception:
            original_len = 0
    else:
        original_len = 0
    if original_len <= 0:
        original_len = _legacy_seq_len(past_key_values)
    if original_len <= 0:
        return past_key_values, CachePruneStats(
            applied=False, original_len=0, kept_len=0, pruned=0,
            reason="unsupported_cache")

    device = None
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
            device = first[0].device
    else:
        key_cache = getattr(past_key_values, "key_cache", None)
        if isinstance(key_cache, list) and key_cache and torch.is_tensor(key_cache[0]):
            device = key_cache[0].device

    keep_indices = streaming_keep_indices(
        original_len, cfg.sink_size, cfg.window_size, device=device)
    kept_len = int(keep_indices.numel())
    if kept_len >= original_len:
        return past_key_values, CachePruneStats(
            applied=False, original_len=original_len, kept_len=kept_len,
            pruned=0, reason="within_budget")

    new_past, layers = _prune_cache_object(past_key_values, keep_indices, original_len)

    if layers <= 0:
        return past_key_values, CachePruneStats(
            applied=False, original_len=original_len, kept_len=kept_len,
            pruned=0, reason="unsupported_cache")
    return new_past, CachePruneStats(
        applied=True, original_len=original_len, kept_len=kept_len,
        pruned=original_len - kept_len, layers=layers)


def _replace_output_past(out, new_past) -> bool:
    """尽量把模型输出里的 `past_key_values` 换成裁剪后的对象。"""
    if isinstance(out, dict):
        out["past_key_values"] = new_past
        return True
    try:
        setattr(out, "past_key_values", new_past)
        return True
    except Exception:
        pass
    try:
        out["past_key_values"] = new_past
        return True
    except Exception:
        return False


def install_streaming_kv_pruning(model, cfg: SparseConfig) -> StreamingKVPruneHandle:
    """给模型 forward 安装 StreamingLLM KV cache 裁剪。

    这条路径服务 `generate()` 等标准 HF 调用:每次 forward 正常运行,随后把输出
    里的 `past_key_values` 裁成 sink+window。若某个模型输出对象不可修改,wrapper
    会保持输出不变,但 `handle.last_stats.reason` 会暴露失败原因。
    """
    if cfg.mode != "streaming_llm":
        raise ValueError("KV cache pruning 当前只支持 mode=streaming_llm")

    original = model.forward
    handle = StreamingKVPruneHandle(
        owner=model, attr_name="forward", original=original, config=cfg)

    def wrapped(self, *args, **kwargs):
        out = original(*args, **kwargs)
        past = getattr(out, "past_key_values", None)
        if past is None and isinstance(out, dict):
            past = out.get("past_key_values")
        new_past, stats = prune_streaming_past_key_values(past, cfg)
        if stats.applied and not _replace_output_past(out, new_past):
            stats = CachePruneStats(
                applied=False,
                original_len=stats.original_len,
                kept_len=stats.kept_len,
                pruned=0,
                layers=stats.layers,
                reason="output_immutable",
            )
        handle.last_stats = stats
        return out

    setattr(model, "forward", MethodType(wrapped, model))
    return handle
