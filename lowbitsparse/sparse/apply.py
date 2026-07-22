"""把稀疏注意力挂到 HF causal LM 的 causal mask 构造上。

设计上尽量贴近 quant/apply.py:
- 先找到“模型里真正负责构造 causal mask 的位置”。
- 再包一层轻量 wrapper，只改 mask，不改权重和主干计算。
- 最后返回一个可恢复的 handle，方便 benchmark 前后切换回原行为。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MethodType

import torch

from .config import SparseConfig
from .masks import build_sparse_attention_mask


def _candidate_owners(model):
    """枚举可能持有 `_update_causal_mask` 的对象。

    HF 不同模型家族会把主体模块挂在不同字段上(model / transformer / backbone)。
    这里先尝试常见入口，再遍历全模型兜底，尽量让稀疏注入保持模型无关。
    """
    seen = set()
    for obj in (getattr(model, "model", None), getattr(model, "transformer", None),
                getattr(model, "backbone", None), model):
        if obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            yield obj
    for module in model.modules():
        if id(module) not in seen:
            seen.add(id(module))
            yield module


def _get_past_length(past_key_values) -> int:
    """从 cache/past_key_values 里估算历史长度。

    不同 HF 版本的 cache 结构不完全一致,所以这里做多分支兼容:
    - 优先用 `get_seq_length()`。
    - 否则从 tuple/list 里读第一个 key tensor 的序列维度。
    """
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and first:
            first = first[0]
        if torch.is_tensor(first) and first.dim() >= 2:
            return int(first.shape[-2])
    return 0


def _infer_shape(base_mask, bound, target) -> tuple:
    """推断 query / kv 长度以及 device / dtype。

    `_update_causal_mask` 在不同模型里签名不完全一致,有的直接返回 mask,
    有的通过 inputs_embeds / attention_mask / past_key_values 间接推断形状。
    这个辅助函数就是把这些信息拼齐,供稀疏 mask 构造使用。
    """
    device = None
    dtype = None
    q_len = None
    kv_len = None

    if torch.is_tensor(base_mask):
        device = base_mask.device
        dtype = base_mask.dtype
        if base_mask.dim() >= 4:
            q_len = int(base_mask.shape[-2])
            kv_len = int(base_mask.shape[-1])
        elif base_mask.dim() == 3:
            q_len = int(base_mask.shape[-2])
            kv_len = int(base_mask.shape[-1])

    tensor = None
    # 优先从常见的输入张量里取 device/dtype/seq_len。
    for key in ("inputs_embeds", "hidden_states", "input_embeds"):
        value = bound.arguments.get(key)
        if torch.is_tensor(value):
            tensor = value
            break
    if tensor is None and torch.is_tensor(base_mask):
        tensor = base_mask

    if tensor is not None:
        device = tensor.device if device is None else device
        dtype = tensor.dtype if dtype is None else dtype
        if q_len is None and tensor.dim() >= 2:
            q_len = int(tensor.shape[1])

    if kv_len is None:
        attn_mask = bound.arguments.get("attention_mask")
        if torch.is_tensor(attn_mask) and attn_mask.dim() >= 2:
            kv_len = int(attn_mask.shape[-1])
        else:
            # 没显式传 attention_mask 时,用 past 长度 + 当前 query 长度估计。
            past = bound.arguments.get("past_key_values")
            if past is None:
                past = bound.arguments.get("past_key_value")
            kv_len = _get_past_length(past) + (q_len or 0)

    if device is None:
        try:
            device = next(target.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return q_len, kv_len, device, dtype


def _merge_masks(base_mask, sparse_mask):
    """把原 causal mask 和稀疏 mask 合成一个最终 mask。

    bool mask 用逻辑与；additive mask 用逐元素取更小值，避免重复加法把
    已经是 `-inf` 的位置再叠一遍造成不必要的数值污染。
    """
    if base_mask is None:
        return sparse_mask
    if torch.is_tensor(base_mask) and base_mask.dtype == torch.bool:
        return base_mask & (sparse_mask > torch.finfo(sparse_mask.dtype).min / 2)
    if torch.is_tensor(base_mask):
        return torch.minimum(base_mask, sparse_mask)
    return sparse_mask


@dataclass
class SparsePatchHandle:
    """保存一次 patch 的上下文,支持 benchmark 后恢复原始实现。"""
    owner: object
    original: object
    config: SparseConfig
    owner_name: str

    def restore(self):
        """恢复被替换前的 `_update_causal_mask`。"""
        setattr(self.owner, "_update_causal_mask", self.original)


def install_sparse_attention(model, cfg: SparseConfig) -> SparsePatchHandle:
    """将稀疏注意力挂到第一个可用的 `_update_causal_mask` 上。

    这里不碰 attention 权重,也不替换现有 attention module,
    只是拦截 causal mask 的生成环节,属于“最小侵入”的实现。
    """
    for owner in _candidate_owners(model):
        method = getattr(owner, "_update_causal_mask", None)
        if not callable(method):
            continue

        sig = inspect.signature(method)

        def wrapped(self, *args, **kwargs):
            # 先走原始逻辑,保留模型家族自己的 mask 处理细节。
            bound = sig.bind_partial(*args, **kwargs)
            base_mask = method(*args, **kwargs)
            # 再叠加 sparse 约束;如果形状推断失败,就退回原始 mask。
            q_len, kv_len, device, dtype = _infer_shape(base_mask, bound, owner)
            if q_len is None or kv_len is None:
                return base_mask
            sparse_mask = build_sparse_attention_mask(q_len, kv_len, cfg,
                                                      device=device, dtype=dtype)
            return _merge_masks(base_mask, sparse_mask)

        setattr(owner, "_update_causal_mask", MethodType(wrapped, owner))
        owner_name = owner.__class__.__name__
        return SparsePatchHandle(owner=owner, original=method,
                                 config=cfg, owner_name=owner_name)

    raise NotImplementedError(
        "当前模型未暴露 `_update_causal_mask`，无法注入稀疏注意力。")
