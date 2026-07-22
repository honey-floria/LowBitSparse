"""稀疏注意力配置。

这个模块只收口超参,不掺入任何实现逻辑:
- `mode` 决定稀疏策略(sliding / streaming / block)。
- `window_size` / `sink_size` / `block_size` 控制不同模式的可见范围。
- `cache_pruning` 决定是否启用 M2-c 的 KV cache 真实裁剪路径。
- `benchmark_lengths` 统一给 M2 benchmark 用,方便把 2k/4k/8k/16k 这组长度
  放进同一份 YAML 里复现。
"""
from dataclasses import dataclass


def _normalize_mode(mode: str) -> str:
    """把用户可能写的短别名规范化成内部统一命名。

    这样 YAML 里可以写 `sliding` / `streaming` 这种简写,
    代码内部仍只处理一套稳定的字符串。
    """
    aliases = {
        "sliding": "sliding_window",
        "window": "sliding_window",
        "streaming": "streaming_llm",
        "streamingllm": "streaming_llm",
        "block": "block_sparse",
        "blocksparse": "block_sparse",
    }
    mode = (mode or "sliding_window").lower()
    return aliases.get(mode, mode)


@dataclass
class SparseConfig:
    """稀疏注意力超参。

    字段含义:
        mode:            稀疏模式名,当前支持 sliding_window / streaming_llm / block_sparse。
        window_size:      滑窗宽度,决定每个 query 允许看多少个最近 token。
        sink_size:        StreamingLLM 的 sink token 数,前几个 token 永久在线。
        block_size:       block_sparse 的块边长。
        block_lookback:   块稀疏里向后保留几个块。
        cache_pruning:    是否启用 StreamingLLM KV cache 裁剪(M2-c)。
        benchmark_lengths: benchmark 统一评测的序列长度集合。
    """

    mode: str = "sliding_window"
    window_size: int = 1024
    sink_size: int = 64
    block_size: int = 128
    block_lookback: int = 1
    cache_pruning: bool = False
    benchmark_lengths: tuple = (2048, 4096, 8192, 16384)

    @classmethod
    def from_dict(cls, d: dict) -> "SparseConfig":
        """从 YAML/dict 构造配置,忽略未知键。

        这样 sparse 段可以随着实验逐步加字段,旧配置不会因为多余键直接报错。
        """
        if not d:
            return cls()
        fields = cls.__dataclass_fields__
        payload = {k: v for k, v in d.items() if k in fields}
        if "mode" in payload:
            # 允许 YAML 里写短别名,统一转换后再进入后续逻辑。
            payload["mode"] = _normalize_mode(payload["mode"])
        if "benchmark_lengths" in payload and payload["benchmark_lengths"] is not None:
            # YAML 里通常写 list,这里转 tuple 方便后面当只读序列用。
            payload["benchmark_lengths"] = tuple(int(x) for x in payload["benchmark_lengths"])
        return cls(**payload)
