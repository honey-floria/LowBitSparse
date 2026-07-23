"""稀疏注意力 benchmark。

这里对应 M2 的验收需求:
- 同一模型、同一配置下先跑 baseline。
- 再挂 sparse mask 跑同一组长度。
- 最后对比 PPL / prefill / decode / memory。

和 quant/ 的思路一致:benchmark 本身不发明新算法,只把“怎么测”固定下来。
"""
from __future__ import annotations

import torch

from lowbitsparse.eval import (
    eval_wikitext2_ppl,
    profile_chunked_prefill_latency,
    profile_latency,
    profile_memory,
)
from lowbitsparse.utils import get_logger

from .apply import install_sparse_attention
from .cache import prune_streaming_past_key_values
from .config import SparseConfig
from .ring_cache import build_ring_graph_decode


log = get_logger()


def _run_ppl(model, tokenizer, seqlen: int, eval_cfg: dict) -> dict:
    """只跑 WikiText-2 PPL,供普通 M2 和 M2-c 的质量参考复用。"""
    eval_cfg = eval_cfg or {}
    try:
        return eval_wikitext2_ppl(
            model, tokenizer,
            seqlen=seqlen,
            stride=seqlen,
            max_samples=eval_cfg.get("max_samples"),
        )
    except ImportError as exc:
        # 这层报错比直接把 datasets 的 ImportError 冒出去更可读。
        raise RuntimeError(
            "稀疏 benchmark 需要 `datasets` 依赖和 WikiText-2 数据集; "
            "当前环境缺少相关组件,无法计算 PPL。"
        ) from exc


def _run_latency_memory(model, tokenizer, seqlen: int, profile_cfg: dict,
                        past_pruner=None) -> dict:
    """跑单个长度的 latency + memory,可选接入 KV cache pruner。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    profile_cfg = profile_cfg or {}
    lat = profile_latency(
        model, tokenizer,
        prefill_len=seqlen,
        decode_tokens=profile_cfg.get("decode_tokens", 128),
        warmup=profile_cfg.get("warmup", 2),
        repeats=profile_cfg.get("repeats", 5),
        past_pruner=past_pruner,
        reset_peak_after_prefill=profile_cfg.get("reset_peak_after_prefill", False),
        # M2-e 验证:torch.compile+StaticCache 无裁剪 decode。与 past_pruner 互斥
        # (static cache 形状固定),故仅在 baseline 路径(past_pruner=None)生效。
        compile_decode=(profile_cfg.get("compile_decode", False)
                        and past_pruner is None),
    )
    mem = profile_memory()
    return {"latency": lat, "memory": mem}


def _run_chunked_prefill_latency_memory(model, tokenizer, seqlen: int,
                                        sparse_cfg: SparseConfig,
                                        profile_cfg: dict) -> dict:
    """M2-d 性能路径:分块 prefill + StreamingLLM KV 裁剪。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    profile_cfg = profile_cfg or {}

    def pruner(past):
        return prune_streaming_past_key_values(past, sparse_cfg)

    lat = profile_chunked_prefill_latency(
        model, tokenizer,
        prefill_len=seqlen,
        chunk_size=sparse_cfg.prefill_chunk_size,
        decode_tokens=profile_cfg.get("decode_tokens", 128),
        warmup=profile_cfg.get("warmup", 2),
        repeats=profile_cfg.get("repeats", 5),
        past_pruner=pruner,
        reset_peak_after_prefill=profile_cfg.get("reset_peak_after_prefill", False),
    )
    mem = profile_memory()
    return {"latency": lat, "memory": mem}


def _run_one(model, tokenizer, seqlen: int, eval_cfg: dict, profile_cfg: dict) -> dict:
    """跑单个长度的一次完整评测。

    PPL / latency / memory 三个指标保持和 M0/M1 一致,这样 M2 的结果可以直接
    和前面里程碑横向比较。
    """
    if torch.cuda.is_available():
        # 评测前清掉 CUDA 峰值,避免把上一轮 baseline 的峰值算进 sparse。
        torch.cuda.reset_peak_memory_stats()
    return {
        "ppl": _run_ppl(model, tokenizer, seqlen, eval_cfg),
        **_run_latency_memory(model, tokenizer, seqlen, profile_cfg),
    }


def _rows_from_runs(baseline, sparse) -> list:
    """把 baseline/sparse 原始结果合成带 speedup 和显存差的表格。"""
    rows = []
    for base, sp in zip(baseline, sparse):
        # speedup 这里保持口径清晰:
        # - prefill 用时间比值(>1 表示 sparse 更快)
        # - decode 用吞吐比值(>1 表示 sparse 更快)
        base_prefill = base["latency"]["prefill_ms_median"]
        sp_prefill = sp["latency"]["prefill_ms_median"]
        base_decode = base["latency"]["decode_tps_median"]
        sp_decode = sp["latency"]["decode_tps_median"]
        rows.append({
            "seqlen": base["seqlen"],
            "baseline": base,
            "sparse": sp,
            "speedup": {
                "prefill": round(base_prefill / max(sp_prefill, 1e-6), 3),
                "decode": round(sp_decode / max(base_decode, 1e-6), 3),
            },
            "memory_delta_mb": (
                None if base["memory"]["peak_mb"] is None or sp["memory"]["peak_mb"] is None
                else round(base["memory"]["peak_mb"] - sp["memory"]["peak_mb"], 3)
            ),
        })
    return rows


def _sparse_config_payload(sparse_cfg: SparseConfig, lengths: tuple) -> dict:
    return {
        "mode": sparse_cfg.mode,
        "window_size": sparse_cfg.window_size,
        "sink_size": sparse_cfg.sink_size,
        "block_size": sparse_cfg.block_size,
        "block_lookback": sparse_cfg.block_lookback,
        "cache_pruning": sparse_cfg.cache_pruning,
        "chunked_prefill": getattr(sparse_cfg, "chunked_prefill", False),
        "prefill_chunk_size": getattr(sparse_cfg, "prefill_chunk_size", 512),
        "ring_graph": getattr(sparse_cfg, "ring_graph", False),
        "benchmark_lengths": list(lengths),
    }


def benchmark_sparse_attention(model, tokenizer, sparse_cfg: SparseConfig,
                               eval_cfg: dict = None,
                               profile_cfg: dict = None) -> dict:
    """先跑原始模型，再安装稀疏 mask，输出逐长度对照表。

    返回结果已经整理成 JSON 友好的结构,可直接交给 `save_results` 落盘。
    """
    if getattr(sparse_cfg, "ring_graph", False):
        return benchmark_streaming_ring_graph(
            model, tokenizer, sparse_cfg, eval_cfg=eval_cfg, profile_cfg=profile_cfg)
    if getattr(sparse_cfg, "chunked_prefill", False):
        return benchmark_streaming_chunked_prefill(
            model, tokenizer, sparse_cfg, eval_cfg=eval_cfg, profile_cfg=profile_cfg)
    if sparse_cfg.cache_pruning:
        return benchmark_streaming_kv_pruning(
            model, tokenizer, sparse_cfg, eval_cfg=eval_cfg, profile_cfg=profile_cfg)

    lengths = tuple(sparse_cfg.benchmark_lengths)
    baseline = []
    for length in lengths:
        log.info("[M2] baseline benchmark seqlen=%d", length)
        baseline.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                     eval_cfg, profile_cfg)})

    # sparse 评测直接复用同一个 model,只切换 mask 行为。
    patch = install_sparse_attention(model, sparse_cfg)
    log.info("[M2] sparse patch installed: %s.%s",
             patch.owner_name, patch.attr_name)
    try:
        sparse = []
        for length in lengths:
            log.info("[M2] sparse(%s) benchmark seqlen=%d", sparse_cfg.mode, length)
            sparse.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                        eval_cfg, profile_cfg)})
    finally:
        patch.restore()

    return {
        "sparse_config": _sparse_config_payload(sparse_cfg, lengths),
        "rows": _rows_from_runs(baseline, sparse),
    }


def benchmark_streaming_kv_pruning(model, tokenizer, sparse_cfg: SparseConfig,
                                   eval_cfg: dict = None,
                                   profile_cfg: dict = None) -> dict:
    """M2-c:StreamingLLM KV cache 裁剪 benchmark。

    质量指标仍用 StreamingLLM additive mask 跑 teacher-forced PPL 作为参考;
    latency/memory 则不安装 additive mask,而是在 decode cache 上真实裁剪 K/V,
    使 `kv_len` 变成 sink+window 量级。
    """
    if sparse_cfg.mode != "streaming_llm":
        raise ValueError("M2-c cache_pruning 当前只支持 mode=streaming_llm")

    lengths = tuple(sparse_cfg.benchmark_lengths)
    baseline = []
    for length in lengths:
        log.info("[M2-c] baseline benchmark seqlen=%d", length)
        baseline.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                     eval_cfg, profile_cfg)})

    # PPL 仍通过 additive mask 表示 StreamingLLM 可见性,仅作为质量参考。
    patch = install_sparse_attention(model, sparse_cfg)
    try:
        sparse_ppl = []
        for length in lengths:
            log.info("[M2-c] sparse quality(reference mask) seqlen=%d", length)
            sparse_ppl.append({"seqlen": length, "ppl": _run_ppl(model, tokenizer,
                                                                 length, eval_cfg)})
    finally:
        patch.restore()

    def pruner(past):
        return prune_streaming_past_key_values(past, sparse_cfg)

    sparse = []
    for item in sparse_ppl:
        length = item["seqlen"]
        log.info("[M2-c] streaming KV prune latency seqlen=%d", length)
        sparse.append({
            "seqlen": length,
            "ppl": item["ppl"],
            **_run_latency_memory(model, tokenizer, length, profile_cfg,
                                  past_pruner=pruner),
        })

    return {
        "sparse_config": _sparse_config_payload(sparse_cfg, lengths),
        "benchmark_kind": "streaming_kv_pruning",
        "quality_note": (
            "sparse.ppl 使用 StreamingLLM additive mask 的 teacher-forced PPL 作为质量参考;"
            "latency/memory 使用真实 KV cache 裁剪路径。"
        ),
        "rows": _rows_from_runs(baseline, sparse),
    }


def benchmark_streaming_chunked_prefill(model, tokenizer, sparse_cfg: SparseConfig,
                                        eval_cfg: dict = None,
                                        profile_cfg: dict = None) -> dict:
    """M2-d:chunked prefill / local attention benchmark。

    质量指标仍用 StreamingLLM additive mask 的 teacher-forced PPL 作参考;性能路径
    不安装 4D additive mask,而是把 prefill 拆成多个 query chunk,chunk 之间真实裁剪
    KV cache 到 sink+window。这样避免一次性构造完整 `[batch,1,q,kv]` mask/cache。
    """
    if sparse_cfg.mode != "streaming_llm":
        raise ValueError("M2-d chunked_prefill 当前只支持 mode=streaming_llm")

    lengths = tuple(sparse_cfg.benchmark_lengths)
    baseline = []
    for length in lengths:
        log.info("[M2-d] baseline benchmark seqlen=%d", length)
        baseline.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                     eval_cfg, profile_cfg)})

    # 质量:additive mask teacher-forced PPL,与 M2-b/M2-c/M2-e 口径一致。
    patch = install_sparse_attention(model, sparse_cfg)
    try:
        sparse_ppl = []
        for length in lengths:
            log.info("[M2-d] sparse quality(reference mask) seqlen=%d", length)
            sparse_ppl.append({"seqlen": length, "ppl": _run_ppl(model, tokenizer,
                                                                 length, eval_cfg)})
    finally:
        patch.restore()

    sparse = []
    for item in sparse_ppl:
        length = item["seqlen"]
        log.info("[M2-d] chunked prefill latency seqlen=%d chunk=%d",
                 length, sparse_cfg.prefill_chunk_size)
        # chunked prefill 的 chunk 内部仍是 dense causal;为了让每块只看有限历史,
        # chunk 间用同一 StreamingLLM 预算裁剪 cache。chunk_size 越小,越接近严格
        # token 级 local attention,但 forward 次数越多。
        sparse.append({
            "seqlen": length,
            "ppl": item["ppl"],
            **_run_chunked_prefill_latency_memory(
                model, tokenizer, length, sparse_cfg, profile_cfg),
        })

    return {
        "sparse_config": _sparse_config_payload(sparse_cfg, lengths),
        "benchmark_kind": "streaming_chunked_prefill",
        "quality_note": (
            "sparse.ppl 使用 StreamingLLM additive mask 的 teacher-forced PPL 作为质量参考;"
            "latency/memory 使用 chunked prefill + StreamingLLM KV cache 裁剪路径。"
        ),
        "rows": _rows_from_runs(baseline, sparse),
    }


def benchmark_streaming_ring_graph(model, tokenizer, sparse_cfg: SparseConfig,
                                   eval_cfg: dict = None,
                                   profile_cfg: dict = None) -> dict:
    """M2-e:有界 ring-buffer KV cache + CUDA graph decode benchmark。

    质量指标仍用 StreamingLLM additive mask 的 teacher-forced PPL 作参考(同 M2-b/M2-c);
    decode latency / memory 走真实 ring-buffer + CUDA graph replay 路径(见 ring_cache)。
    prefill 不是 M2-e 的优化对象,speedup 只看 decode;sparse 行的 prefill 沿用 baseline
    值(speedup=1.0),避免误导。
    """
    if sparse_cfg.mode != "streaming_llm":
        raise ValueError("M2-e ring_graph 当前只支持 mode=streaming_llm")

    lengths = tuple(sparse_cfg.benchmark_lengths)
    profile_cfg = profile_cfg or {}
    baseline = []
    for length in lengths:
        log.info("[M2-e] baseline benchmark seqlen=%d", length)
        baseline.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                     eval_cfg, profile_cfg)})

    # 质量:additive mask teacher-forced PPL(仅参考,同 M2-c)。
    patch = install_sparse_attention(model, sparse_cfg)
    try:
        sparse_ppl = []
        for length in lengths:
            log.info("[M2-e] sparse quality(reference mask) seqlen=%d", length)
            sparse_ppl.append({"seqlen": length, "ppl": _run_ppl(model, tokenizer,
                                                                 length, eval_cfg)})
    finally:
        patch.restore()

    # decode latency/memory:ring-buffer + CUDA graph replay。
    sparse = []
    for item, base in zip(sparse_ppl, baseline):
        length = item["seqlen"]
        log.info("[M2-e] ring+graph decode seqlen=%d", length)
        rg = build_ring_graph_decode(
            model, sink_size=sparse_cfg.sink_size, window_size=sparse_cfg.window_size,
            prefill_len=length, decode_tokens=profile_cfg.get("decode_tokens", 128),
            warmup=profile_cfg.get("warmup", 2), repeats=profile_cfg.get("repeats", 5),
            device=next(model.parameters()).device)
        # 组装成与 _rows_from_runs 兼容的行:decode 用 graph tps,prefill 沿用 baseline,
        # memory 用 ring decode 阶段峰值(恒定)。graph 不可用时回退 baseline 值并标注。
        if rg.get("available"):
            decode_tps = rg["decode_tps_median"]
            peak_mb = rg["decode_peak_mb"]
        else:
            decode_tps = base["latency"]["decode_tps_median"]
            peak_mb = base["memory"]["peak_mb"]
        sparse.append({
            "seqlen": length,
            "ppl": item["ppl"],
            "latency": {
                "prefill_len": length,
                "decode_tokens": profile_cfg.get("decode_tokens", 128),
                "prefill_ms_median": base["latency"]["prefill_ms_median"],  # 非优化对象
                "decode_tps_median": decode_tps,
                "ring_graph": rg,
            },
            "memory": {"peak_mb": peak_mb, "current_mb": base["memory"]["current_mb"]},
        })

    return {
        "sparse_config": _sparse_config_payload(sparse_cfg, lengths),
        "benchmark_kind": "streaming_ring_graph",
        "quality_note": (
            "sparse.ppl 使用 StreamingLLM additive mask 的 teacher-forced PPL 作为质量参考;"
            "decode latency/memory 使用 ring-buffer KV cache + CUDA graph replay 路径;"
            "prefill 非 M2-e 优化对象,沿用 baseline 值。"
        ),
        "rows": _rows_from_runs(baseline, sparse),
    }
