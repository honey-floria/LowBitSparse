"""稀疏注意力 benchmark。

这里对应 M2 的验收需求:
- 同一模型、同一配置下先跑 baseline。
- 再挂 sparse mask 跑同一组长度。
- 最后对比 PPL / prefill / decode / memory。

和 quant/ 的思路一致:benchmark 本身不发明新算法,只把“怎么测”固定下来。
"""
from __future__ import annotations

import torch

from lowbitsparse.eval import eval_wikitext2_ppl, profile_latency, profile_memory
from lowbitsparse.utils import get_logger

from .apply import install_sparse_attention
from .config import SparseConfig


log = get_logger()


def _run_one(model, tokenizer, seqlen: int, eval_cfg: dict, profile_cfg: dict) -> dict:
    """跑单个长度的一次完整评测。

    PPL / latency / memory 三个指标保持和 M0/M1 一致,这样 M2 的结果可以直接
    和前面里程碑横向比较。
    """
    if torch.cuda.is_available():
        # 评测前清掉 CUDA 峰值,避免把上一轮 baseline 的峰值算进 sparse。
        torch.cuda.reset_peak_memory_stats()
    eval_cfg = eval_cfg or {}
    profile_cfg = profile_cfg or {}
    try:
        ppl = eval_wikitext2_ppl(
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
    lat = profile_latency(
        model, tokenizer,
        prefill_len=seqlen,
        decode_tokens=profile_cfg.get("decode_tokens", 128),
        warmup=profile_cfg.get("warmup", 2),
        repeats=profile_cfg.get("repeats", 5),
    )
    mem = profile_memory()
    return {"ppl": ppl, "latency": lat, "memory": mem}


def benchmark_sparse_attention(model, tokenizer, sparse_cfg: SparseConfig,
                               eval_cfg: dict = None,
                               profile_cfg: dict = None) -> dict:
    """先跑原始模型，再安装稀疏 mask，输出逐长度对照表。

    返回结果已经整理成 JSON 友好的结构,可直接交给 `save_results` 落盘。
    """
    lengths = tuple(sparse_cfg.benchmark_lengths)
    baseline = []
    for length in lengths:
        log.info("[M2] baseline benchmark seqlen=%d", length)
        baseline.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                     eval_cfg, profile_cfg)})

    # sparse 评测直接复用同一个 model,只切换 mask 行为。
    patch = install_sparse_attention(model, sparse_cfg)
    try:
        sparse = []
        for length in lengths:
            log.info("[M2] sparse(%s) benchmark seqlen=%d", sparse_cfg.mode, length)
            sparse.append({"seqlen": length, **_run_one(model, tokenizer, length,
                                                        eval_cfg, profile_cfg)})
    finally:
        patch.restore()

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

    return {
        "sparse_config": {
            "mode": sparse_cfg.mode,
            "window_size": sparse_cfg.window_size,
            "sink_size": sparse_cfg.sink_size,
            "block_size": sparse_cfg.block_size,
            "block_lookback": sparse_cfg.block_lookback,
            "benchmark_lengths": list(lengths),
        },
        "rows": rows,
    }
