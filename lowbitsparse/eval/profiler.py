"""延迟与显存 profiler:prefill / decode 延迟、显存峰值。"""
import statistics
import time

import torch


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def profile_latency(
    model,
    tokenizer,
    prefill_len: int = 512,
    decode_tokens: int = 128,
    warmup: int = 2,
    repeats: int = 5,
    device: str = None,
) -> dict:
    """测量 prefill 延迟与 decode 吞吐(tokens/s),取中位数。"""
    if device is None:
        device = next(model.parameters()).device
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, prefill_len), device=device)

    prefill_times, decode_tps = [], []
    for i in range(warmup + repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        _sync(device)
        t1 = time.perf_counter()
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)

        _sync(device)
        t2 = time.perf_counter()
        for _ in range(decode_tokens):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        _sync(device)
        t3 = time.perf_counter()

        if i >= warmup:  # 跳过 warmup
            prefill_times.append(t1 - t0)
            decode_tps.append(decode_tokens / (t3 - t2))

    return {
        "prefill_len": prefill_len,
        "decode_tokens": decode_tokens,
        "prefill_ms_median": round(statistics.median(prefill_times) * 1e3, 3),
        "decode_tps_median": round(statistics.median(decode_tps), 2),
    }


def profile_memory(device: str = None) -> dict:
    """返回当前与峰值显存(MB)。调用前建议 reset_peak_memory_stats。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not str(device).startswith("cuda"):
        return {"peak_mb": None, "current_mb": None}
    return {
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3),
        "current_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 3),
    }
