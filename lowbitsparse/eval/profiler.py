"""延迟与显存 profiler。

分别度量两类推理性能,对应压缩项目最关心的加速比来源:
- prefill:一次性处理整段 prompt 的耗时(受序列长度、注意力复杂度影响,M2 稀疏重点);
- decode :自回归逐 token 生成的吞吐(tokens/s,受 KV cache、单步算子影响)。
以及显存峰值,用于评估量化/稀疏带来的显存收益。
"""
import statistics   # 取中位数,抵抗个别抖动
import time         # 高精度计时

import torch


def _sync(device):
    """CUDA 计时前的同步屏障。

    GPU 算子异步下发,不同步会导致计时只统计到"下发"而非"执行完"。
    仅 cuda 设备需要;cpu 为同步执行,无需处理。
    """
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()   # 性能测试无需梯度
def profile_latency(
    model,                       # 已 eval 的模型
    tokenizer,                   # 预留(当前用随机 token,不实际编码文本)
    prefill_len: int = 512,      # prefill 阶段输入长度
    decode_tokens: int = 128,    # decode 阶段生成的 token 数
    warmup: int = 2,             # 预热轮数(不计入统计,排除首次编译/缓存开销)
    repeats: int = 5,            # 正式测量轮数,取中位
    device: str = None,          # 设备,None 取模型所在设备
) -> dict:
    """测量 prefill 延迟与 decode 吞吐,均取中位数以抵抗抖动。"""
    if device is None:
        device = next(model.parameters()).device
    vocab = model.config.vocab_size   # 词表大小,用于生成合法随机 token
    # 构造随机输入 [1, prefill_len];测延迟只关心计算量,内容无所谓
    input_ids = torch.randint(0, vocab, (1, prefill_len), device=device)

    prefill_times, decode_tps = [], []   # 分别收集各轮 prefill 耗时 / decode 吞吐
    # 共跑 warmup+repeats 轮,前 warmup 轮丢弃
    for i in range(warmup + repeats):
        # ---- prefill 阶段:一次性前向整段输入 ----
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)   # use_cache 生成 KV cache 供 decode 复用
        _sync(device)
        t1 = time.perf_counter()
        past = out.past_key_values               # 缓存的 K/V,decode 阶段增量复用
        nxt = out.logits[:, -1:].argmax(-1)      # 取末位 logits 的贪心 token 作为首个输入

        # ---- decode 阶段:自回归逐 token 生成 ----
        _sync(device)
        t2 = time.perf_counter()
        for _ in range(decode_tokens):
            # 每步只喂 1 个新 token + 复用 past_key_values,模拟真实解码
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values           # 滚动更新 KV cache
            nxt = out.logits[:, -1:].argmax(-1)  # 贪心取下一 token
        _sync(device)
        t3 = time.perf_counter()

        if i >= warmup:   # 跳过预热轮,只统计稳定态
            prefill_times.append(t1 - t0)                    # 本轮 prefill 秒数
            decode_tps.append(decode_tokens / (t3 - t2))     # 本轮 decode 吞吐 tok/s

    return {
        "prefill_len": prefill_len,       # 记录配置
        "decode_tokens": decode_tokens,
        # 中位数比均值更抗个别毛刺;prefill 转毫秒更直观
        "prefill_ms_median": round(statistics.median(prefill_times) * 1e3, 3),
        "decode_tps_median": round(statistics.median(decode_tps), 2),
    }


def profile_memory(device: str = None) -> dict:
    """返回当前与峰值显存(MB)。

    参数:
        device: 设备;None 时自动探测(有 cuda 用 cuda,否则 cpu)。
    返回:
        {peak_mb, current_mb};非 cuda 设备两者均为 None。

    用法建议:在开始一段计算前调用 torch.cuda.reset_peak_memory_stats(),
    计算结束后调用本函数,得到的 peak_mb 即该段的显存峰值(含 KV cache),
    可用于对比量化/稀疏前后的显存收益。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 非 GPU 环境无显存概念,返回 None 占位,保持 json 结构一致
    if not str(device).startswith("cuda"):
        return {"peak_mb": None, "current_mb": None}
    return {
        # 历史峰值:自上次 reset 以来分配过的最大显存
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3),
        # 当前时刻已分配显存
        "current_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 3),
    }
