"""延迟与显存 profiler。

分别度量两类推理性能,对应压缩项目最关心的加速比来源:
- prefill:一次性处理整段 prompt 的耗时(受序列长度、注意力复杂度影响,M2 稀疏重点);
- decode :自回归逐 token 生成的吞吐(tokens/s,受 KV cache、单步算子影响)。
以及显存峰值,用于评估量化/稀疏带来的显存收益。
"""
import statistics   # 取中位数,抵抗个别抖动
import time         # 高精度计时
import inspect      # 判断模型 forward 是否接受 cache_position

import torch


def _sync(device):
    """CUDA 计时前的同步屏障。

    GPU 算子异步下发,不同步会导致计时只统计到"下发"而非"执行完"。
    仅 cuda 设备需要;cpu 为同步执行,无需处理。
    """
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _supports_kwarg(fn, name: str) -> bool:
    """判断 callable 是否显式或通过 **kwargs 接受某个参数。"""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def _as_prune_stats(stats):
    """把 cache pruner 返回的统计对象规范化成 dict。"""
    if stats is None:
        return None
    if hasattr(stats, "to_dict"):
        return stats.to_dict()
    if isinstance(stats, dict):
        return dict(stats)
    return None


def _apply_past_pruner(past, past_pruner):
    """调用外部 KV cache pruner,兼容只返回 past 或返回 `(past, stats)`。"""
    if past_pruner is None:
        return past, None
    result = past_pruner(past)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], _as_prune_stats(result[1])
    return result, None


def _try_build_cudagraph_decode(model, prefill_len, decode_tokens, device):
    """尝试手动捕获单步 decode 的 CUDA graph(M2-e overhead 验证)。

    背景:当前 eager 逐 token decode 是 overhead-bound —— 0.5B 在 A100 上纯带宽
    理论 ~0.66ms/step,实测 ~27ms/step,40x 差距来自每步的 kernel launch / Python
    循环固定开销,而非 KV attention。要验证这个判断,就消除逐步 launch overhead。

    为什么手动捕获而非 torch.compile(reduce-overhead):后者的 CUDA graphs 会被
    transformers cache `update()` 里的原地 `cumulative_length.add_()` 判定为
    "mutated inputs" 而自动跳过(实测 72 instances,overhead 没被消除)。手动
    `torch.cuda.graph` 捕获允许对固定 buffer 原地 mutate,不受此限,且正是 M2-e
    ring-buffer 要用的原语。

    做法:StaticCache 固定形状 → prefill 填一次 → 固定单步输入 buffer → side-stream
    warmup(捕获前必需)→ 捕获一次 decode step。这是"overhead 上限"探针:replay 同
    一张 graph,固定输入、不更新 token,只测消除 launch 开销后单步的纯执行时间(每步
    计算量恒定,故 replay 合理)。不追求生成正确性。任何一步失败返回 `{"error": ...}`。
    """
    if not str(device).startswith("cuda"):
        return None  # CUDA graph 只在 GPU 有意义
    try:
        from transformers import StaticCache
    except Exception:
        return None
    try:
        max_len = prefill_len + decode_tokens + 1
        dtype = next(model.parameters()).dtype
        vocab = model.config.vocab_size
        # 固定形状 KV cache:decode 全程不 resize,满足 CUDA graph 前提。
        static_cache = StaticCache(
            config=model.config, max_batch_size=1, max_cache_len=max_len,
            device=device, dtype=dtype)
        # prefill 一次,把 cache 填到 prefill_len(graph 只捕获之后的单步 decode)。
        prefill_ids = torch.randint(0, vocab, (1, prefill_len), device=device)
        prefill_pos = torch.arange(prefill_len, device=device, dtype=torch.long)
        model(prefill_ids, use_cache=True, past_key_values=static_cache,
              cache_position=prefill_pos)
        # 固定的单步 decode 输入 buffer(地址固定,graph 捕获/replay 复用同一块)。
        static_input = torch.randint(0, vocab, (1, 1), device=device)
        static_cpos = torch.tensor([prefill_len], device=device, dtype=torch.long)
        # 捕获前必须在 side stream 上 warmup,完成 lazy init / autotune。
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(static_input, use_cache=True, past_key_values=static_cache,
                      cache_position=static_cpos)
        torch.cuda.current_stream().wait_stream(s)
        # 捕获单步 decode。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(static_input, use_cache=True, past_key_values=static_cache,
                  cache_position=static_cpos)
        return graph, max_len
    except Exception as e:   # noqa: BLE001 — 探针,任何失败都降级并记录原因
        return {"error": repr(e)}


@torch.no_grad()   # 性能测试无需梯度
def profile_latency(
    model,                       # 已 eval 的模型
    tokenizer,                   # 预留(当前用随机 token,不实际编码文本)
    prefill_len: int = 512,      # prefill 阶段输入长度
    decode_tokens: int = 128,    # decode 阶段生成的 token 数
    warmup: int = 2,             # 预热轮数(不计入统计,排除首次编译/缓存开销)
    repeats: int = 5,            # 正式测量轮数,取中位
    device: str = None,          # 设备,None 取模型所在设备
    past_pruner=None,            # 可选:M2-c KV cache 裁剪回调
    reset_peak_after_prefill: bool = False,  # 可选:只统计 decode 阶段峰值显存
    compile_decode: bool = False,  # 可选:M2-e 验证,手动 CUDA graph 单步 decode(量 overhead 上限)
) -> dict:
    """测量 prefill 延迟与 decode 吞吐,均取中位数以抵抗抖动。

    `compile_decode=True` 时额外测一条 torch.compile + StaticCache 的无裁剪 decode,
    用来验证 decode 是否 overhead-bound(见 `_try_build_static_decode`)。该分支与
    `past_pruner` 互斥(static cache 形状固定,不能每步裁剪),结果写入 `compiled_decode`。
    """
    if device is None:
        device = next(model.parameters()).device
    vocab = model.config.vocab_size   # 词表大小,用于生成合法随机 token
    # 构造随机输入 [1, prefill_len];测延迟只关心计算量,内容无所谓
    input_ids = torch.randint(0, vocab, (1, prefill_len), device=device)

    prefill_times, decode_tps = [], []   # 分别收集各轮 prefill 耗时 / decode 吞吐
    prune_applied_steps = 0
    prune_last = None
    pass_cache_position = past_pruner is not None and _supports_kwarg(model.forward, "cache_position")
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
        past, stats = _apply_past_pruner(past, past_pruner)
        prune_last = stats or prune_last
        if stats and stats.get("applied"):
            prune_applied_steps += 1
        if reset_peak_after_prefill and str(device).startswith("cuda"):
            # M2-c decode-only 显存口径:prefill 用来建初始 cache,但峰值从裁剪后开始统计。
            torch.cuda.reset_peak_memory_stats()

        # ---- decode 阶段:自回归逐 token 生成 ----
        _sync(device)
        t2 = time.perf_counter()
        for step in range(decode_tokens):
            # 每步只喂 1 个新 token + 复用 past_key_values,模拟真实解码
            kwargs = {"past_key_values": past, "use_cache": True}
            if pass_cache_position:
                # 裁剪后 cache 物理长度变短,但 RoPE 位置仍需按原始绝对 token 递增。
                kwargs["cache_position"] = torch.tensor(
                    [prefill_len + step], device=device, dtype=torch.long)
            out = model(nxt, **kwargs)
            past = out.past_key_values           # 滚动更新 KV cache
            past, stats = _apply_past_pruner(past, past_pruner)
            prune_last = stats or prune_last
            if stats and stats.get("applied"):
                prune_applied_steps += 1
            nxt = out.logits[:, -1:].argmax(-1)  # 贪心取下一 token
        _sync(device)
        t3 = time.perf_counter()

        if i >= warmup:   # 跳过预热轮,只统计稳定态
            prefill_times.append(t1 - t0)                    # 本轮 prefill 秒数
            decode_tps.append(decode_tokens / (t3 - t2))     # 本轮 decode 吞吐 tok/s

    result = {
        "prefill_len": prefill_len,       # 记录配置
        "decode_tokens": decode_tokens,
        # 中位数比均值更抗个别毛刺;prefill 转毫秒更直观
        "prefill_ms_median": round(statistics.median(prefill_times) * 1e3, 3),
        "decode_tps_median": round(statistics.median(decode_tps), 2),
    }
    if past_pruner is not None:
        result["cache_pruning"] = {
            "applied_steps": prune_applied_steps,
            "last": prune_last,
            "cache_position_passed": pass_cache_position,
            "reset_peak_after_prefill": reset_peak_after_prefill,
        }

    # ---- 可选:M2-e 验证分支,torch.compile + StaticCache 无裁剪 decode ----
    # 目的是量出消除逐步 overhead 后 decode 的吞吐上限,与上面的 eager decode 对比。
    # 若 compile 后 tok/s 大幅跳升,则坐实 decode 为 overhead-bound(而非 KV-bound),
    # 固定大小 ring-buffer KV cache 值得做;若几乎不变,则瓶颈另有其处。
    if compile_decode:
        built = _try_build_cudagraph_decode(model, prefill_len, decode_tokens, device)
        if built is None:
            result["compiled_decode"] = {
                "available": False,
                "reason": "not_cuda_or_static_cache_unavailable",
            }
        elif isinstance(built, dict):   # 捕获过程抛错,记录原因
            result["compiled_decode"] = {"available": False, "reason": built["error"]}
        else:
            graph, max_len = built
            graph_tps = []
            for i in range(warmup + repeats):
                # replay 同一张 graph decode_tokens 次;每步计算量恒定,只测
                # 消除 launch overhead 后的单步纯执行时间(不追求 token 正确性)。
                _sync(device)
                t2 = time.perf_counter()
                for _ in range(decode_tokens):
                    graph.replay()
                _sync(device)
                t3 = time.perf_counter()
                if i >= warmup:
                    graph_tps.append(decode_tokens / (t3 - t2))
            result["compiled_decode"] = {
                "available": True,
                "method": "manual_cudagraph_replay",
                "decode_tps_median": round(statistics.median(graph_tps), 2),
                "max_cache_len": max_len,
                "speedup_vs_eager": round(
                    statistics.median(graph_tps)
                    / max(result["decode_tps_median"], 1e-6), 3),
            }

    return result


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
