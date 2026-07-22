"""M2-e 前置:CUDA graph decode 可行性探针(独立脚本,不接 benchmark/稀疏)。

背景:forward_scaling 探针已坐实 decode 是 overhead-bound —— 单次 forward 有 ~26.5ms
的固定地板(kernel launch + Python 调度),与 token 数几乎无关(512 token 仅比 1 token
慢 9%)。消除这个地板的正解是 CUDA graph(固定形状 + 一次 launch replay),理论收益
可达数十倍。但 CUDA graph 在本环境(transformers + StaticCache)已两次失败:
  ① torch.compile(reduce-overhead) 的 cudagraphs 被 cache update() 的原地
     cumulative_length.add_() 判为 mutated inputs 自动跳过;
  ② 手动 torch.cuda.CUDAGraph() + StaticCache 触发 index_copy_ 越界 device-assert,
     污染 CUDA context 带崩整个 benchmark。

本脚本把 CUDA graph decode 从 benchmark 里隔离出来,**分阶段 + 每阶段 sync**,精确定位
上次崩在哪一步,并逐步排查 StaticCache 的正确用法。任何一步失败都打印清晰标记后退出,
不会污染后续(每个 STAGE 独立 try)。目标:先让 replay 在 A100 上不崩地跑出一个 tok/s,
确认可行后再规划完整 ring-buffer M2-e。

用法:
    python scripts/cudagraph_probe.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --prefill 2048 --decode 128
"""
import argparse
import statistics
import time


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stage(name):
    print(f"\n[STAGE] {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import torch
    from lowbitsparse.models import load_model_and_tokenizer

    _stage("0. load model")
    model, _ = load_model_and_tokenizer(
        model_name=args.model, dtype=args.dtype, device=args.device)
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vocab = model.config.vocab_size
    print(f"  device={device} dtype={dtype} vocab={vocab}")
    if not str(device).startswith("cuda"):
        print("  [SKIP] 非 cuda 设备,CUDA graph 无意义")
        return

    # --- STAGE 1: StaticCache 能否构造 ---
    _stage("1. build StaticCache")
    try:
        from transformers import StaticCache
        max_len = args.prefill + args.decode + 1
        cache = StaticCache(config=model.config, max_batch_size=1,
                            max_cache_len=max_len, device=device, dtype=dtype)
        _sync()
        print(f"  OK max_cache_len={max_len}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE1] {e!r}")
        return

    # --- STAGE 2: prefill 填充 cache(上次疑似崩在这里的 cache_position) ---
    _stage("2. prefill fill cache")
    try:
        with torch.no_grad():
            ids = torch.randint(0, vocab, (1, args.prefill), device=device)
            pos = torch.arange(args.prefill, device=device, dtype=torch.long)
            out = model(ids, use_cache=True, past_key_values=cache,
                        cache_position=pos)
            nxt = out.logits[:, -1:].argmax(-1)
        _sync()
        print(f"  OK prefill_len={args.prefill}, next token shape={tuple(nxt.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE2] {e!r}")
        return

    # --- STAGE 3: 单步 eager decode(不捕获,先确认 cache_position 用法正确) ---
    # 这是上次 device-assert 最可疑的点:decode 时 cache_position 必须指向 cache 里
    # 下一个"绝对写入槽位",越界就会触发 index_copy_ 的 device assert。
    _stage("3. single eager decode step (no graph)")
    try:
        with torch.no_grad():
            static_input = nxt.clone()                     # [1,1]
            static_cpos = torch.tensor([args.prefill], device=device, dtype=torch.long)
            out = model(static_input, use_cache=True, past_key_values=cache,
                        cache_position=static_cpos)
        _sync()
        print(f"  OK single decode step, logits shape={tuple(out.logits.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE3] {e!r}  <- cache_position 越界最可能在此暴露")
        return

    # --- STAGE 4: side-stream warmup(捕获前必需) ---
    _stage("4. side-stream warmup")
    try:
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(args.warmup):
                    model(static_input, use_cache=True, past_key_values=cache,
                          cache_position=static_cpos)
            torch.cuda.current_stream().wait_stream(s)
        _sync()
        print(f"  OK warmup x{args.warmup}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE4] {e!r}")
        return

    # --- STAGE 5: 捕获单步 decode ---
    _stage("5. capture CUDA graph")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_out = model(static_input, use_cache=True,
                                   past_key_values=cache, cache_position=static_cpos)
        _sync()
        print("  OK graph captured")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE5] {e!r}")
        return

    # --- STAGE 6: replay 计时 ---
    _stage("6. replay timing")
    try:
        tps = []
        for i in range(args.warmup + args.repeats):
            _sync()
            t0 = time.perf_counter()
            for _ in range(args.decode):
                graph.replay()
            _sync()
            t1 = time.perf_counter()
            if i >= args.warmup:
                tps.append(args.decode / (t1 - t0))
        graph_tps = statistics.median(tps)
        step_ms = 1e3 / graph_tps
        print(f"  OK graph replay decode = {graph_tps:.1f} tok/s ({step_ms:.3f} ms/step)")
        print(f"\n[RESULT] eager 地板 ~27ms/step (37 tok/s) → graph {step_ms:.3f} ms/step "
              f"({graph_tps:.1f} tok/s), 提速 ~{graph_tps/37:.1f}x")
        print("  若提速显著(>3x),则 overhead 确可被 graph 消除,ring-buffer M2-e 可行。")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL @STAGE6] {e!r}")
        return


if __name__ == "__main__":
    main()
