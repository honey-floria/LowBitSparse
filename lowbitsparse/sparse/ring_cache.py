"""M2-e:有界 ring-buffer KV cache + CUDA graph decode。

背景(见 doc/OPTIMIZATION.md M2-e 探针条目):0.5B decode 是 overhead-bound —— 单次
forward ~26.5ms 固定地板(kernel launch + Python 调度),与 token 数几乎无关。消除它的
正解是 CUDA graph(固定形状 + 一次 launch replay),探针实测 decode 37→113 tok/s(3.1x)。

CUDA graph 要求每步张量形状恒定。StaticCache 满足形状恒定,但它有个随每次 forward 自增
的内部计数器(cumulative_length),被烘焙进 graph 后盲目 replay 会单调累加至 index_copy_
越界 device-assert(探针 STAGE6 崩因)。

RingKVCache 的解法:持有固定 `sink+window` 大小的 K/V buffer,`update` 走**回绕写入**——
sink 段(前 sink 个 token)永久保留,window 段循环覆盖最老槽位。写入位置由 Python 计数器
算出一个**合法且有界**的 slot;graph 捕获时该 slot 被烘焙为常量,replay 反复写同一合法槽位
→ 永不越界。这正是 ring-buffer 相对 StaticCache 的关键差别。

范围(M2-e Benchmark 证明路线):latency/memory 走此真实执行路径;quality 仍由 benchmark
侧用 StreamingLLM additive mask 的 teacher-forced PPL 作参考。**不做 RoPE 相位忠实修正**,
graph 路径下不追求生成 token 的正确性(回绕的旧 key 带旧相位,仅用于测速/测显存)。
"""
from __future__ import annotations

import statistics
import time

import torch


class RingKVCache:
    """固定大小 sink+window 的回绕 KV cache(duck-typed HF Cache 最小接口)。

    每层持有形状 `[batch, n_kv_heads, sink+window, head_dim]` 的 K/V buffer,大小恒定,
    decode 阶段显存不随序列增长。`update` 返回的也是这块恒定形状 buffer,故 attention
    的输入形状每步不变,CUDA graph 可反复 replay。
    """

    def __init__(self, sink_size: int, window_size: int):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.total = self.sink_size + self.window_size
        # 每层一份 buffer,惰性按首次写入的形状/设备/dtype 分配。
        self.key_buf: dict[int, torch.Tensor] = {}
        self.value_buf: dict[int, torch.Tensor] = {}
        # window 段的下一个写入偏移(0..window_size-1),按层独立推进。
        self._wptr: dict[int, int] = {}
        # 已写入 token 数(用于 get_seq_length;上限 total)。
        self._filled: dict[int, int] = {}

    # ---- HF Cache duck-typed 接口 ----
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self._filled.get(layer_idx, 0))

    def get_max_cache_shape(self):
        return self.total

    def get_max_length(self):
        return self.total

    def _ensure_buf(self, key_states: torch.Tensor, layer_idx: int):
        """按首次写入的 [B, H, *, D] 分配该层恒定 buffer。"""
        if layer_idx in self.key_buf:
            return
        b, h, _, d = key_states.shape
        shape = (b, h, self.total, d)
        self.key_buf[layer_idx] = torch.zeros(
            shape, device=key_states.device, dtype=key_states.dtype)
        self.value_buf[layer_idx] = torch.zeros(
            shape, device=key_states.device, dtype=key_states.dtype)
        self._wptr[layer_idx] = 0
        self._filled[layer_idx] = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs=None):
        """写入新 K/V 并返回恒定形状的完整 buffer。

        - prefill(seq_len > 1):填 sink 段(前 sink 个)+ window 段(最后 window 个),
          置好 window 写指针。
        - decode(seq_len == 1):写 window 段当前槽位并推进指针(回绕)。graph 捕获时
          该槽位被烘焙为常量,replay 反复写同一合法槽位,不越界。
        """
        self._ensure_buf(key_states, layer_idx)
        kbuf = self.key_buf[layer_idx]
        vbuf = self.value_buf[layer_idx]
        seq_len = key_states.shape[-2]
        sink, win, total = self.sink_size, self.window_size, self.total

        if seq_len > 1:
            # prefill:把 sink+window 铺进 buffer 供后续 decode 用,但**返回真实完整
            # K/V**——prefill 的 q_len 与 kv_len 必须一致,attention mask 才对得上;若返回
            # 截短的 buffer(1088)而 q_len=2048 会形状不匹配直接报错,连 next token 都拿不到。
            n_sink = min(sink, seq_len)
            if n_sink > 0:
                kbuf[:, :, :n_sink] = key_states[:, :, :n_sink]
                vbuf[:, :, :n_sink] = value_states[:, :, :n_sink]
            n_win = min(win, max(seq_len - n_sink, 0))
            if n_win > 0:
                kbuf[:, :, sink:sink + n_win] = key_states[:, :, seq_len - n_win:]
                vbuf[:, :, sink:sink + n_win] = value_states[:, :, seq_len - n_win:]
            self._wptr[layer_idx] = n_win % win if win > 0 else 0
            self._filled[layer_idx] = min(n_sink + n_win, total)
            return key_states, value_states
        else:
            # decode:回绕写 window 段一个槽位,返回**恒定形状 buffer**——这是 CUDA graph
            # 能反复 replay 的关键(形状每步不变,写入槽位有界合法,永不越界)。
            if win > 0:
                slot = sink + self._wptr[layer_idx]
                kbuf[:, :, slot:slot + 1] = key_states
                vbuf[:, :, slot:slot + 1] = value_states
                self._wptr[layer_idx] = (self._wptr[layer_idx] + 1) % win
            self._filled[layer_idx] = min(self._filled.get(layer_idx, 0) + 1, total)
            return kbuf, vbuf

    def reset(self):
        """清零 buffer 与指针(复用同一 cache 跑多轮)。"""
        for k in self.key_buf:
            self.key_buf[k].zero_()
            self.value_buf[k].zero_()
            self._wptr[k] = 0
            self._filled[k] = 0


def build_ring_graph_decode(model, sink_size: int, window_size: int,
                            prefill_len: int, decode_tokens: int, device,
                            warmup: int = 3, repeats: int = 5) -> dict:
    """用 RingKVCache 捕获并 replay 单步 decode 的 CUDA graph,返回吞吐/显存。

    复用探针脚本已验证的流程:build ring cache → prefill 填充(自动裁到 sink+window)→
    单步 eager sanity → side-stream warmup → capture → replay 计时。任何一步失败返回
    `{available: False, reason}`。device-assert 无法 try 兜,靠 buffer 恒定大小预防越界。

    返回 `{available, method, decode_tps_median, decode_peak_mb, cache_len, ...}`。
    """
    if not str(device).startswith("cuda"):
        return {"available": False, "reason": "not_cuda"}
    try:
        vocab = model.config.vocab_size
        cache = RingKVCache(sink_size, window_size)
        with torch.no_grad():
            # prefill:cache 内部只保留 sink+window,buffer 大小恒定。
            ids = torch.randint(0, vocab, (1, prefill_len), device=device)
            pos = torch.arange(prefill_len, device=device, dtype=torch.long)
            out = model(ids, use_cache=True, past_key_values=cache, cache_position=pos)
            nxt = out.logits[:, -1:].argmax(-1)
            # 单步 decode 的 cache_position 固定指向 window 之后的下一个绝对位置。
            static_input = nxt.clone()
            static_cpos = torch.tensor([prefill_len], device=device, dtype=torch.long)
            # 单步 eager sanity(不捕获)。
            model(static_input, use_cache=True, past_key_values=cache,
                  cache_position=static_cpos)
            torch.cuda.synchronize()
            # side-stream warmup(捕获前必需)。
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    model(static_input, use_cache=True, past_key_values=cache,
                          cache_position=static_cpos)
            torch.cuda.current_stream().wait_stream(s)
            # capture 单步 decode。
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                model(static_input, use_cache=True, past_key_values=cache,
                      cache_position=static_cpos)
            # replay 计时,并量 decode 阶段峰值显存(恒定 cache 下应不随长度增长)。
            torch.cuda.reset_peak_memory_stats()
            tps = []
            for i in range(warmup + repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(decode_tokens):
                    graph.replay()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                if i >= warmup:
                    tps.append(decode_tokens / (t1 - t0))
            peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        return {
            "available": True,
            "method": "ring_cuda_graph_replay",
            "decode_tps_median": round(statistics.median(tps), 2),
            "decode_peak_mb": peak_mb,
            "cache_len": sink_size + window_size,
            "sink_size": sink_size,
            "window_size": window_size,
        }
    except Exception as e:   # noqa: BLE001 — 探针,任何失败降级并记录原因
        return {"available": False, "reason": repr(e)}
