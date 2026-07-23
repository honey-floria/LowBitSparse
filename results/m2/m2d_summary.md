# M2-d StreamingLLM chunked prefill 回填

结果文件: `results/m2d_streaming_chunked_s64_w1024_c512.json`

## 结论
M2-d 的 chunked prefill + KV prune 路径已经真实生效:prefill 按 512-token
chunk 推进,chunk 间 cache 稳定裁到 `kept_len=1088`(sink 64 + window 1024)。

它达成了长序列显存目标,但没有达成 prefill 加速目标:平均 prefill speedup
只有 `0.198x`,原因是 chunked 路径把一次 dense prefill 拆成 4-32 次 forward,
Python/kernel launch 与 cache 裁剪开销显著增加。

## 平均值
| 指标 | baseline | chunked | 变化 |
| --- | --- | --- | --- |
| prefill latency | 97.03 ms | 513.39 ms | 0.198x |
| decode tok/s | 37.75 | 34.02 | 0.901x |
| peak memory saved | - | - | +2105.74 MB |
| ΔPPL | - | - | +0.841 |

## 分长度
| seqlen | baseline prefill | chunked prefill | prefill speedup | decode speedup | peak saved MB | ΔPPL | chunks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 32.451 ms | 130.219 ms | 0.249x | 0.899x | +330.911 | +0.242 | 4 |
| 4096 | 45.857 ms | 268.202 ms | 0.171x | 0.915x | +980.317 | +0.707 | 8 |
| 8192 | 100.381 ms | 549.883 ms | 0.183x | 0.895x | +2265.722 | +1.061 | 16 |
| 16384 | 209.428 ms | 1105.268 ms | 0.189x | 0.895x | +4846.022 | +1.354 | 32 |

## 解释
M2-d 证明 chunked prefill 能把长序列显存从完整历史 cache 降到窗口级 cache,
16k 时节省约 4.85GB。但在 Qwen2.5-0.5B + A100 上,一次 dense prefill 已经很快,
拆 chunk 后多次 forward 和每层 cache 裁剪开销盖过了显存收益,所以速度显著变慢。

结论:保留 M2-d 作为显存优先/超长上下文兜底路径;常规速度目标继续依赖 M2-e
ring-buffer + CUDA graph 的 decode 路径。
