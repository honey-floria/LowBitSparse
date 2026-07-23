# M2-c StreamingLLM KV prune 回填

结果文件: `results/m2c_streaming_kvprune_s64_w1024.json`

## 结论
新版 HF cache 容器兼容层生效后,A100 重跑已经确认 KV cache 真实裁剪:
`applied_steps=903`,`layers=24`,`kept_len=1088`(sink 64 + window 1024)。

M2-c 三项目标里 2 项达标:质量达标、显存达标;decode 加速未达标。

## 平均值
| 指标 | baseline | prune | 变化 |
| --- | --- | --- | --- |
| decode tok/s | 37.71 | 34.36 | 0.911x |
| peak memory saved | - | - | +157.19 MB |
| ΔPPL | - | - | +0.841 |

## 分长度
| seqlen | decode speedup | peak saved MB | ΔPPL | cache prune |
| --- | --- | --- | --- | --- |
| 2048 | 0.907x | +23.999 | +0.242 | applied, kept_len=1088 |
| 4096 | 0.915x | +75.906 | +0.707 | applied, kept_len=1088 |
| 8192 | 0.915x | +168.311 | +1.061 | applied, kept_len=1088 |
| 16384 | 0.907x | +360.542 | +1.354 | applied, kept_len=1088 |

## 解释
KV 裁剪解决了长序列显存,但没有解决 Qwen2.5-0.5B 在 A100 上的 decode 延迟。
裁剪后每步仍受 Python/kernel launch overhead 与权重带宽主导,而每步 24 层 cache 裁剪
记账带来额外开销,所以 decode 约慢 9%。
