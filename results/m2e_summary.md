# M2-e StreamingLLM ring-buffer + CUDA graph 回填

结果文件: `results/m2e_streaming_ringgraph_s64_w1024.json`

## 结论
M2-e 把 decode 从 M2-c 的 `0.911x` 反转为平均 `5.331x`。
核心收益来自 CUDA graph 消除逐 token overhead;ring-buffer 固定 cache 长度为
1088,让长序列下 graph replay 的形状稳定且显存随窗口封顶。

质量口径仍是 StreamingLLM additive mask 的 teacher-forced PPL 参考,不是完整生成质量验证。

## 平均值
| 指标 | baseline | ring+graph | 变化 |
| --- | --- | --- | --- |
| decode tok/s | 36.29 | 193.43 | 5.331x |
| peak memory saved | - | - | +137.94 MB |
| ΔPPL | - | - | +0.841 |

## 分长度
| seqlen | baseline decode | ring+graph decode | decode speedup | peak saved MB | ΔPPL |
| --- | --- | --- | --- | --- | --- |
| 2048 | 36.43 | 194.58 | 5.341x | +18.482 | +0.242 |
| 4096 | 36.16 | 192.48 | 5.323x | +61.248 | +0.707 |
| 8192 | 36.48 | 192.86 | 5.287x | +144.497 | +1.061 |
| 16384 | 36.07 | 193.80 | 5.373x | +327.541 | +1.354 |

## 边界
当前 ring+graph 是 benchmark proof,不是可直接替换 `generate()` 的完整实现:
还没有做 RoPE 相位忠实修正,也没有验证生成 token 与 dense/StreamingLLM 语义一致。
