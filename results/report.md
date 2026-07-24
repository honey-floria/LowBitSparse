# LowBitSparse M4 报告

生成时间: `2026-07-24T08:17:25.333557+00:00`
数据来源: `results`，共读取 `31` 个 JSON。

## 结论摘要

- 压缩: GPTQ INT4 + embedding INT8 达到 315.3 MB / 2.988x，PPL 15.4275；embedding INT4 达到 250.4 MB / 3.763x，但 PPL 升至 16.6881。
- 精度恢复: M3 蒸馏把 RTN INT4 student 从 15.9786 拉到 14.2716，恢复 teacher-student 缺口 63.0%，压缩比保持 2.136x。
- 加速: M2-e ring-buffer + CUDA graph 在 2k/4k/8k/16k 上平均 decode 5.331x，长序列 decode 显存节省随长度增加。
- 组合: 当前组合项为独立实测结果的派生汇总，不声称已经完成量化+稀疏+蒸馏的同一模型端到端联合评测。
- 1.5B: 本地和结果目录没有 1.5B 实测 JSON；报告不把 1.5B 外推当作结论。

## 基线

| 模型 | PPL | 体积MB | prefill ms | decode tok/s | peak MB |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 14.2445 | 942.3 | 28.89 | 38.20 | 4574.3 |

## 曲线一: 压缩比 vs PPL

| 实验 | 方法 | bit | group | PPL | ΔPPL | 体积MB | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m1_rtn_int8_g128 | rtn | 8 | 128 | 14.2326 | -0.0119 | 611.7 | 1.540x |
| m1_rtn_int4_g128 | rtn | 4 | 128 | 17.0580 | 2.8135 | 441.1 | 2.136x |
| m1_gptq_int4_g128 | gptq | 4 | 128 | 15.4347 | 1.1902 | 441.1 | 2.136x |
| m1_gptq_int4_embint8 | gptq | 4 | 128 | 15.4275 | 1.1830 | 315.3 | 2.988x |
| m1_gptq_int4_embint4 | gptq | 4 | 128 | 16.6881 | 2.4436 | 250.4 | 3.763x |

## 曲线二: 长序列稀疏加速

| 实验 | 类型 | avg ΔPPL | prefill | decode | avg mem delta MB |
| --- | --- | --- | --- | --- | --- |
| m2_sparse_block_b128_l1 | - | 4.519 | 0.656x | 0.869x | -170.0 |
| m2_sparse_sliding_w1024 | - | 44.871 | 0.659x | 0.871x | -170.0 |
| m2_sparse_streaming_s64_w1024 | - | 0.841 | 0.659x | 0.871x | -170.5 |
| m2c_streaming_kvprune_s64_w1024 | streaming_kv_pruning | 0.841 | 0.998x | 0.906x | 157.2 |
| m2d_streaming_chunked_s64_w1024_c512 | streaming_chunked_prefill | 0.841 | 0.198x | 0.901x | 2105.7 |
| m2e_streaming_compile_probe | - | 0.841 | 0.662x | 0.881x | -170.0 |
| m2e_streaming_ringgraph_s64_w1024 | streaming_ring_graph | 0.841 | 1.000x | 5.331x | 137.9 |
| m2e_streaming_ringgraph_s64_w1024_1.5b | streaming_ring_graph | 0.434 | 1.000x | 4.148x | 371.9 |

M2-e 逐长度曲线:

| seqlen | ΔPPL | prefill speedup | decode speedup | memory delta MB |
| --- | --- | --- | --- | --- |
| 2048 | 0.242 | 1.000x | 5.341x | 18.5 |
| 4096 | 0.707 | 1.000x | 5.323x | 61.2 |
| 8192 | 1.061 | 1.000x | 5.287x | 144.5 |
| 16384 | 1.353 | 1.000x | 5.373x | 327.5 |

## 曲线三: 蒸馏恢复

| step | PPL |
| --- | --- |
| 0 | 15.9786 |
| 20 | 15.0685 |
| 40 | 14.7596 |
| 60 | 14.5166 |
| 80 | 14.3035 |
| 100 | 14.2716 |

## 组合汇总

| 组合 | 状态 | 短上下文PPL | 体积MB | 压缩比 | 长上下文ΔPPL参考 | decode |
| --- | --- | --- | --- | --- | --- | --- |
| GPTQ INT4 + emb INT8 + ring-graph sparse | derived | 15.4275 | 315.3 | 2.988x | 0.841 | 5.331x |
| GPTQ INT4 + emb INT4 + ring-graph sparse | derived | 16.6881 | 250.4 | 3.763x | 0.841 | 5.331x |
| M3 distilled RTN INT4 + ring-graph sparse | derived | 14.2716 | 441.1 | 2.136x | 0.841 | 5.331x |

## 1.5B 复现状态

未发现 `qwen1.5b` 相关结果 JSON。当前 M4 报告只对 0.5B 实测结果负责；1.5B 复现需要在 A100/Colab 上补跑后重新生成本报告。

建议补跑命令:

```bash
python main.py eval --config configs/qwen1.5b_base.yaml
python main.py quant --config configs/qwen1.5b_gptq_int4_embint8.yaml
python main.py sparse --config configs/qwen1.5b_sparse_streaming_ringgraph.yaml
python scripts/build_m4_report.py
```

## 最终判断

0.5B 主线已经闭合:推荐路径是 GPTQ INT4 + embedding INT8 作为压缩默认点；需要更接近 FP16 精度时，用 M3 distilled RTN INT4；需要长序列 decode 加速时，用 M2-e ring-buffer + CUDA graph。M2-d 只作为显存优先的超长 prefill 兜底路径。
