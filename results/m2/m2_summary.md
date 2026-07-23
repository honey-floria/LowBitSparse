# M2 稀疏注意力结果汇总

环境: `NVIDIA A100-SXM4-40GB` / `torch 2.11.0+cu128` / `CUDA 12.8`

结论: 三种稀疏模式都已跑通,但当前实现没有带来加速;StreamingLLM 质量最好,Sliding Window 质量最差。

## 均值对比

`memory_delta_mb` 为 `baseline_peak - sparse_peak`,负值表示 sparse 峰值更高。

| 模式 | 平均 prefill speedup | 平均 decode speedup | 平均 memory delta MB | 平均 ΔPPL |
| --- | --- | --- | --- | --- |
| Sliding Window w1024 | 0.659x | 0.871x | -170.0 | +44.871 |
| StreamingLLM s64 w1024 | 0.659x | 0.871x | -170.5 | +0.841 |
| Block-sparse b128 l1 | 0.656x | 0.869x | -170.0 | +4.519 |

## 关键点

- 最好的 prefill 也只有 2048 长度下约 0.92x，最长 16384 长度掉到约 0.39x。
- decode speedup 全部小于 1。
- peak memory 没有下降，最长序列反而多出约 512 MB。
- 质量上 StreamingLLM 最稳，Sliding Window 的 PPL 恶化最重。

## 结果文件

- `results/m2_sparse_sliding_w1024.json`
- `results/m2_sparse_streaming_s64_w1024.json`
- `results/m2_sparse_block_b128_l1.json`
