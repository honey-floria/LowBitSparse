# LowBitSparse 优化与复盘记录

> 本文件记录项目**每一步的分析、优化决策、实验结果与复盘**。
> 与 `TODO.md`(规划/设计/任务)配套:TODO 说「要做什么」,本文件说「做了什么、为什么、效果如何、下一步」。

---

## 使用说明
- 按时间倒序或里程碑顺序追加条目,不删除历史(错误记录也有价值)。
- 每条建议用统一模板(见下)。基线数据、关键曲线截图路径一并记录。
- 每个里程碑收尾写一段「里程碑复盘」。

### 条目模板
```
## [日期] <标题 / 里程碑-步骤>
**背景 / 目标**:要解决什么问题,为什么做。
**分析**:观察到的现象、瓶颈定位、假设。
**方案**:采取的做法(含被否决的备选与原因)。
**实验设置**:模型 / bit / 数据 / 超参 / 环境。
**结果**:关键指标(前→后),曲线/表格路径。
**结论 & 下一步**:是否达标,遗留问题,后续动作。
```

---

## 基线速查表(持续更新)
| 项 | 模型 | 配置 | PPL(WikiText-2) | 体积 | prefill/decode 延迟 | 显存峰值 |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | Qwen2.5-0.5B-Instruct | seqlen/stride 2048;prefill512/decode128 | 14.2445 | 942.3 MB(494.03M 参数) | 30.42 ms / 37.19 tok/s | 4574.3 MB(current 959.3 MB) |

> 环境:A100-SXM4-40GB,torch 2.11.0+cu128,CUDA 12.8。数据源 `results/m0_fp16_baseline.json`。
> 压缩比基准(体积分母)= 942.3 MB;INT8 目标 ≈471 MB(2x),INT4 目标 ≈235–270 MB(3.5–4x)。

---

## 复盘记录

## [2026-07-20] M0 — 脚手架与基线(已在 A100 实跑完成)
**背景 / 目标**:搭好可复现骨架,拿到 FP16 基线,作为一切压缩的对照。

**分析**:
- 原仓库仅有 PyCharm 示例 `main.py` 和空 `doc/`,一切从零开始。
- 关键约束是 Colab 会断连,基线数据必须可复现且能落 Drive,否则后续 M1-M4 的相对指标无参照。
- 本地 Windows 无 GPU、无 torch,故把"数值/模型相关"逻辑延迟到函数内 import,使脚手架接线能在本地验证,重活留给 A100。

**方案**:
- 包结构 `lowbitsparse/{models,quant,sparse,distill,eval,utils}`,CLI 分 `eval|quant|sparse|distill` 四子命令,M0 只实现 `eval`,其余为占位(调用即报错提示对应里程碑)。
- PPL 用 strided 滑窗法(stride<seqlen 时只对新 token 计损),对齐社区标准算法,避免不重叠切分高估 PPL。
- profiler 分离 prefill 延迟与 decode 吞吐,warmup + 中位数,降低抖动。
- 结果统一 `save_results` 落 `results/<exp_id>.json`,内嵌 env(torch/cuda/gpu)保证可复现。

**实验设置**:模型 Qwen2.5-0.5B-Instruct;FP16;seqlen/stride 2048;prefill 512 / decode 128。

**结果**(A100-40GB,见 `results/m0_fp16_baseline.json`):
- 本地验证:全文件 `py_compile` 通过;接线跑通;占位命令正确抛出。
- 体积:494.03M 参数 × 2B(FP16)= 942.3 MB,与理论值自洽,作为压缩比分母。
- 精度:WikiText-2 PPL = 14.2445(seqlen/stride 2048),量级正常,可作参照点。
- 延迟:prefill 512 tok = 30.42 ms(正常);decode = 37.19 tok/s(偏低,见下方分析)。
- 显存:峰值 4574 MB / 40GB,富余极大;current 959 MB ≈ 权重常驻。

**关键分析 / 踩坑**:
1. **decode 吞吐是"开销受限"假象,非模型瓶颈**。0.5B 在 A100 上按显存带宽估算理论上限 ~1600 tok/s,实测仅 37(~2%)。原因是逐 token Python 循环 + eager 模式下每步几百个 kernel 的固定启动开销主导,而 0.5B 单步真实计算极小。→ **影响 M2**:稀疏加速只在长序列(注意力占比高)显现;当前 seqlen=512 且开销受限,加速比会被吃掉。
2. **PPL 用了 stride=2048(不重叠),会轻微高估**。不改——基线价值在横向可比,M1-M4 全程同 stride 即可,量化前后差值有效;改了要重跑基线。
3. **wikitext 数据集 id 需带命名空间**:新版 huggingface_hub 拒绝裸 `wikitext`,已改用 `Salesforce/wikitext`。

**结论 & 下一步**:
- M0 验收达标,基线已入速查表。
- 遗留待办:①M2 延迟评测改用长序列(2k/4k/8k/16k),不用 512;②进入 M1,先实现 RTN 伪量化打通量化→评测闭环,压缩比基准 942.3 MB 已确定。

## [2026-07-21] M1-a — RTN 伪量化(代码+本地验证完成,待 A100 实测)
**背景 / 目标**:打通"量化→评测"闭环,先用最简单的 RTN 拿到 INT8/INT4 的 PPL 与压缩比,作为 GPTQ/AWQ 的对照。

**分析 / 设计决策**:
- **伪量化(fake-quant)而非真 INT kernel**:权重量化-反量化后仍以 FP16 存,forward 走普通 matmul。这样能真实反映低 bit 对精度的影响,又无需 INT4 kernel,纯 PyTorch 可跑;压缩比用"理论体积"解析计算而非实际磁盘。
- **构造时量化一次并缓存**:FakeQuantLinear 在 __init__ 里量化一次存为 buffer,不在每步 forward 重复量化 —— PPL 评测要多次前向,缓存显著更快。代价是权重变只读(M3 QAT 时再改可训练 scale)。
- **非对称为默认**:RTN 下非对称(带 zero)通常比对称精度好;对称省 1 份 zero 存储,留作消融。
- **非整除 group 用右侧 padding**:in_features 未必被 group_size 整除(如 Qwen 896 不整除 256),pad 到整数倍再向量化,最后切回,避免 Python 循环。
- **压缩比口径**:量化权重按 n_bits + 每组 scale/zero(FP16)开销计,其余(embedding/norm/bias/lm_head)按真实 dtype 计。故等效 bit 会略高于 n_bits(实测 tiny 模型 4bit→4.299)。

**关键预判(待 A100 验证)**:
- 0.5B 的 embedding 约占 27% 参数且不量化,故**整体压缩比会低于"权重 4x"的直觉**,预计 INT4 整体约 2–2.5x,而非 TODO 里理想的 3.5–4x。这不是 bug,是小模型 embedding 占比高的必然结果 —— 报告时要把"量化层等效 bit"与"整体压缩比"分开讲。

**实验设置**:Qwen2.5-0.5B;RTN;n_bits∈{8,4};group_size=128;非对称;skip=lm_head。

**结果**:
- 本地(CPU torch 2.13)全部通过:5 条 RTN 单测(shape/dtype、位宽单调性 INT8<INT4<INT3、INT8 相对误差<1%、padding、对称路径);合成 tiny 模型端到端替换+forward+压缩统计正常。
- 0.5B 实测 PPL/压缩比:**待 A100 跑** `python main.py quant --config configs/qwen0.5b_int8.yaml` 与 `..._int4.yaml`。

**结论 & 下一步**:
- RTN 闭环达标。下一步 A100 实测 INT8(健全性,PPL 应接近 FP16 14.24)→ INT4(观察退化)→ group_size 扫描 → 再上 GPTQ/AWQ。

<!-- 后续条目在此追加,遵循上方模板 -->
