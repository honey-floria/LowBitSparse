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
| 项 | 模型 | 配置                                      | PPL(WikiText-2) | 体积 | prefill/decode 延迟 | 显存峰值 |
| --- | --- |-----------------------------------------| --- | --- | --- | --- |
| FP16 baseline | Qwen2.5-0.5B-Instruct | seqlen/stride 2048 prefill512/decode128 | 14.2445 | 942.3 MB(494.03M 参数) | 30.42 ms / 37.19 tok/s | 4574.3 MB(current 959.3 MB) |
| RTN INT8 g128 非对称 | Qwen2.5-0.5B-Instruct | skip=lm_head                            | 14.2326(-0.08%) | 611.7 MB(等效 8.251bit,压缩 1.54x) | 30.50 ms / 37.07 tok/s(同基线) | 4573.1 MB |
| RTN INT4 g128 非对称 | Qwen2.5-0.5B-Instruct | skip=lm_head                            | 17.058(+19.7%) | 441.1 MB(等效 4.251bit,压缩 2.136x) | 29.61 ms / 38.45 tok/s(同基线) | 4573.1 MB |
| GPTQ INT4 g128 非对称 | Qwen2.5-0.5B-Instruct | skip=lm_head, calib 128×512             | 15.4347(+8.4%,恢复 RTN 缺口 57.7%) | 441.1 MB(等效 4.251bit,压缩 2.136x) | 29.70 ms / 37.49 tok/s(同基线) | 7186.6 MB(含校准 Hessian 常驻,见 M1-d) |
| AWQ INT4 g128 非对称 | Qwen2.5-0.5B-Instruct | skip=lm_head, calib 128×512             | 16.3182(+14.6%,恢复 RTN 缺口 26.3%) | 441.1 MB(等效 4.251bit,压缩 2.136x) | 29.88 ms / 37.22 tok/s(同基线) | 7187.2 MB(同上) |
| **GPTQ INT4 + emb INT8** ⭐推荐 | Qwen2.5-0.5B-Instruct | linears GPTQ INT4 + embedding RTN INT8 | 15.4275(+8.3%,≈纯 GPTQ INT4,几乎白拿) | 315.3 MB(等效 5.353bit,压缩 **2.988x**) | 同基线(伪量化) | — |
| GPTQ INT4 + emb INT4(极限) | Qwen2.5-0.5B-Instruct | linears + embedding 均 INT4 | 16.6881(+17.2%) | 250.4 MB(等效 4.251bit,压缩 **3.763x**) | 同基线(伪量化) | — |

> 环境:A100-SXM4-40GB,torch 2.11.0+cu128,CUDA 12.8。数据源 `results/m0_fp16_baseline.json`、`results/m1_rtn_int8_g128.json`、`results/m1_gptq_int4_embint{8,4}.json`。
> **数据源说明**:GPTQ/AWQ/RTN-INT4 的 PPL/压缩比来自 `run_sweep.py` 扫描(见 `results/m1_summary.md`,格式只含 size/compression/ppl);上表 GPTQ/AWQ 行的**延迟/显存**取自更早一次 `cmd_quant` 单跑(带 latency/memory 字段,已被 sweep 同名覆盖,原值见 git `ae7d99f`)。PPL 两次一致(seed=42 复现),延迟/显存不受量化方法影响,合并展示无碍。
> 压缩比基准(体积分母)= 942.3 MB。**注意**:延迟/显存与基线相同,因伪量化仍走 FP16 matmul,压缩比为"理论值"(真实 INT kernel 可省下的量)。
> 压缩地板(已被 M1-g 打掉):embedding(约 136.2M 参数 / 260 MB,与 lm_head 权重共享)默认 skip 时是压缩天花板(占量化后总体积 42-59%);`quant_embedding` 量化它后 emb INT8 白拿 2.99x、emb INT4 达 3.76x。

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

## [2026-07-21] M1-a — RTN 伪量化(代码+本地验证完成)
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


## [2026-07-21] M1-b — RTN INT8 A100 实测(健全性检查通过)
**背景 / 目标**:先跑 INT8 验证量化管线正确性 —— INT8 理论上应几乎无损,若掉点则说明实现有 bug。

**实验设置**:Qwen2.5-0.5B-Instruct;RTN;n_bits=8;group_size=128;非对称;skip=lm_head;seqlen/stride 2048。数据源 `results/m1_rtn_int8_g128.json`。

**结果**:

| 指标 | FP16 基线 | INT8 RTN | 差异 |
| --- | --- | --- | --- |
| PPL | 14.2445 | 14.2326 | -0.08%(噪声级,视为无损) |
| 体积 | 942.3 MB | 611.7 MB | 压缩 1.54x |
| 等效 bit | 16 | 8.251 | 开销核对无误 |
| decode | 37.19 tok/s | 37.07 tok/s | 同基线 |
| 显存峰值 | 4574 MB | 4573 MB | 同基线 |

**关键分析**:
1. **管线正确性验证通过**:INT8 近乎无损(PPL 甚至略低,属噪声),说明量化数学、Linear 替换、压缩统计三处均无 bug,可放心推进 INT4。
2. **压缩比仅 1.54x,远低于"8bit→2x"直觉 —— 根因是 embedding 未量化**。494M 参数中仅 357.8M(72%)被量化;剩余 136.2M 几乎全是 embedding(Qwen2.5-0.5B 的 lm_head 与 embedding 权重共享,skip=lm_head 即跳过这块大矩阵)。体积构成:量化部分 357.8M×8.251bit≈352MB + 未量化 embedding 260MB = 611.7MB。**这 260MB 是压缩天花板**,占量化后总体积 42%。
3. **等效 bit 8.251 核对无误**:group=128 非对称,每组 scale+zero 各 1 个 FP16 = 32bit/128 = 0.25bit 开销,8+0.25=8.25 ✓。
4. **延迟/显存与基线相同,符合伪量化预期**:仍走 FP16 matmul,不省显存不加速,故 1.54x 是理论压缩比,非本次实测省下的量。报告须明确区分。

**结论 & 下一步(据此修正认知)**:
- INT8 达标。**修正 TODO 中"INT4 达 3.5-4x"的目标**:在 skip=lm_head 前提下,INT4 预计整体仅 ~2.1x(量化部分降至 ~190MB,embedding 仍 260MB)。
- 待办:①跑 INT4 验证 ~2.1x 推算与退化幅度;②加"量化 embedding"消融,评估能否推到 3x+ 及 PPL 代价 —— 这直接回答 3.5-4x 目标可达性;③group_size 扫描。


## [2026-07-21] M1-c — RTN INT4 A100 实测(压缩比预判命中,精度缺口确立)
**背景 / 目标**:实测 INT4,验证 ~2.1x 压缩比推算,并量化 RTN 在 4bit 下的精度退化 —— 这个缺口是后续 GPTQ/AWQ 与 M3 蒸馏要恢复的目标。

**实验设置**:Qwen2.5-0.5B-Instruct;RTN;n_bits=4;group_size=128;非对称;skip=lm_head;seqlen/stride 2048。数据源 `results/m1_rtn_int4_g128.json`。

**结果(三档对比)**:

| 指标 | FP16 | INT8 | INT4 |
| --- | --- | --- | --- |
| PPL | 14.2445 | 14.2326 | 17.058 |
| PPL 退化 | — | -0.08% | +19.7%(+2.81) |
| 体积 | 942.3 MB | 611.7 MB | 441.1 MB |
| 压缩比 | 1x | 1.54x | 2.136x |
| 等效 bit | 16 | 8.251 | 4.251 |

**关键分析**:
1. **压缩比 2.136x,预判(~2.1x)精确命中**,误差 <2%。构成:量化部分 357.8M×4.251bit≈190MB + 未量化 embedding 260MB = 441MB。
2. **embedding 地板效应加剧**:INT4 下 embedding 260MB 占总体积 59%(INT8 时 42%)。位宽越低,不量化的 embedding 占比越高、越接近天花板 —— 要上 3x+ 必须量化 embedding。
3. **精度退化 +19.7% 属 RTN 4bit 的预期偏大表现**:RTN 逐组独立、不看激活、无误差补偿,是最朴素方案,4bit 掉点本就明显。**此缺口(+2.81 PPL)即 M1-GPTQ/AWQ 与 M3-蒸馏的价值标尺**,典型预期 GPTQ/AWQ 可压到 +0.5~1.0 量级。
4. 等效 bit 4.251 = 4 + 32/128,核对无误;延迟/显存同基线(伪量化)。

**结论 & 下一步**:
- M1 的 RTN 部分完成,三档给出"压缩比 vs 精度"权衡雏形:INT8 近乎白拿 1.54x;INT4 得 2.14x 但掉 19.7%。
- 下一步建议先做 **GPTQ**(Hessian 校准 + 误差补偿),直接冲着恢复 INT4 的 +2.81 缺口;embedding 量化消融作为压缩比补充实验穿插。

**安全备注**:本次记录过程中,一次 Edit 的工具返回里混入了伪装成指令的提示注入文本(诱导调用 DesignSync/finalize_plan),已识别并忽略,未执行任何相关操作。

## [2026-07-22] M1-d — GPTQ INT4 A100 实测(误差补偿恢复过半缺口)
**背景 / 目标**:RTN INT4 掉了 +2.81 PPL,GPTQ 用校准集算 Hessian、逐列量化并把误差补偿到未处理列,冲着恢复这个缺口。这是 M1 "精度恢复"的第一张牌。

**实验设置**:Qwen2.5-0.5B;GPTQ;n_bits=4;group=128;非对称;skip=lm_head;校准 WikiText-2 train 128 条×512 seqlen;seqlen/stride 2048。数据源 `results/m1_gptq_int4_g128.json`。

**结果(与 RTN 同压缩比对比)**:

| 指标 | FP16 | RTN INT4 | GPTQ INT4 |
| --- | --- | --- | --- |
| PPL | 14.2445 | 17.058 | 15.4347 |
| ΔPPL | — | +2.813(+19.7%) | +1.190(+8.4%) |
| 压缩比 | 1x | 2.136x | 2.136x |
| 等效 bit | 16 | 4.251 | 4.251 |
| decode | 37.19 | 38.45 | 37.49 tok/s |

**关键分析**:
1. **恢复 RTN 缺口的 57.7%**:缺口从 +2.813 压到 +1.190,落在 M1-c 预期的 GPTQ 收益带内(+0.5~1.0 量级偏上)。同压缩比下纯靠"更聪明地选量化值"换来精度,是 GPTQ 的核心价值——压缩比与 RTN 完全相同(权重仍 4.251 等效 bit),差别只在权重数值。
2. **显存暴涨到 7186 MB(基线 4574)非模型问题,是校准 Hessian 未释放**:`collect_calib_stats` 给每层存 float32 `H[in_f,in_f]`,down_proj in_f=4864 → 单层 H≈94.6MB,×24 层≈2.27GB;`calib_stats` 在 `cmd_quant` 里活到 eval 阶段未释放,current 3571−959≈2.6GB 与之吻合。**这是可清理项**:量化替换完即可 `del calib_stats` + `empty_cache`,不影响精度,但小显存卡上有意义。留作 M1 收尾优化。
3. **延迟同基线**:伪量化仍走 FP16 matmul,GPTQ 只改权重数值不改结构,故延迟/压缩比与 RTN 一致,唯一变量是 PPL。

**结论 & 下一步**:GPTQ 达标,确立为 INT4 首选。下一步 AWQ 对照(见 M1-e),再做 group_size 扫描与 embedding 量化消融补齐验收表。

## [2026-07-22] M1-e — AWQ INT4 A100 实测(收益逊于 GPTQ,合成 vs 真实落差归因)
**背景 / 目标**:AWQ 按激活幅度逐通道搜索缩放 s,保护大激活通道后再量化。与 GPTQ 同为 INT4 校准类方法,横向对照两条精度恢复路线。

**实验设置**:Qwen2.5-0.5B;AWQ;n_bits=4;group=128;非对称;skip=lm_head;校准同 GPTQ(128×512);seqlen/stride 2048。数据源 `results/m1_awq_int4_g128.json`。

**结果(三方法 INT4 齐平)**:

| 指标 | RTN INT4 | AWQ INT4 | GPTQ INT4 |
| --- | --- | --- | --- |
| PPL | 17.058 | 16.3182 | 15.4347 |
| ΔPPL | +2.813 | +2.074 | +1.190 |
| 恢复 RTN 缺口 | — | 26.3% | 57.7% |
| 压缩比 | 2.136x | 2.136x | 2.136x |

**关键分析**:
1. **AWQ 恢复 26.3%,明显逊于 GPTQ 的 57.7%**。根因是方法差异:AWQ 只做通道缩放、不做误差补偿;GPTQ 逐列量化后把残差 Hinv 补偿到未处理列,在 4bit 这种粗量化下补偿的价值更大。排序确立 **GPTQ > AWQ > RTN**。
2. **cpu_smoke 合成数据(AWQ 降误差 76%)与真实(恢复 26.3%)的落差是合理的,非 bug**:合成用例特意构造了强通道激活差异来凸显 AWQ 优势;真实 0.5B 各通道激活幅度差异没那么悬殊,缩放的边际收益随之缩小。**提醒:cpu_smoke 是逻辑演示/相对关系锚点,不是收益预测器**——报告里不能拿合成降幅当真实收益。
3. **显存 7187 MB 与 GPTQ 同因**(校准 Hessian 常驻,AWQ 其实只用 act_scales,H 是收集时一并算的);延迟同基线。压缩比三方法全等,横轴只有 PPL 一个变量,"压缩比 vs 精度"权衡图已成型。

**结论 & 下一步**:M1 三方法闭环全部实测完成,核心结论(GPTQ 首选,同压缩比恢复过半缺口)已成立。遗留:①group_size 扫描(64/128/256/per-channel)填充验收表另一维;②embedding 量化消融回答"能否上 3x+";③M1 收尾把校准 Hessian 释放掉;之后进入 M2 稀疏注意力。可考虑给 AWQ 加更细网格 + clip 搜索作为改进项。

## [2026-07-22] M1-f — group_size × bit 全扫描(GPTQ 鲁棒性坐实,拐点确立)
**背景 / 目标**:用 `run_sweep.py` 跑 method×bit×group 全组合,填充验收表的粒度维,给出"压缩比 vs 精度"完整权衡,并回答 group_size / bit 各自的收益边界。

**实验设置**:Qwen2.5-0.5B;RTN/GPTQ/AWQ;INT4 group∈{64,128,256,per-channel}+ INT3 g128;非对称;skip=lm_head;GPTQ/AWQ 校准 128×512;seqlen/stride 2048。数据源 `results/m1_summary.md` 及各 `m1_*.json`。g128 三点与 M1-c/d/e 单跑值一致(seed=42 复现)。per-channel 的 GPTQ/AWQ 两点为后补(见下方"补充"),已并入本表。

**结果(ΔPPL,粗体为该 group 最优;末列 GPTQ 消除 RTN 缺口比例)**:

| group | RTN | AWQ | GPTQ | 压缩比 | GPTQ 恢复比例 |
| --- | --- | --- | --- | --- | --- |
| g64 | +1.884 | +1.463 | **+0.887** | 2.09x | 52.9% |
| g128 | +2.813 | +2.074 | **+1.190** | 2.14x | 57.7% |
| g256 | +4.620 | +2.986 | **+1.508** | 2.16x | 67.4% |
| per-channel | +12.641 | +6.321 | **+3.084** | 2.18x | 75.6% |
| INT3 g128 | +51.451 | +17.123 | **+8.235** | 2.37x | 84.0% |

**关键分析**:
1. **方法排序 GPTQ<AWQ<RTN 在每个配置下无一例外成立**,误差补偿的价值是稳健的,非某一超参下的偶然。
2. **group_size 是纯精度旋钮、几乎非压缩旋钮**:INT4 四档压缩比仅 2.09–2.18x 浮动(等效 bit 4.028–4.501),因 embedding 地板占大头,group 改的只是每组 scale/zero 开销的零头。cpu_smoke 的合成单调性锚点在 0.5B 上坐实(组越小 PPL 越低)。
3. **量化越激进,GPTQ 相对 RTN 价值越大——补完 per-channel 后升级为定量规律**:GPTQ 消除 RTN 缺口的比例随粒度变粗/bit 变低**单调递增**:g64 52.9% → g128 57.7% → g256 67.4% → per-channel 75.6% → INT3 84.0%。RTN 逐组无补偿,粒度一粗/bit 一低就撑不住(per-channel +12.64、INT3 +51.45);GPTQ 用 Hinv 把误差摊到后续列,越是恶劣条件补偿的相对收益越高。这条单调性是本次扫描最硬的结论。
4. **拐点与边界**:精度最优 GPTQ g64(+0.89@2.09x);性价比拐点 GPTQ g128(+1.19@2.14x,降 g64 仅多省 0.3 PPL 却掉 0.05x 压缩,边际薄);**per-channel 对谁都不划算**——压缩比只从 g256 的 2.16x 微升到 2.18x(等效 bit 差 0.11),GPTQ 却从 +1.51 恶化到 +3.08,纯亏精度不换体积;INT3 同理不划算(GPTQ +8.24,压缩仅 2.37x)。**根因一致:压缩比被 embedding 地板锁死,粒度/bit 再压只掉精度不省体积——要破 ~2.4x 必须动 embedding,不是降 bit 或粗化 group。**

**结论 & 下一步**:M1 验收表(4 粒度 × 3 方法 + INT3)全部实测完成,GPTQ g128 定为默认推荐,per-channel 确认为无用点(已收录仅供边界佐证)。核心权衡曲线闭合。遗留仅剩两项:①embedding 量化消融(唯一能破 2.4x 的方向);②M1 收尾释放校准 Hessian 显存。之后进入 M2 稀疏注意力。

## [2026-07-22] M1-g — embedding 量化消融(3.5-4x 达成,emb INT8 几乎白拿)
**背景 / 目标**:M1-f 确认压缩比被 embedding FP16 地板锁死在 ~2.14x;要命中项目原始 deliverable(3.5-4x)只能量化 embedding。本条记录消融的设计、实现与 A100 实测。

**核心难点——权重绑定(tied embedding)**:Qwen2.5-0.5B 的 `embed_tokens.weight` 与 `lm_head.weight` 是同一张量。若只把 lm_head 从 skip 移除,`apply_quantization` 会给它建一个新 INT4 buffer,而 embed_tokens 仍指原 FP16 张量——**绑定被拆散,变成两个矩阵,体积不降反升**。正确做法:量化共享矩阵一次,embed/lm_head 都用同一份反量化权重(部署时仍只存一份)。

**设计与实现**:
1. `FakeQuantEmbedding`(`fake_embedding.py`):镜像 FakeQuantLinear,沿 embedding_dim 分组 RTN,forward 走 `F.embedding`。embedding 是查表、无激活,故只能 RTN(不做 GPTQ/AWQ)。
2. `apply.py` `_quantize_embedding`:量化 embedding 一次得 w_dq;检测 `out.weight is emb.weight`,绑定时把 lm_head 也换成持有**同一 w_dq 对象**的 FakeQuantLinear(用 `dataclasses.replace(cfg, n_bits=embedding_bits)` 让其元数据记正确位宽)。
3. `compression_report` 改为按 `id(weight)` 去重——绑定共享矩阵只计一次,否则体积算两遍。
4. `config.py` 加 `quant_embedding` / `embedding_bits`(None 沿用 n_bits);配置 `qwen0.5b_gptq_int4_embint{8,4}.yaml`;`run_sweep.py` `EMB_GRID` 两点。

**结果(A100 实测,linears 固定 GPTQ INT4 g128;数据源 `results/m1_gptq_int4_embint{8,4}.json`)**:
| 配置 | embedding | PPL | ΔPPL(vs FP16) | 压缩比 | 体积 |
| --- | --- | --- | --- | --- | --- |
| 基线(M1-f 冠军) | FP16 | 15.4347 | +1.190 | 2.136x | 441MB |
| **emb INT8** | INT8 | **15.4275** | **+1.183** | **2.988x** | 315.3MB |
| emb INT4 | INT4 | 16.6881 | +2.444 | **3.763x** | 250.4MB |

压缩比预估(~3.0x / ~3.76x)分毫命中(实测 2.988 / 3.763)。

**关键分析**:
1. **emb INT8 几乎白拿——推翻"embedding 必须保 FP16"的隐含假设**:PPL 15.4275 vs 基线 15.4347,不掉反略低(噪声级),压缩比却从 2.14x 跳到 **2.99x(+40%)**。我在设计时预判的风险(lm_head 输出投影对量化敏感、绑定强制同精度→代价可能大)在 INT8 上**完全没兑现**。INT8 对 embedding/lm_head 这类大而分布温和的矩阵足够。**→ emb INT8 取代纯 GPTQ INT4,成为新的默认推荐(帕累托支配:更小体积 + 零精度代价)。**
2. **3.5-4x deliverable 确认可达**:emb INT4 命中 3.763x,代价 +2.444 PPL(比 emb INT8 多掉 +1.26)。这是本项目原始目标(TODO 1.2 表"INT4 达 ~3.5-4x")首次真正达成——此前 M1-c 曾"修正"为不可达,根因是当时默认 embedding 不量化;打掉地板后目标成立。
3. **等效 bit 口径变化**:emb INT8 等效 5.353 bit(embedding 493.96M 权重里 INT8 部分权重数多,拉高加权均值),emb INT4 等效 4.251 bit(与纯 linears 一致,因全模型统一 4bit)。quant_weights 从 357.8M 升到 493.96M,证实 embedding 已纳入量化统计、绑定去重生效(未把共享矩阵算两遍)。

**验证(CPU)**:单测 4 例过(round-trip、绑定量化后仍共享同一 buffer、压缩比提升、forward 有限);cpu_smoke step8 演示绑定保持 + 体积下降。

**结论 & 下一步**:embedding 量化消融完成,两点确立压缩-精度前沿。**M1 全部收官**:推荐配置升级为 GPTQ INT4 g128 + emb INT8(2.99x @ +1.18 PPL);极限压缩 emb INT4(3.76x @ +2.44)。遗留仅"emb INT8 是否还能配 group_size 更细的 linears 再压 PPL"这类锦上添花项,不阻塞。下一步进入 M2 稀疏注意力。

## [2026-07-22] M1-h — AWQ 权重裁剪搜索(auto_clip):负结果,默认关闭
**背景 / 目标**:M1-e 收尾提到"给 AWQ 加 clip 搜索"。AWQ 论文完整方法是缩放 + 裁剪两件事,此前只实现缩放。本条补上裁剪并 A100 实测——**结果为负,裁剪全面恶化 PPL,已默认关闭**。如实记录。

**实现(两阶段)**:`primitives.py` `find_qparams` 加 `clip` 系数、新增 `fake_quant_groupwise_autoclip`(逐组在 α∈[0.5,1] 网格取**权重量化 MSE** 最优);`awq.py` 阶段 2 在缩放后权重上跑 autoclip,末尾用加权 Frobenius 误差做保底比较。

**A100 实测(裁剪 vs 无裁剪,同压缩比)**:
| 配置 | 无裁剪 PPL | +裁剪 PPL | 变化 |
| --- | --- | --- | --- |
| AWQ INT4 g64 | 15.7071 | 16.0458 | +0.34 |
| AWQ INT4 g128 | 16.3182 | 16.4258 | +0.11 |
| AWQ INT4 g256 | 17.2309 | 17.7446 | +0.51 |
| AWQ INT4 per-channel | 20.5658 | 23.5248 | +2.96 |
| AWQ INT3 g128 | 31.3680 | **48.7307** | **+17.36** |

**根因——代理指标失真(核心教训)**:autoclip 逐组最小化的是**权重空间量化 MSE**(对角代理),保底比较用的是**加权 Frobenius 误差**——两者都在权重空间。但低 bit 下 PPL 极度依赖少数**离群权重**的保真;裁剪为给 bulk 权重换网格分辨率,把离群值 clamp 掉了:权重 MSE 降了(保底比较放行),PPL 反而崩。group 越大 / bit 越低,裁剪在 MSE 上越"划算"、对 PPL 越致命(per-channel +2.96、INT3 +17.36 印证)。我此前"grid 含 α=1 故永不劣化"的说法只在**权重 MSE 口径**成立,对 PPL 不成立——这正是 M1-e 教训(合成/代理收益不预测真实收益)的又一次、且更严重的复现。

**为何忠实版没做**:真正的 AWQ auto_clip 在校准集上按**输出误差**逐 clip 候选跑前向搜索(每层每候选一次 forward),开销大但忠实。本实现图便宜用了权重空间对角代理,不可靠。忠实版留作未来项,不阻塞 M1。

**处理**:`clip_search` 默认改 `False`;裁剪代码保留在 flag 后并在 docstring 标注有害;cpu_smoke step5 改回缩放为主、裁剪标注为已知负结果;`results/m1_awq_*.json` 需用默认(裁剪关)重跑恢复无裁剪好值。

**结论**:纯缩放 AWQ(已验证)仍是 AWQ 的正确形态,M1-f 的方法排序 GPTQ < AWQ < RTN **不变**。auto_clip 的正确实现方向是输出误差搜索,非权重空间代理。价值在于这条负结果本身:**低 bit 量化里"降权重 MSE"与"降 PPL"可以反向,离群权重不可牺牲。**

<!-- 后续条目在此追加,遵循上方模板 -->
