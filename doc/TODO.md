# LowBitSparse 项目 TODO / 规划 / 设计 / 测试文档

> 小 Transformer(Qwen2.5-0.5B / 1.5B)通用压缩工具箱
> 低 bit 权重量化 + 稀疏注意力 + 量化感知蒸馏
> 运行平台:Google Colab A100(40GB)

---

## 0. 文档说明

- 本文件是项目的**唯一事实来源(single source of truth)**,涵盖规划、技术设计、任务清单、测试方案。
- 每步的实验记录、踩坑、优化与复盘写在 `doc/OPTIMIZATION.md`,不要写在本文件。
- 任务状态标记:`[ ]` 未开始 / `[~]` 进行中 / `[x]` 完成 / `[!]` 阻塞。
- 每完成一个里程碑,回到本文件更新状态,并在 OPTIMIZATION.md 追加复盘。

---

## 1. 项目目标与产出

### 1.1 一句话目标
构建一个可复现的压缩工具箱,验证「量化 + 稀疏 + 蒸馏」三条路线在小模型上的收益,并给出**压缩比、加速比、精度恢复曲线**三类量化结论。

### 1.2 核心产出(Deliverables)
| 类别 | 指标 | 目标(0.5B 为主,1.5B 验证) |
| --- | --- | --- |
| 压缩比 | 模型体积 / 平均比特数 | INT4 达到 ~3.5-4x 体积压缩 |
| 精度 | WikiText-2 PPL、下游任务 | INT4 PPL 退化 < 1.0(蒸馏后) |
| 加速比 | 长序列 prefill / decode 延迟 | 稀疏注意力 8k+ 序列 ≥ 1.5x |
| 恢复曲线 | 蒸馏 step vs PPL | 恢复 RTN-INT4 损失的 ≥ 60% |

### 1.3 里程碑(来自 README)
- **M1** 权重量化(INT8 → INT4)+ 精度评测
- **M2** 稀疏注意力(长序列加速)
- **M3** 量化感知蒸馏恢复精度
- **M4** 消融 + 报告

---

## 2. 技术设计

### 2.1 整体架构
```
LowBitSparse/
├── main.py                  # CLI 入口:量化 / 评测 / 稀疏 / 蒸馏 子命令
├── configs/                 # YAML 实验配置(模型、bit、group_size、稀疏模式…)
│   ├── qwen0.5b_int4.yaml
│   └── qwen1.5b_int4.yaml
├── lowbitsparse/
│   ├── models/              # 模型加载、包装、hook 注入
│   ├── quant/               # RTN / GPTQ / AWQ 量化器,伪量化 Linear
│   ├── sparse/              # 滑窗 / StreamingLLM / 块稀疏 注意力
│   ├── distill/             # QAT 蒸馏训练循环(KL + 特征对齐)
│   ├── eval/                # PPL、lm-eval、延迟/显存 profiler
│   └── utils/               # 校准数据、日志、checkpoint、Drive 挂载
├── scripts/                 # 一键脚本 & Colab notebook
├── results/                 # 指标 json / 曲线图 / 报告
└── doc/                     # TODO.md, OPTIMIZATION.md
```

### 2.2 依赖与环境(Colab A100)
- Python 3.10+,`torch>=2.3`(cu121),`transformers`,`datasets`,`accelerate`
- 量化:`auto-gptq` / `autoawq`(参考实现)+ 自研伪量化(教学/可控)
- 评测:`lm-eval`(harness),`evaluate`
- A100 40GB:0.5B/1.5B 全程 FP16 训练可放下;蒸馏用梯度检查点保险
- 数据落 Google Drive,避免 Colab 断连丢失(checkpoint + results)

### 2.3 量化设计(M1)
- **RTN(round-to-nearest)**:baseline,per-channel / per-group 对称量化,group_size=128。
- **GPTQ**:用校准集算 Hessian,逐列量化并补偿误差,精度更高、成本更高。
- **AWQ**:按激活幅度搜索每通道缩放,保护重要权重通道。
- 统一抽象:`Quantizer.quantize(weight, act_stats) -> (q_weight, scale, zero)`;推理走**伪量化(fake-quant)**先保证精度可测,再评估真实 INT4 kernel 收益。
- 校准数据:WikiText-2 / C4 各取 128 条、seqlen 2048。

### 2.4 稀疏注意力设计(M2)
- **Sliding Window**:局部窗口 w(如 1024),O(n·w) 复杂度。
- **StreamingLLM**:attention sink(前 k 个 token)+ 滑窗,支持超长流式。
- **Block-sparse**:块级 mask,便于 kernel 加速。
- 以 FlashAttention / SDPA 的 mask 或变体实现;先测**质量(PPL/长文任务)**,再测**延迟/显存**。

### 2.5 量化感知蒸馏设计(M3)
- Teacher:FP16 原模型;Student:量化后模型(伪量化,保留可训练 scale)。
- 损失:`L = α·KL(logits) + β·CE(hard) + γ·特征/注意力对齐(可选)`。
- 训练策略:LoRA / 仅训 scale / 全参(A100 上 0.5B 可全参),小步数(数百-数千 step)。
- 目标:恢复 RTN-INT4 相对 FP16 的精度损失。

---

## 3. 任务清单

### M0 — 项目脚手架 & 基线
- [x] 初始化目录结构与 `lowbitsparse` 包、`requirements.txt`
- [x] `main.py` 改造为 CLI(argparse):`quant|eval|sparse|distill`
- [x] 模型加载封装:Qwen2.5-0.5B-Instruct(HF),设备/精度管理
- [x] 评测器:WikiText-2 PPL + 延迟/显存 profiler(基线数据)
- [x] Colab notebook:挂载 Drive、装依赖、跑通 FP16 基线
- [x] **验收**:FP16 基线已实跑并入库(PPL 14.24 / 942.3MB / 30.4ms·37.2tok/s / 峰值4.57GB,见 OPTIMIZATION.md 速查表)

### M1 — 权重量化
- [x] 伪量化 Linear 层(per-group 对称/非对称,量化元数据记录)
- [x] RTN 量化器 + INT8 / INT4 / INT3 支持(非整除 group 用 padding)
- [x] 模型替换 + 理论压缩比/等效 bit 统计(apply.py)
- [x] 单元测试(round-trip 误差、位宽单调性、padding、对称路径)
- [ ] GPTQ 量化器(Hessian 校准 + 误差补偿)
- [ ] AWQ 量化器(激活感知缩放搜索)
- [ ] group_size 扫描(64/128/256)、per-channel vs per-group
- [~] **验收**:三方法 × 多 bit 的 PPL 与压缩比表格 + 曲线(RTN INT8/INT4 已实测;待补 GPTQ/AWQ 与 group_size 扫描)

### M2 — 稀疏注意力
- [ ] 注意力 hook / 替换机制(不改原权重)
- [ ] Sliding Window 实现 + 窗口大小扫描
- [ ] StreamingLLM(sink + 窗口)实现
- [ ] Block-sparse 实现(可选)
- [ ] 长序列基准:2k/4k/8k/16k 的 PPL 与延迟/显存
- [ ] **验收**:加速比曲线 + 长文质量保持表

### M3 — 量化感知蒸馏
- [ ] 蒸馏数据管道(教师 logits 缓存或在线前向)
- [ ] KL + CE 损失 + 可选特征对齐
- [ ] 训练循环:AMP、梯度检查点、checkpoint 到 Drive
- [ ] 消融:全参 vs 仅 scale vs LoRA;α/β 权重
- [ ] **验收**:蒸馏 step vs PPL 恢复曲线

### M4 — 消融 & 报告
- [ ] 汇总所有实验到统一表格(results/summary.json)
- [ ] 组合实验:量化+稀疏、量化+稀疏+蒸馏
- [ ] 1.5B 模型上复现关键结论
- [ ] **验收**:`results/report.md` 三类曲线 + 结论

---

## 4. 测试与评测方案

### 4.1 精度测试
- **困惑度**:WikiText-2(主)、C4(辅),固定 seqlen 2048、stride 评测。
- **下游任务**:lm-eval 选 3-4 个轻量任务(如 arc_easy、hellaswag、piqa)。
- **生成质量**:固定 prompt 集合,人工/规则 spot-check。

### 4.2 效率测试
- **压缩比**:磁盘体积(.safetensors)+ 理论平均比特数。
- **延迟**:prefill(不同 seqlen)与 decode(tokens/s),warmup + 多次取中位。
- **显存**:峰值显存(`torch.cuda.max_memory_allocated`),含 KV cache。

### 4.3 正确性/回归测试
- [ ] 单元测试:量化-反量化数值误差在容差内(pytest)。
- [ ] 伪量化前向与参考实现的 logits 对齐(相对误差阈值)。
- [ ] 稀疏 mask 形状/因果性断言。
- [ ] 每次实验固定随机种子,结果 json 可复现。

### 4.4 实验记录规范
- 每个实验一个 `results/<exp_id>.json`:配置 + 指标 + 环境(GPU、commit)。
- 曲线统一用脚本从 json 生成,避免手工作图。

---

## 5. 风险与对策
| 风险 | 影响 | 对策 |
| --- | --- | --- |
| Colab 断连丢失进度 | 高 | checkpoint + results 落 Drive,支持断点续跑 |
| INT4 无高效 kernel | 中 | 先伪量化测精度,加速比用理论+参考库佐证 |
| 蒸馏不收敛/过拟合校准集 | 中 | 小 lr、早停、hold-out 集验证 |
| 1.5B 显存/时长超预算 | 中 | 梯度检查点、缩小 batch、优先 0.5B 出结论 |
| 依赖版本冲突 | 低 | requirements 固定版本,notebook 首格校验 |

---

## 6. 执行顺序建议
M0 → M1(先 RTN 打通闭环,再 GPTQ/AWQ)→ M2 → M3 → M4。
每个里程碑结束:更新本文件状态 + 在 `OPTIMIZATION.md` 写复盘。
