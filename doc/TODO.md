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
| 压缩比 | 模型体积 / 平均比特数 | INT4 达到 ~3.5-4x 体积压缩 ✅ 达成(emb INT4 3.76x;emb INT8 2.99x 零精度代价,见 M1-g) |
| 精度 | WikiText-2 PPL、下游任务 | INT4 PPL 退化 < 1.0(蒸馏后) |
| 加速比 | 长序列 prefill / decode 延迟 | decode 已由 M2-e ring+graph 达成 ~5.3x;prefill 仍开放(M2-d) |
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
├── main.py                      # CLI 入口:eval/quant/sparse 已实现、distill 占位;量化+评测闭环
├── configs/                     # YAML 实验配置
│   ├── qwen0.5b_base.yaml       #   0.5B FP16 基线(M0)
│   ├── qwen1.5b_base.yaml       #   1.5B FP16 基线
│   ├── qwen0.5b_int8.yaml       #   RTN INT8(量化健全性检查)
│   ├── qwen0.5b_int4.yaml       #   RTN INT4
│   ├── qwen0.5b_gptq_int4.yaml  #   GPTQ INT4(含校准参数)
│   ├── qwen0.5b_awq_int4.yaml   #   AWQ INT4(含校准参数)
│   ├── qwen0.5b_sparse_sliding.yaml   #   Sliding Window M2
│   ├── qwen0.5b_sparse_streaming.yaml #   StreamingLLM M2
│   ├── qwen0.5b_sparse_block.yaml     #   Block-sparse M2(可选)
│   ├── qwen0.5b_sparse_streaming_kvprune.yaml   #   M2-c KV cache 裁剪
│   ├── qwen0.5b_sparse_streaming_compile.yaml   #   M2-e 前置 compile/graph 探针
│   └── qwen0.5b_sparse_streaming_ringgraph.yaml #   M2-e ring-buffer + CUDA graph
├── lowbitsparse/
│   ├── models/loader.py         # 模型/分词器加载(dtype/设备管理)、体积统计
│   ├── quant/                   # 权重量化(M1 核心)
│   │   ├── config.py            #   QuantConfig 超参 dataclass(可从 YAML 构造)
│   │   ├── primitives.py        #   共享定点数学:求 scale/zero、量化-反量化、分组伪量化
│   │   ├── rtn.py               #   RTN 量化(baseline,只看权重,无需校准)
│   │   ├── gptq.py              #   GPTQ 量化(Hessian 阻尼求逆 + 逐列误差补偿)
│   │   ├── awq.py               #   AWQ 量化(激活感知逐通道缩放网格搜索)
│   │   ├── calibration.py       #   校准数据采样 + hook 收集逐层 Hessian/激活统计
│   │   ├── fake_linear.py       #   FakeQuantLinear:持有反量化权重,forward 走标准 matmul
│   │   └── apply.py             #   遍历替换 Linear、按 method 路由、压缩比/等效 bit 统计
│   ├── sparse/                  # 滑窗 / StreamingLLM / 块稀疏 注意力(M2)
│   ├── distill/                 # QAT 蒸馏训练循环(M3,未落地)
│   ├── eval/                    # 评测
│   │   ├── ppl.py               #   WikiText-2 strided PPL
│   │   └── profiler.py          #   prefill/decode 延迟、显存峰值
│   └── utils/common.py          # 随机种子、日志、YAML 加载、结果落盘、环境采集
├── scripts/                     # 一键脚本 & Colab notebook
│   ├── cpu_smoke.py             #   CPU 秒级冒烟:步骤1-7 演示量化数学+校准流水线(无下载)
│   ├── run_sweep.py             #   method×bit×group_size 全组合扫描,落盘 json
│   ├── summarize.py             #   汇总 results/*.json 为验收表格(含 ΔPPL)
│   └── run.ipynb                #   Colab:挂 Drive、装依赖、跑 M0/M1/M2/M2-c/M2-e
├── tests/                       # pytest:test_rtn / test_gptq / test_awq / test_group_size
├── results/                     # 指标 json / 曲线图 / 报告
└── doc/                         # TODO.md、OPTIMIZATION.md、CPU_SMOKE_MAP.md
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
- [x] GPTQ 量化器(Hessian 校准 + 逐列误差补偿,gptq.py + calibration.py)
- [x] AWQ 量化器(激活感知逐通道缩放网格搜索,awq.py)
- [x] AWQ 权重裁剪搜索(auto_clip)— **负结果,已默认关闭**(见 M1-h)
      A100 实测裁剪全面恶化 PPL(INT3 +17.36)。根因:权重空间 MSE 代理与 PPL 反向,
      裁剪牺牲了低 bit 下不可牺牲的离群权重。纯缩放 AWQ 仍是正确形态。
      ⚠️ `results/m1_awq_*.json` 现为裁剪坏值,需 `python scripts/run_sweep.py --only awq --no-emb` 重跑恢复。
- [x] group_size 扫描(64/128/256/per-channel)+ per-channel vs per-group(run_sweep.py 网格 + 单调性测试)
- [x] **验收**:三方法 × INT4 的 PPL 与压缩比表格(见 `results/m1_summary.md` + OPTIMIZATION 速查表)
      A100 实测结论:同压缩比 2.136x 下 GPTQ(+1.19)> AWQ(+2.07)> RTN(+2.81),GPTQ 恢复 RTN 缺口 57.7%。
      遗留(转入下方,不阻塞 M1 核心结论):group_size 扫描曲线、embedding 量化消融、校准 Hessian 显存释放。
- [x] group_size × bit 扫描实跑(INT4 g64/128/256 + RTN per-channel + INT3 g128,见 M1-f)
      结论:group 是纯精度旋钮(压缩比恒 ~2.1x);GPTQ 对粗粒度/低 bit 鲁棒性远超 RTN;拐点 GPTQ g128。
- [x] 补跑 GPTQ/AWQ per-channel(见 M1-f 补充)
      结论:GPTQ +3.08 / AWQ +6.32 / RTN +12.64;GPTQ 恢复比例随粒度变粗单调升至 75.6%;但 per-channel 纯亏精度不换体积(2.18x vs g256 2.16x),确认为无用点。
- [x] embedding 量化消融:破 2.4x 地板(见 M1-g)
      实测:emb INT8 = 2.99x @ PPL 15.43(≈纯 GPTQ INT4,几乎白拿);emb INT4 = 3.76x @ PPL 16.69。
      结论:emb INT8 帕累托支配纯 GPTQ INT4,升级为默认推荐;3.5-4x deliverable 达成。
- [x] M1 收尾优化:量化后释放 `calib_stats`(`free_calib_stats`,main.py/run_sweep.py 评测前调用;cpu_smoke step7 演示)
      A100 已确认:embint 跑日志显示释放 2607.9MB,peak 从 ~7187MB 回落到 4574MB(=基线),预测兑现。

### M2 — 稀疏注意力
- [x] 注意力 hook / 替换机制(不改原权重)
- [x] Sliding Window 实现 + 窗口大小扫描
- [x] StreamingLLM(sink + 窗口)实现
- [x] Block-sparse 实现(可选)
- [x] 长序列基准:2k/4k/8k/16k 的 PPL 与延迟/显存(A100 已回填 `results/m2_*.json`)
- [x] **验收**:加速比曲线 + 长文质量保持表(已回填;M2-e decode 达标,prefill 仍转 M2-d)

> 备注(2026-07-22 更新):M2-a/b 的 additive mask 路径确实没拿到加速(功能完成+负反馈),但这不是 M2 的终点。M2-c 让 KV 裁剪真实生效(显存转正,decode 仍不涨),**M2-e 最终翻案**:先用探针定量证明 decode 是 overhead-bound(非 KV-bound),再用有界 ring-buffer KV cache + CUDA graph 拿到 **decode ~5.3x + 恒定显存**。核心教训:小模型 decode 的瓶颈是 kernel launch overhead,加速来自 CUDA graph,稀疏(固定小 cache)是让 graph 在长序列下可行的使能条件,而非加速本身的来源。

#### M2 后续优化计划
- [x] **M2-c StreamingLLM KV cache 裁剪**:只保留 attention sink + 最近 window 的 K/V,让 decode 阶段真实缩短 `kv_len`,不再只靠 mask 屏蔽旧 token。
      验收目标:StreamingLLM 质量保持 `ΔPPL < 1.5`,decode speedup > 1.2x,peak memory 不高于 dense baseline。
      本轮结果(3 中 2 达标):ΔPPL ✅(最差 16k +1.35)、peak memory ✅(转为正节省,16k 省 361MB)、decode speedup ❌(仍 0.91x)。
- [x] Cache 兼容层:支持 HF `past_key_values` / `DynamicCache` / tuple cache 的读取、裁剪与回写,保证 `generate()` 和手写 profiler 都可用。
- [x] M2-c 单测与 smoke:覆盖 sink 保留、窗口边界、cache 长度单调裁剪、restore 行为、无 cache 的 prefill 退化路径。
- [x] M2-c A100 回归:2k/4k/8k/16k decode-only benchmark,输出 `results/m2c_streaming_kvprune_*.json` 与汇总表。
      结果回填(2026-07-22 重跑,新版 HF cache 容器兼容层已落地 commit ccc8052/b28fdef):裁剪**已真实生效**——`applied: true`、`applied_steps=903`、覆盖全 24 层、cache 稳定裁到 `kept_len=1088`(sink 64 + window 1024)。peak memory 由旧版负反馈翻为正节省(2k +24MB → 16k +361MB,随序列增长),质量 ΔPPL 守在 1.5 内。**但 decode speedup 仍 0.907–0.915x(未达 1.2x)**:根因是 0.5B decode 为权重带宽瓶颈而非 KV 瓶颈,把 KV 从 16384 砍到 1088 对每步延迟几乎无影响,而 903×24 层的 Python 侧裁剪记账开销盖过了注意力上省下的时间(稳态每步仅 `pruned:1`)。结构性结论,非 bug——decode 加速已转 M2-e ring+graph 路径验证。
- [x] **M2-e 有界 ring-buffer KV cache + CUDA graph decode**(实际落地形态,替代原"kernel-aware hook"设想):
      两个 A100 探针先定量锁定根因——decode 是 **overhead-bound**(单次 forward ~26.5ms 固定地板,512 token 仅比 1 token 慢 9.3%),CUDA graph 可消除。再实现 `RingKVCache`(固定 sink+window=1088 回绕写入,decode 返回恒定形状 buffer)+ graph 捕获。
      **结果(decode 目标 ≥1.2x 大幅超标):2k/4k/8k/16k decode speedup 全部 ~5.3x(36→193 tok/s),显存恒定在 1088(省 18→327MB 随长度增长),ΔPPL 与 M2-c 一致守 1.5 内**。数据源 `results/m2e_streaming_ringgraph_s64_w1024.json`。关键洞察:decode 加速来自 CUDA graph 消 overhead,稀疏的作用是让固定形状 cache 在任意长序列下可行。范围为 Benchmark 证明(latency/memory 真实路径,quality 用 additive mask 参考,不做 RoPE 相位忠实修正)。
      代码:`lowbitsparse/sparse/ring_cache.py`、`scripts/cudagraph_probe.py --ring`、`configs/qwen0.5b_sparse_streaming_ringgraph.yaml`。
- [ ] **M2-d chunked prefill / local attention**(仍开放):prefill 阶段避免构造完整 `[batch,1,q,kv]` additive mask,按 query chunk 只看局部 K/V。验收目标:8k/16k peak memory 低于 dense baseline,prefill 不慢于 dense。(M2-e 已解决 decode 侧,prefill 加速仍待此项。)

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
