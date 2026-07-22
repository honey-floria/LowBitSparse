# CPU 冒烟脚本 → 源码 对照说明

`scripts/cpu_smoke.py` 是一个不下载模型、无需 GPU 的核心逻辑演示。本文把它的
**每一段输出**映射到**背后的源码位置**,方便"看输出 → 跳源码"地理解实现。

> 运行:`python scripts/cpu_smoke.py`(乱码时加 `PYTHONUTF8=1`)
> 行号会随代码变动,若对不上以文件内的函数/类名为准。

---

## 步骤1 — RTN 量化数学

**脚本输出**
```
[步骤1] RTN 量化数学:一个权重矩阵在不同 bit 下的相对误差
  INT8: 相对误差 = 0.5932%
  INT4: 相对误差 = 10.0613%
  INT3: 相对误差 = 21.5904%
```

**脚本位置**:`scripts/cpu_smoke.py` `step1_rtn_math()`(L38-47)

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| `rtn_quantize_weight(w, cfg)` | `quant/rtn.py:60` | 对外入口,读 cfg 调核心函数 |
| 量化-反量化核心 | `quant/rtn.py:10` `_quantize_groupwise` | 分组、padding、round、反量化 |
| 对称分支 | `quant/rtn.py:38` `if symmetric` | zero=0,范围 ±(2^(b-1)-1) |
| 非对称分支 | `quant/rtn.py:45` `else` | 带 zero,范围 [0,2^b-1] |
| 位宽越低误差越大 | 由 `n_bits` 决定 qmax | INT8<INT4<INT3,单调性见 `tests/test_rtn.py` |

---

## 步骤2 — 就地替换 Linear + 压缩统计

**脚本输出**
```
[步骤2] 就地替换 Linear -> FakeQuantLinear + 压缩统计
  替换前 q_proj 类型: Linear
  替换了 4 个 Linear(lm_head 被 skip)
  替换后 q_proj 类型: FakeQuantLinear
  lm_head 是否仍为 Linear: True
  压缩统计: 等效 4.294 bit, 体积 1.316 MB, 量化权重数 655360
```

**脚本位置**:`scripts/cpu_smoke.py` `step2_apply_and_report()`(L50-65)

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 收集待量化 Linear | `quant/apply.py:15` `_iter_linear_names` | 先收集再替换,避免遍历中改结构 |
| 识别 Linear | `quant/apply.py:23` `isinstance(module, nn.Linear)` | 只挑线性层 |
| skip 判断(lm_head) | `quant/apply.py:24` `any(k in name for k in skip)` | 命中关键字则跳过 |
| 就地替换 | `quant/apply.py:33` `apply_quantization` → `:46` `setattr(parent, child, fq)` | 在原 model 上换子模块 |
| 构造量化层 | `quant/fake_linear.py:23` `__init__` | 量化一次 → `:38` 存 buffer |
| "等效 bit / 体积" | `quant/apply.py:50` `compression_report` | 量化权重按 n_bits+scale/zero,其余按真实 dtype |
| 量化层比特统计 | `quant/apply.py:63` `isinstance(module, FakeQuantLinear)` | 含 group 数 × 16bit 开销 |
| 其余参数比特 | `quant/apply.py:80` `for p in model.parameters()` | embedding/norm/bias 按 element_size |

---

## 步骤3 — 伪量化前后前向输出对比

**脚本输出**
```
[步骤3] 伪量化前后前向输出对比
  输出 shape 一致: (2, 1000) == (2, 1000)
  输出相对差异: 9.5427%
```

**脚本位置**:`scripts/cpu_smoke.py` `step3_forward_compare()`(L68-80)

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 量化后 forward | `quant/fake_linear.py:44` `forward` | 走标准 `F.linear`,结构/算子不变 |
| 权重已被污染 | `quant/fake_linear.py:38` buffer `weight` | 存的是反量化权重,故数值变 |
| shape 不变、数值变 | 同上 | 解释"PPL 变、延迟不变":只换数值不换结构 |

---

## 步骤4 — GPTQ vs RTN

**脚本输出**
```
[步骤4] GPTQ(Hessian 校准 + 误差补偿) vs RTN
  INT3 输出误差(越低越好): RTN=698.186  GPTQ=515.351
  GPTQ 相对 RTN 降低: 26.19%
```

**脚本位置**:`scripts/cpu_smoke.py` `step4_gptq_vs_rtn()`

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 构造相关激活 → Hessian | `cpu_smoke.py` `_corr_input` + `H=XᵀX` | 相关输入使 H 非对角,GPTQ 才有发挥空间 |
| GPTQ 量化 | `quant/gptq.py` `gptq_quantize_weight` | 阻尼+Cholesky 逆+逐列误差补偿 |
| Hessian 求逆 | `quant/gptq.py` `_cholesky_inverse_upper` | 对角阻尼防病态,失败则加大阻尼重试 |
| 逐列量化+补偿 | `gptq.py` for 循环 | 每列量化后误差按 Hinv 补偿到右侧未处理列 |
| 定点数学复用 | `quant/primitives.py` `find_qparams`/`quant_dequant` | 与 RTN/AWQ 共享 |

---

## 步骤5 — AWQ vs RTN

**脚本输出**
```
[步骤5] AWQ(激活感知逐通道缩放) vs RTN
  INT3 激活加权误差(越低越好): RTN=6300.3  AWQ=1500.9
  AWQ 相对 RTN 降低: 76.18%
```

**脚本位置**:`scripts/cpu_smoke.py` `step5_awq_vs_rtn()`

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| AWQ 量化 | `quant/awq.py` `awq_quantize_weight` | 按激活幅度网格搜索逐通道缩放 s |
| 缩放-量化-还原 | `awq.py` `Ws=W·s → 量化 → ·(1/s)` | 保护大激活通道,量化粒度更细 |
| ratio 网格搜索 | `awq.py` `for k in range(n_grid+1)` | ratio=0 退化为 RTN,取加权误差最小 |
| 激活统计来源 | `quant/calibration.py` `act_scales` | 每通道 mean(\|x\|),校准一次前向得到 |

---

## 步骤6 — group_size 扫描

**脚本输出**
```
[步骤6] group_size 扫描(RTN INT4 相对误差)
      group=64: 相对误差 = 9.0802%
     group=128: 相对误差 = 10.0148%
     group=256: 相对误差 = 10.9043%
   per-channel: 相对误差 = 11.7130%
```

**脚本位置**:`scripts/cpu_smoke.py` `step6_group_sweep()`

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 分组量化 | `quant/primitives.py` `fake_quant_groupwise` | group_size 控制 scale 局部性 |
| per-channel | `group_size=-1` → 整行一组 | 粒度最粗,误差最大 |
| 单调性锚点 | `tests/test_group_size.py` | 组越小误差越低,per-channel 最差 |

---

## 步骤7 — 端到端校准流水线(GPTQ/AWQ 真实代码路径)

步骤 4-5 只单独调量化数学函数;步骤 7 走**生产链路**:hook 收集逐层统计 →
按 method 路由 → 就地替换 → 压缩统计,是理解"CLI/扫描脚本实际怎么跑"的入口。

**脚本输出**
```
[步骤7] 端到端校准流水线:collect_calib_stats -> apply_quantization
  [gptq] 校准层数=4, 替换=4, 等效4.294bit, 输出相对差异=11.4153%
       down_proj: H(1024, 1024) act(1024,) -> FakeQuantLinear
       释放校准统计 4.8 MB(真实 0.5B 上每层 H 可达 ~95MB,累计数 GB);清空后 stats 层数=0
  [awq] 校准层数=4, 替换=4, 等效4.294bit, 输出相对差异=10.8592%
       down_proj: H(1024, 1024) act(1024,) -> FakeQuantLinear
       释放校准统计 4.8 MB(真实 0.5B 上每层 H 可达 ~95MB,累计数 GB);清空后 stats 层数=0
```

**脚本位置**:`scripts/cpu_smoke.py` `step7_calib_pipeline()`

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 定位待量化层名 | `quant/apply.py` `target_linear_names` | 与 apply 内部同一套 skip 逻辑 |
| hook 收集逐层统计 | `quant/calibration.py` `collect_calib_stats` | 一次前向,增量累积 H / 激活幅度 |
| 单层累积器 | `quant/calibration.py` `_StatHook` | `H += xᵀx`;`act_sum += \|x\|` |
| 按 method 路由 | `quant/apply.py` `_compute_w_dq` | gptq 用 H,awq 用 act_scales,rtn 返回 None |
| 就地替换 | `quant/apply.py` `apply_quantization` | 传入预量化 w_dq 构造 FakeQuantLinear |
| 持有预量化权重 | `quant/fake_linear.py` `__init__(w_dq=...)` | w_dq 非空则跳过内部 RTN |
| 压缩统计 | `quant/apply.py` `compression_report` | 等效 bit / 体积,与方法无关 |
| 释放校准统计 | `quant/calibration.py` `free_calib_stats` | 量化后清空 H/act,估算并回收显存;`main.py`/`run_sweep.py` 评测前均调用 |

> 真实 0.5B 模型 + WikiText-2 校准的完整流程见 `scripts/run_sweep.py`(需 GPU/下载模型),
> 结果用 `scripts/summarize.py` 一键汇总成验收表格。CLI 入口为 `main.py cmd_quant`。
> **显存提示**:GPTQ/AWQ 校准 H 每层 float32 [in,in](0.5B 的 down_proj ~95MB,累计数 GB);
> 量化替换后即由 `free_calib_stats` 释放,故评测阶段显存回落到接近基线(见 OPTIMIZATION M1-d)。

---

## 步骤8 — embedding 量化消融(绑定权重,突破压缩地板)

前 7 步 embedding 始终 FP16(skip),是压缩地板(真实 0.5B 占量化后体积 ~42%)。
步骤 8 演示量化 embedding:绑定(tied)模型下 embed_tokens 与 lm_head 是同一张量,
量化一次、两者共享,压缩比才真正下降(否则拆散绑定反而变大)。

**脚本输出**
```
[步骤8] embedding 量化消融(绑定权重,embed/lm_head 共享)
  基线(emb FP16):  等效 4.312 bit, 体积 1.01 MB
  消融(emb INT8):  等效 7.447 bit, 体积 0.285 MB
  embed 类型: FakeQuantEmbedding, lm_head 类型: FakeQuantLinear
  绑定保持(embed.weight is lm_head.weight): True
  量化后 forward 输出 shape: (2, 16, 1000), 数值有限: True
```
> 注:TinyLM 里 embedding 占绝对多数(vocab 1000×256 且绑定),故量化它体积骤降;
> 等效 bit 升高是因 INT8 embedding 权重数远多于 INT4 linears,拉高了加权均值——
> 这是玩具规模的失真,真实 0.5B 的账见 OPTIMIZATION M1-g。

**脚本位置**:`scripts/cpu_smoke.py` `step8_embedding_quant()`(用 `TiedTinyLM` 绑定模型)

**对应源码**

| 输出/动作 | 源码 | 说明 |
| --- | --- | --- |
| 量化 embedding | `quant/apply.py` `_quantize_embedding` | 沿 embedding_dim 分组 RTN,替换 embed_tokens |
| 绑定检测 | `quant/apply.py` `_quantize_embedding` | `out.weight is emb.weight` → lm_head 复用同一 w_dq |
| 伪量化 Embedding 层 | `quant/fake_embedding.py` `FakeQuantEmbedding` | 缓存反量化权重,forward 走 `F.embedding` |
| 共享 buffer 去重 | `quant/apply.py` `compression_report` | 按 `id(weight)` 去重,绑定矩阵只计一次 |
| embedding 位宽 | `quant/config.py` `embedding_bits` | None 沿用 n_bits;可与 linears 分设 |

> 真实实验(linears GPTQ INT4 + embedding INT8/INT4)由 `run_sweep.py` 的 `EMB_GRID`
> 或 `configs/qwen0.5b_gptq_int4_embint{8,4}.yaml` 触发。

---

## 维护约定(重要)

**每次新增/修改量化或压缩算法,必须同步更新 `scripts/cpu_smoke.py` 与本对照文件。**

- 新算法(如 GPTQ/AWQ、稀疏、蒸馏)落地后,在 `cpu_smoke.py` 增加一个 `stepN_xxx()`
  演示其核心逻辑(用迷你模型/合成数据,保持 CPU 秒级可跑,无需下载)。
- 在本文件新增对应的"步骤N"小节,列出输出 → 源码位置映射。
- 行号变动时更新表格;函数/类改名时同步。
- 目的:让任何人在 CPU 上一条命令就能理解"当前项目实现了哪些算法、各自逻辑在哪"。

