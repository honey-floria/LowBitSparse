"""量化感知蒸馏模块(M3)。

对外只暴露三类能力：
1. 配置解析: DistillConfig
2. 数据 / 模型准备: load_token_ids、prepare_distill_student、export_distill_student
3. 训练入口: distill_loss、run_distillation_loop、run_distillation_from_config

这样 `main.py` 和测试可以直接 `from lowbitsparse.distill import ...`，
不用知道内部文件怎么拆。
"""
from .config import DistillConfig
from .data import (
    batch_cycle,
    build_ppl_evaluator,
    fixed_length_windows,
    load_token_ids,
    make_batches,
    strided_ppl_from_ids,
)
from .modules import (
    DistillEmbedding,
    DistillLinear,
    export_distill_student,
    prepare_distill_student,
)
from .train import (
    distill_loss,
    run_distillation_from_config,
    run_distillation_loop,
)

__all__ = [
    "DistillConfig",
    "batch_cycle",
    "build_ppl_evaluator",
    "fixed_length_windows",
    "load_token_ids",
    "make_batches",
    "strided_ppl_from_ids",
    "DistillEmbedding",
    "DistillLinear",
    "export_distill_student",
    "prepare_distill_student",
    "distill_loss",
    "run_distillation_from_config",
    "run_distillation_loop",
]
