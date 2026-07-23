"""量化感知蒸馏模块(M3: KL + CE + feature alignment)。"""
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
