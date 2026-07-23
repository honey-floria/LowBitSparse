"""M3 量化感知蒸馏训练循环。"""
from __future__ import annotations

from contextlib import nullcontext
from itertools import cycle
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from lowbitsparse.quant import compression_report
from lowbitsparse.quant.config import QuantConfig
from lowbitsparse.utils import get_logger, save_results, set_seed

from .config import DistillConfig
from .data import (
    build_ppl_evaluator,
    fixed_length_windows,
    load_token_ids,
    make_batches,
    strided_ppl_from_ids,
)
from .modules import export_distill_student, prepare_distill_student


log = get_logger()


def _forward_outputs(model, input_ids, output_hidden_states: bool = False):
    """兼容 HF / toy 模型的 forward 调用。"""
    kwargs = {"use_cache": False}
    if output_hidden_states:
        kwargs["output_hidden_states"] = True
    try:
        return model(input_ids, **kwargs)
    except TypeError:
        kwargs.pop("output_hidden_states", None)
        try:
            return model(input_ids, **kwargs)
        except TypeError:
            return model(input_ids)


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError("模型输出缺少 logits")


def _extract_hidden(outputs):
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is None:
        return None
    if isinstance(hidden, (tuple, list)) and hidden:
        return hidden[-1]
    return hidden


def distill_loss(student_logits, teacher_logits, labels, temperature: float,
                 alpha_kd: float, beta_ce: float, student_hidden=None,
                 teacher_hidden=None, gamma_hidden: float = 0.0) -> tuple[torch.Tensor, dict]:
    """组合 KL + CE + 可选 hidden 对齐。"""
    t = max(float(temperature), 1e-6)
    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_student.reshape(-1, shift_student.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    kd = F.kl_div(
        F.log_softmax(shift_student / t, dim=-1),
        F.softmax(shift_teacher / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)

    feat = student_logits.new_tensor(0.0)
    if gamma_hidden > 0 and student_hidden is not None and teacher_hidden is not None:
        feat = F.mse_loss(student_hidden.float(), teacher_hidden.float())

    total = alpha_kd * kd + beta_ce * ce + gamma_hidden * feat
    metrics = {
        "loss": float(total.detach().cpu()),
        "kd": float(kd.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "feat": float(feat.detach().cpu()),
    }
    return total, metrics


def run_distillation_loop(teacher, student, train_batches: Iterable[torch.Tensor], cfg: DistillConfig,
                          evaluator: Callable[[torch.nn.Module], dict] | None = None,
                          device: str | None = None) -> dict:
    """对任意 teacher/student 跑蒸馏。"""
    if device is None:
        device = next(student.parameters()).device
    if not isinstance(train_batches, list):
        train_batches = list(train_batches)
    if not train_batches:
        raise ValueError("train_batches 为空")

    teacher.eval()
    student.train()
    for p in teacher.parameters():
        p.requires_grad_(False)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("student 没有可训练参数")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    amp_enabled = bool(cfg.use_amp and str(device).startswith("cuda"))
    amp_dtype = torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()

    history = []

    def record(step: int, train_metrics: dict, grad_norm: float | None = None):
        item = {"step": step, **train_metrics}
        if grad_norm is not None:
            item["grad_norm"] = float(grad_norm)
        if evaluator is not None:
            student.eval()
            with torch.no_grad():
                item.update(evaluator(student))
            student.train()
        history.append(item)
        return item

    # step 0 先记录基线。
    record(0, {"loss": None, "kd": None, "ce": None, "feat": None})

    batch_iter = cycle(train_batches)
    for step in range(1, cfg.max_steps + 1):
        batch = next(batch_iter).to(device)
        labels = batch.clone()

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_out = _forward_outputs(teacher, batch, output_hidden_states=cfg.gamma_hidden > 0)
            teacher_logits = _extract_logits(teacher_out)
            teacher_hidden = _extract_hidden(teacher_out) if cfg.gamma_hidden > 0 else None

        with autocast_ctx():
            student_out = _forward_outputs(student, batch, output_hidden_states=cfg.gamma_hidden > 0)
            student_logits = _extract_logits(student_out)
            student_hidden = _extract_hidden(student_out) if cfg.gamma_hidden > 0 else None
            loss, metrics = distill_loss(
                student_logits, teacher_logits, labels,
                temperature=cfg.temperature,
                alpha_kd=cfg.alpha_kd,
                beta_ce=cfg.beta_ce,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                gamma_hidden=cfg.gamma_hidden,
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            else:
                grad_norm = None
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            else:
                grad_norm = None
            optimizer.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.max_steps:
            msg = f"[M3] step={step} loss={metrics['loss']:.4f} kd={metrics['kd']:.4f} ce={metrics['ce']:.4f}"
            if cfg.gamma_hidden > 0:
                msg += f" feat={metrics['feat']:.4f}"
            log.info(msg)

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            record(step, metrics, grad_norm=float(grad_norm) if grad_norm is not None else None)
        else:
            item = {"step": step, **metrics}
            if grad_norm is not None:
                item["grad_norm"] = float(grad_norm)
            history.append(item)

    return {"history": history}


def run_distillation_from_config(cfg_dict: dict) -> dict:
    """M3 的高层入口:加载 teacher / student / 数据 / 训练并保存结果。"""
    cfg = DistillConfig.from_dict(cfg_dict)
    set_seed(cfg.seed)
    from lowbitsparse.models import load_model_and_tokenizer, model_size_report

    teacher_cfg = cfg.teacher or cfg_dict.get("model", {})
    student_cfg = cfg.student or teacher_cfg
    teacher_name = teacher_cfg.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
    student_name = student_cfg.get("name", teacher_name)
    teacher_dtype = teacher_cfg.get("dtype", "float16")
    student_dtype = student_cfg.get("dtype", teacher_dtype)
    device = teacher_cfg.get("device", student_cfg.get("device", "cuda"))

    teacher, tokenizer = load_model_and_tokenizer(
        model_name=teacher_name,
        dtype=teacher_dtype,
        device=device,
    )
    student, _ = load_model_and_tokenizer(
        model_name=student_name,
        dtype=student_dtype,
        device=device,
    )
    device = next(student.parameters()).device

    if cfg.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    if hasattr(student, "config") and hasattr(student.config, "use_cache"):
        student.config.use_cache = False
    if hasattr(teacher, "config") and hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False

    qcfg = cfg.quant
    student, replaced = prepare_distill_student(student, qcfg)
    log.info("[M3] student prepared: replaced=%d, quant=%s", replaced, qcfg)

    train_ids = load_token_ids(tokenizer, cfg.dataset_id, cfg.dataset_config, cfg.train_split)
    eval_ids = load_token_ids(tokenizer, cfg.dataset_id, cfg.dataset_config, cfg.eval_split)
    train_windows = fixed_length_windows(train_ids, cfg.seqlen, cfg.train_samples, shuffle=True, seed=cfg.seed)
    eval_windows = fixed_length_windows(eval_ids, cfg.seqlen, cfg.eval_samples, shuffle=False)
    train_batches = make_batches(train_windows, cfg.batch_size)
    eval_evaluator = build_ppl_evaluator(eval_windows.reshape(-1), cfg.seqlen, stride=cfg.seqlen)

    teacher_base = model_size_report(teacher)
    student_base = model_size_report(student)
    teacher_ppl = eval_evaluator(teacher)
    student_init_ppl = eval_evaluator(student)

    train_result = run_distillation_loop(
        teacher=teacher,
        student=student,
        train_batches=train_batches,
        cfg=cfg,
        evaluator=eval_evaluator,
        device=device,
    )

    exported = export_distill_student(student, qcfg)
    compression = compression_report(exported)
    student_final_ppl = eval_evaluator(exported)

    if cfg.save_student_path:
        torch.save(
            {
                "config": cfg.to_dict(),
                "student_state_dict": exported.state_dict(),
            },
            cfg.save_student_path,
        )

    results = {
        "config": cfg.to_dict(),
        "teacher": {"size": teacher_base, "ppl": teacher_ppl},
        "student_init": {"size": student_base, "ppl": student_init_ppl},
        "student_final": {"ppl": student_final_ppl},
        "compression": compression,
        **train_result,
    }
    path = save_results(results, cfg.out_dir, cfg.exp_id)
    log.info("[M3] 结果已保存: %s", path)
    return results
