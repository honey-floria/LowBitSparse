"""M3 量化感知蒸馏训练循环。

本文件负责把配置、数据、teacher/student、训练 loop 和结果落盘串起来。
核心目标是：student 在 forward 中承受 INT4 fake-quant 误差，反向通过 STE
更新 FP32 主权重，最终导出成 M1 推理路径可用的 FakeQuant* 模块。
"""
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


def _trainable_report(model) -> dict:
    """统计 student 当前可训练参数量。

    返回:
        trainable_params: requires_grad=True 的参数总数。
        param_tensors:    模型 parameter 总数。
        buffer_tensors:   模型 buffer 总数，包含 scale/LoRA 模式冻结的基础权重。
        logical_tensors:  parameters + buffers，作为消融时的模型张量规模近似。
        trainable_pct:    可训练参数占 logical_tensors 的比例。

    用途:
        M3 消融需要横向比较 full / scale / LoRA 的训练成本；这里把参数量
        写进结果 JSON，避免只看 PPL 而忽略优化器状态和显存差异。
    """
    param_total = sum(p.numel() for p in model.parameters())
    buffer_total = sum(b.numel() for b in model.buffers())
    total = param_total + buffer_total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total else 0.0
    return {
        "trainable_params": int(trainable),
        "param_tensors": int(param_total),
        "buffer_tensors": int(buffer_total),
        "logical_tensors": int(total),
        "trainable_pct": round(pct, 4),
    }


def _forward_outputs(model, input_ids, output_hidden_states: bool = False):
    """兼容 HF / toy 模型的 forward 调用。

    参数:
        model: HuggingFace causal LM 或测试里的 toy LM。调用方只要求它能接受
               `input_ids` 并返回 logits；HF 模型会额外支持 `use_cache` 和
               `output_hidden_states`，toy 模型不一定支持。
        input_ids: token batch，shape=[batch, seqlen]，dtype 通常是 torch.long。
                   这里不传 labels，因为 teacher/student 的 CE/KL 由 `distill_loss`
                   统一计算。
        output_hidden_states: 是否请求 hidden states。只有 `gamma_hidden > 0` 时才应
                              打开；打开后会让 HF 模型保留每层 hidden，显存开销更高。

    返回:
        模型原始 forward 输出。HF 模型通常返回 CausalLMOutput，toy 模型可能返回
        tuple / SimpleNamespace。

    逻辑:
        - 蒸馏训练不需要 KV cache，所以优先传 `use_cache=False`。
        - toy 模型可能不接受 output_hidden_states 或 use_cache，这里逐级回退。
    """
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
    """从不同类型的模型输出中取 logits。"""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError("模型输出缺少 logits")


def _extract_hidden(outputs):
    """从 HF 输出中取最后一层 hidden states；没有则返回 None。"""
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is None:
        return None
    if isinstance(hidden, (tuple, list)) and hidden:
        return hidden[-1]
    return hidden


def distill_loss(student_logits, teacher_logits, labels, temperature: float,
                 alpha_kd: float, beta_ce: float, student_hidden=None,
                 teacher_hidden=None, gamma_hidden: float = 0.0) -> tuple[torch.Tensor, dict]:
    """计算 M3 蒸馏损失。

    参数:
        student_logits: student 前向输出 logits，shape=[batch, seqlen, vocab]。
                        该张量参与反向传播，梯度最终回到 Distill* 的 FP32 主权重。
        teacher_logits: teacher 前向输出 logits，shape 必须与 student_logits 一致。
                        调用方应在 no_grad 下计算它，避免 teacher 被训练。
        labels: 原始 input_ids，shape=[batch, seqlen]。函数内部会右移一位：
                logits[:, :-1] 预测 labels[:, 1:]，符合 causal LM 训练口径。
        temperature: KL 蒸馏温度。>1 会软化 teacher 分布，让非 top token 的暗知识
                     参与训练；过低接近硬标签，过高会让分布过平。
        alpha_kd: KL loss 权重，控制“贴近 teacher 软分布”的强度。通常作为主损失。
        beta_ce: hard-label CE loss 权重，控制“拟合真实下一个 token”的强度。
                 若设为 0，就是纯 teacher 蒸馏；若过大，可能削弱量化误差恢复。
        student_hidden: 可选 student 最后一层 hidden states，shape=[batch, seqlen, hidden]。
                        仅在 `gamma_hidden > 0` 时使用。
        teacher_hidden: 可选 teacher 最后一层 hidden states，shape 需与 student_hidden
                        一致；不同架构 teacher/student 一般不能直接开 hidden 对齐。
        gamma_hidden: hidden-state MSE 权重。0 表示关闭；>0 会额外计算特征对齐，
                      同时增加显存和 forward 输出保存量。

    返回:
        (loss, metrics)，metrics 是可写入 history/json 的 Python float。

    逻辑:
        - CE 约束 student 仍拟合真实下一个 token。
        - KL 约束 student 的软分布贴近 teacher，是精度恢复的主项。
        - hidden MSE 默认关闭，保留作后续消融。
    """
    t = max(float(temperature), 1e-6)
    # causal LM 训练口径：第 t 个 token 预测第 t+1 个 token。
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
    """对任意 teacher/student 跑固定步数蒸馏。

    参数:
        teacher: 冻结的 teacher 模型。通常是 FP16/BF16 HF causal LM，负责提供
                 logits 和可选 hidden states；本函数会强制 `requires_grad=False`。
        student: 已经经过 `prepare_distill_student` 的可训练 fake-quant student。
                 它内部的 DistillLinear/DistillEmbedding 持有 FP32 主权重，其余
                 原模型参数保持冻结。
        train_batches: token batch 序列或可迭代对象。每个元素 shape=[batch, seqlen]，
                       dtype=torch.long；函数会转成 list 并用 cycle 重复消费，所以
                       batch 数可以小于 max_steps。
        cfg: `DistillConfig` 实例，提供优化器、AMP、loss 权重、日志和评测频率等
             训练控制参数。
        evaluator: 可选评测回调，签名为 `evaluator(model) -> dict`。返回值会直接
                   merge 到 history 项中，例如 `{"ppl": 14.2}`；评测期间 student
                   会切到 eval，结束后再切回 train。
        device: 训练设备字符串或 torch.device。None 时从 student 第一个参数推断；
                传入 CPU 时 AMP/GradScaler 自动关闭。

    返回:
        {"history": [...]}，history 中每步记录 loss/kd/ce/feat，评测步额外记录 ppl。

    训练逻辑:
        1. teacher eval + 冻结，只做 no_grad 前向。
        2. student train，只优化 requires_grad=True 的蒸馏参数。
        3. A100 优先用 BF16 autocast；只有 FP16 autocast 才启用 GradScaler。
        4. 固定 step 训练，用 cycle(train_batches) 循环消费小数据集。
    """
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
    # prepare_distill_student 会冻结原始模型参数，只让 Distill* wrapper 可训练。
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
    if amp_enabled and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    # BF16 不需要 loss scaling；FP16 才需要 GradScaler 防止 underflow。
    use_scaler = amp_enabled and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()

    # 双保险：即使外部传入了 FP16 trainable 参数，也转成 FP32 主权重再训练。
    for p in trainable_params:
        if p.dtype != torch.float32:
            p.data = p.data.float()

    history = []

    def record(step: int, train_metrics: dict, grad_norm: float | None = None):
        """记录训练指标，并在需要时追加 evaluator 指标。"""
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
            # teacher 只提供 logits / hidden 监督，不参与反向传播。
            teacher_out = _forward_outputs(teacher, batch, output_hidden_states=cfg.gamma_hidden > 0)
            teacher_logits = _extract_logits(teacher_out)
            teacher_hidden = _extract_hidden(teacher_out) if cfg.gamma_hidden > 0 else None

        with autocast_ctx():
            # student forward 内部会做 STE fake-quant，loss 的梯度回到 FP32 主权重。
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

        if use_scaler:
            # FP16 autocast 路径：先 scale loss，再 unscale 后做 grad clip。
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            else:
                grad_norm = None
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / BF16 路径：不需要 GradScaler，直接反向和裁剪。
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
    """M3 的高层入口:加载 teacher / student / 数据 / 训练并保存结果。

    参数:
        cfg_dict: 从 YAML 加载出的原始配置，通常包含 `teacher`、`student`、`quant`
                  和 `distill` 四段。`distill` 子段会被拍平为 `DistillConfig`
                  字段；若没有显式 `teacher`/`student`，会从旧版 `model` 字段回退。
                  模型段常用键为 `name`、`dtype`、`device`，量化段使用
                  `QuantConfig` 支持的字段。

    返回:
        results dict，会同时通过 save_results 写入 `results/<exp_id>.json`。

    流程:
        1. 解析 DistillConfig，加载 teacher/student/tokenizer。
        2. 准备训练/验证 token 窗口和 PPL evaluator。
        3. 先评 teacher PPL，再把 student 改造成可训练 fake-quant 模型。
        4. 记录 student 初始 PPL，执行蒸馏 loop。
        5. 导出成推理用 FakeQuant*，计算压缩比和最终 PPL，落盘结果。
    """
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
    # 蒸馏前向按完整窗口训练，不需要生成缓存；关掉 use_cache 可省显存并避免警告。
    if hasattr(student, "config") and hasattr(student.config, "use_cache"):
        student.config.use_cache = False
    if hasattr(teacher, "config") and hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False

    train_ids = load_token_ids(tokenizer, cfg.dataset_id, cfg.dataset_config, cfg.train_split)
    eval_ids = load_token_ids(tokenizer, cfg.dataset_id, cfg.dataset_config, cfg.eval_split)
    train_windows = fixed_length_windows(train_ids, cfg.seqlen, cfg.train_samples, shuffle=True, seed=cfg.seed)
    eval_windows = fixed_length_windows(eval_ids, cfg.seqlen, cfg.eval_samples, shuffle=False)
    train_batches = make_batches(train_windows, cfg.batch_size)
    eval_evaluator = build_ppl_evaluator(eval_windows.reshape(-1), cfg.seqlen, stride=cfg.seqlen)

    teacher_base = model_size_report(teacher)
    try:
        # 先评 teacher，避免 student 替换过程污染 teacher 侧状态；失败则重载一次兜底。
        teacher_ppl = eval_evaluator(teacher)
    except RuntimeError as exc:
        log.warning("[M3] teacher eval failed once: %s; retry with fresh teacher", exc)
        teacher, _ = load_model_and_tokenizer(
            model_name=teacher_name,
            dtype=teacher_dtype,
            device=device,
        )
        if hasattr(teacher, "config") and hasattr(teacher.config, "use_cache"):
            teacher.config.use_cache = False
        teacher_ppl = eval_evaluator(teacher)

    qcfg = cfg.quant
    student, replaced = prepare_distill_student(
        student, qcfg,
        train_mode=cfg.train_mode,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
    )
    trainable = _trainable_report(student)
    log.info("[M3] student prepared: replaced=%d, mode=%s, trainable=%d (%.4f%%), quant=%s",
             replaced, cfg.train_mode, trainable["trainable_params"],
             trainable["trainable_pct"], qcfg)

    student_base = model_size_report(student)
    student_init_ppl = eval_evaluator(student)

    train_result = run_distillation_loop(
        teacher=teacher,
        student=student,
        train_batches=train_batches,
        cfg=cfg,
        evaluator=eval_evaluator,
        device=device,
    )

    # 训练结束后导出到和 M1 一致的推理 fake-quant 模块，再统计压缩比/PPL。
    exported = export_distill_student(student, qcfg)
    compression = compression_report(exported)
    student_final_ppl = eval_evaluator(exported)

    if cfg.save_student_path:
        # checkpoint 保存导出后的 state_dict，便于后续恢复推理形态 student。
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
        "trainable": trainable,
        "student_final": {"ppl": student_final_ppl},
        "compression": compression,
        **train_result,
    }
    path = save_results(results, cfg.out_dir, cfg.exp_id)
    log.info("[M3] 结果已保存: %s", path)
    return results
