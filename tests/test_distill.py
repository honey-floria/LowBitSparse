"""M3 蒸馏的 CPU 级单元测试。"""
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lowbitsparse.distill import (
    DistillConfig,
    DistillLinear,
    distill_loss,
    export_distill_student,
    prepare_distill_student,
    run_distillation_loop,
)
from lowbitsparse.quant import QuantConfig


class TinyDistillLM(nn.Module):
    """一个最小语言模型,用于验证蒸馏循环。"""

    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, input_ids, labels=None, output_hidden_states=False, use_cache=False):
        x = self.embed_tokens(input_ids)
        h = torch.tanh(self.proj(x))
        logits = self.lm_head(h)
        out = SimpleNamespace(logits=logits)
        if output_hidden_states:
            out.hidden_states = (x, h)
        return out


def test_prepare_and_export_distill_student():
    torch.manual_seed(0)
    model = TinyDistillLM()
    qcfg = QuantConfig(n_bits=4, group_size=8, symmetric=False, method="rtn", skip=("lm_head",))

    model, n = prepare_distill_student(model, qcfg)
    assert n >= 1
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert all(p.dtype == torch.float32 for p in trainable)
    assert isinstance(model.proj, DistillLinear)
    out = model(torch.randint(0, 32, (2, 12)))
    assert out.logits.shape == (2, 12, 32)
    assert torch.isfinite(out.logits).all()

    exported = export_distill_student(model, qcfg)
    out2 = exported(torch.randint(0, 32, (2, 12)))
    assert out2.logits.shape == (2, 12, 32)
    assert torch.isfinite(out2.logits).all()


@pytest.mark.parametrize("mode", ["full", "scale", "lora"])
def test_distill_train_modes_prepare_and_export(mode):
    torch.manual_seed(3)
    model = TinyDistillLM()
    qcfg = QuantConfig(n_bits=4, group_size=8, symmetric=False, method="rtn", skip=("lm_head",))

    model, n = prepare_distill_student(
        model, qcfg,
        train_mode=mode,
        lora_rank=4,
        lora_alpha=8.0,
    )
    assert n >= 1
    assert isinstance(model.proj, DistillLinear)

    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    if mode == "full":
        assert "proj.weight" in trainable
    elif mode == "scale":
        assert set(trainable) == {"proj.weight_scale"}
    else:
        assert set(trainable) == {"proj.lora_A", "proj.lora_B"}

    x = torch.randint(0, 32, (2, 12))
    out = model(x)
    assert out.logits.shape == (2, 12, 32)
    assert torch.isfinite(out.logits).all()

    exported = export_distill_student(model, qcfg)
    out2 = exported(x)
    assert out2.logits.shape == (2, 12, 32)
    assert torch.isfinite(out2.logits).all()


def test_distillation_loop_reduces_eval_loss():
    torch.manual_seed(1)
    teacher = TinyDistillLM()
    student = copy.deepcopy(teacher)
    qcfg = QuantConfig(n_bits=3, group_size=8, symmetric=False, method="rtn", skip=("lm_head",))
    student, _ = prepare_distill_student(student, qcfg)

    eval_batch = torch.randint(0, 32, (2, 12))
    train_batches = [eval_batch.clone() for _ in range(4)]

    def evaluator(model):
        with torch.no_grad():
            s_out = model(eval_batch)
            _, metrics = distill_loss(
                s_out.logits, s_out.logits.detach(), eval_batch,
                temperature=1.0, alpha_kd=0.0, beta_ce=1.0)
            return {"eval_loss": metrics["loss"]}

    cfg = DistillConfig(
        max_steps=20,
        eval_every=4,
        log_every=100,
        use_amp=False,
        lr=5e-2,
        temperature=1.0,
        alpha_kd=0.0,
        beta_ce=1.0,
        gamma_hidden=0.0,
    )
    result = run_distillation_loop(teacher, student, train_batches, cfg, evaluator=evaluator, device="cpu")
    eval_losses = [item["eval_loss"] for item in result["history"] if "eval_loss" in item]
    assert len(eval_losses) >= 2
    assert eval_losses[-1] <= eval_losses[0]
