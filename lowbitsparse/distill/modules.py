"""M3 的可训练 fake-quant 模块和导出工具。"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from lowbitsparse.quant.config import QuantConfig
from lowbitsparse.quant.fake_embedding import FakeQuantEmbedding
from lowbitsparse.quant.fake_linear import FakeQuantLinear
from lowbitsparse.quant.primitives import fake_quant_groupwise


def _ste_fake_quant(w: torch.Tensor, n_bits: int, group_size: int, symmetric: bool) -> torch.Tensor:
    """带 STE 的分组 fake quant。"""
    q = fake_quant_groupwise(w, n_bits, group_size, symmetric)
    return w + (q - w).detach()


class DistillLinear(nn.Module):
    """可训练的量化感知 Linear。"""

    def __init__(self, linear: nn.Linear, cfg: QuantConfig, weight: torch.Tensor | None = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_bits = cfg.n_bits
        self.group_size = linear.in_features if cfg.group_size in (-1, None) else cfg.group_size
        self.symmetric = cfg.symmetric
        self.method = getattr(cfg, "method", "rtn")
        init = weight if weight is not None else fake_quant_groupwise(
            linear.weight.data, cfg.n_bits, cfg.group_size, cfg.symmetric)
        self.weight = nn.Parameter(init.to(linear.weight.dtype).clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        w = _ste_fake_quant(self.weight, self.n_bits, self.group_size, self.symmetric)
        return F.linear(x, w, self.bias)

    def export_fake_quant(self, cfg: QuantConfig, w_dq: torch.Tensor | None = None) -> FakeQuantLinear:
        """导出成推理用 FakeQuantLinear。"""
        holder = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        holder = holder.to(device=self.weight.device, dtype=self.weight.dtype)
        holder.weight = nn.Parameter(self.weight.detach().clone())
        if self.bias is not None:
            holder.bias = nn.Parameter(self.bias.detach().clone())
        return FakeQuantLinear(holder, replace(cfg, method=getattr(cfg, "method", "rtn")), w_dq=w_dq)


class DistillEmbedding(nn.Module):
    """可训练的量化感知 Embedding。"""

    def __init__(self, emb: nn.Embedding, cfg: QuantConfig, weight: torch.Tensor | None = None):
        super().__init__()
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        self.n_bits = cfg.n_bits
        self.group_size = emb.embedding_dim if cfg.group_size in (-1, None) else cfg.group_size
        self.symmetric = cfg.symmetric
        init = weight if weight is not None else fake_quant_groupwise(
            emb.weight.data, cfg.n_bits, cfg.group_size, cfg.symmetric)
        self.weight = nn.Parameter(init.to(emb.weight.dtype).clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = _ste_fake_quant(self.weight, self.n_bits, self.group_size, self.symmetric)
        return F.embedding(input_ids, w, padding_idx=self.padding_idx)

    def export_fake_quant(self, cfg: QuantConfig, w_dq: torch.Tensor | None = None) -> FakeQuantEmbedding:
        """导出成推理用 FakeQuantEmbedding。"""
        holder = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=self.padding_idx)
        holder = holder.to(device=self.weight.device, dtype=self.weight.dtype)
        holder.weight = nn.Parameter(self.weight.detach().clone())
        e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
        return FakeQuantEmbedding(holder, e_bits, cfg.group_size, cfg.symmetric, w_dq=w_dq)


def _iter_replacements(model, skip):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if any(k in name for k in skip):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            child = name.rsplit(".", 1)[-1]
            targets.append((name, parent, child, module))
    return targets


def _find_module_parent(model, target):
    for name, module in model.named_modules():
        if module is target:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            return parent, name.rsplit(".", 1)[-1]
    return None, None


def prepare_distill_student(model, cfg: QuantConfig):
    """把 student 改造成可训练的 fake-quant 版本。"""
    targets = _iter_replacements(model, cfg.skip)
    weight_cache = {}
    n = 0
    tied_output = None
    if cfg.quant_embedding:
        get_in = getattr(model, "get_input_embeddings", None)
        if get_in is not None:
            emb = get_in()
            out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
            if isinstance(emb, nn.Embedding) and isinstance(out, nn.Linear) and out.weight is emb.weight:
                tied_output = out

    def cached_weight(tensor: torch.Tensor, n_bits: int):
        key = (id(tensor), n_bits, cfg.group_size, cfg.symmetric)
        if key not in weight_cache:
            weight_cache[key] = fake_quant_groupwise(tensor, n_bits, cfg.group_size, cfg.symmetric)
        return weight_cache[key]

    # 先替换普通 Linear。
    for name, parent, child, module in targets:
        if isinstance(module, nn.Linear):
            if module is tied_output:
                continue
            if any(k in name for k in cfg.skip):
                continue
            fq = DistillLinear(module, cfg, weight=cached_weight(module.weight.data, cfg.n_bits))
            setattr(parent, child, fq)
            n += 1

    # 再替换 embedding。绑定权重时，把 lm_head 一起换掉并共享同一参数。
    if cfg.quant_embedding:
        get_in = getattr(model, "get_input_embeddings", None)
        if get_in is not None:
            emb = get_in()
            if isinstance(emb, nn.Embedding):
                e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
                emb_w = cached_weight(emb.weight.data, e_bits)
                fq_emb = DistillEmbedding(emb, replace(cfg, n_bits=e_bits), weight=emb_w).to(emb.weight.device)
                model.set_input_embeddings(fq_emb)
                n += 1

                out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
                if tied_output is not None and out is tied_output:
                    fq_lm = DistillLinear(out, replace(cfg, n_bits=e_bits), weight=emb_w)
                    parent, child = _find_module_parent(model, out)
                    if parent is not None:
                        setattr(parent, child, fq_lm)
                        n += 1
    return model, n


def export_distill_student(model, cfg: QuantConfig):
    """把可训练 student 导出成推理用 fake-quant 模型。"""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (DistillLinear, DistillEmbedding)):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            child = name.rsplit(".", 1)[-1]
            targets.append((parent, child, module))

    weight_cache = {}
    for parent, child, module in targets:
        if isinstance(module, DistillLinear):
            key = id(module.weight)
            if key not in weight_cache:
                weight_cache[key] = fake_quant_groupwise(
                    module.weight.data, cfg.n_bits, cfg.group_size, cfg.symmetric)
            fq = module.export_fake_quant(cfg, w_dq=weight_cache[key])
        else:
            e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
            key = (id(module.weight), e_bits)
            if key not in weight_cache:
                weight_cache[key] = fake_quant_groupwise(
                    module.weight.data, e_bits, cfg.group_size, cfg.symmetric)
            fq = module.export_fake_quant(replace(cfg, n_bits=e_bits), w_dq=weight_cache[key])
        setattr(parent, child, fq)
    return model
