"""把量化应用到整个模型,并统计理论压缩体积。

职责:
1) 遍历模型,把符合条件的 nn.Linear 原地替换为 FakeQuantLinear(跳过 skip 列表);
2) 计算量化后的"理论"体积——伪量化权重虽仍以 FP16 存,但真实部署时低 bit 权重
   只占 n_bits + 每组 scale/zero 的开销,据此估算压缩比。
"""
import dataclasses
import math

import torch
import torch.nn as nn

from .config import QuantConfig
from .fake_linear import FakeQuantLinear
from .fake_embedding import FakeQuantEmbedding
from .gptq import gptq_quantize_weight
from .awq import awq_quantize_weight
from .primitives import fake_quant_groupwise


def _iter_linear_names(model, skip):
    """收集待量化的 (全名, 父模块, 子名, Linear) 四元组。

    先收集再替换,避免在遍历 named_modules 的同时修改结构。
    skip 中任一关键字出现在模块全名里,则跳过该层(如 lm_head)。
    全名用于对齐 calibration 收集的逐层统计。
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in skip):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            child = name.rsplit(".", 1)[-1]
            targets.append((name, parent, child, module))
    return targets


def _compute_w_dq(name, linear, cfg, calib_stats):
    """按 cfg.method 计算该层的反量化权重;RTN 返回 None(由层内自算)。"""
    method = getattr(cfg, "method", "rtn")
    if method == "rtn":
        return None                              # 走 FakeQuantLinear 内部 RTN
    stats = (calib_stats or {}).get(name)
    if stats is None:
        raise ValueError(f"{method} 需要校准统计,但缺少层 {name} 的 stats")
    if method == "gptq":
        return gptq_quantize_weight(linear.weight.data, stats["H"], cfg)
    if method == "awq":
        return awq_quantize_weight(linear.weight.data, stats["act_scales"], cfg)
    raise ValueError(f"未知量化方法: {method}")


def _find_module_parent(model, target):
    """在 model 中定位持有 target 模块的 (父模块, 子名);找不到返回 (None, None)。"""
    for name, module in model.named_modules():
        if module is target:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            return parent, name.rsplit(".", 1)[-1]
    return None, None


def _quantize_embedding(model, cfg: QuantConfig):
    """量化输入 embedding;绑定(tied)时 lm_head 与之共享同一反量化权重。

    返回替换的模块数(embedding 计 1,若绑定的 lm_head 也换则再 +1)。
    仅当 cfg.quant_embedding=True 时调用。embedding 用 RTN(查表无激活统计)。
    """
    get_in = getattr(model, "get_input_embeddings", None)
    if get_in is None:
        return 0
    emb = get_in()
    if not isinstance(emb, nn.Embedding):
        return 0

    e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
    device = emb.weight.device
    # embedding 沿 embedding_dim 分组做 RTN 伪量化(一次),供 embed/lm_head 共享
    w_dq = fake_quant_groupwise(emb.weight.data, e_bits,
                                cfg.group_size, cfg.symmetric).to(device)

    fq_emb = FakeQuantEmbedding(emb, e_bits, cfg.group_size,
                                cfg.symmetric, w_dq=w_dq).to(device)
    model.set_input_embeddings(fq_emb)
    n = 1

    # 绑定检测:输出头与输入 embedding 共享同一权重张量时,一并换成
    # 持有同一 w_dq 的 FakeQuantLinear,避免拆散绑定导致体积不降反升。
    out = model.get_output_embeddings() if hasattr(
        model, "get_output_embeddings") else None
    if isinstance(out, nn.Linear) and out.weight is emb.weight:
        # lm_head 用 embedding 的位宽(绑定即同一矩阵),复用 w_dq 保证数值一致
        lm_cfg = dataclasses.replace(cfg, n_bits=e_bits)
        fq_lm = FakeQuantLinear(out, lm_cfg, w_dq=w_dq).to(device)
        parent, child = _find_module_parent(model, out)
        if parent is not None:
            setattr(parent, child, fq_lm)
            n += 1
    return n


def apply_quantization(model, cfg: QuantConfig, calib_stats: dict = None):
    """就地把模型中的 Linear 替换为 FakeQuantLinear;可选量化 embedding。

    参数:
        model:       HF 因果 LM。
        cfg:         QuantConfig。
        calib_stats: GPTQ/AWQ 的逐层统计({全名 → {H, act_scales}});RTN 传 None。
    返回:
        (model, n_replaced):替换后的模型与被替换模块数(含 embedding/lm_head)。
    """
    targets = _iter_linear_names(model, cfg.skip)
    for name, parent, child, linear in targets:
        w_dq = _compute_w_dq(name, linear, cfg, calib_stats)
        # 在原模块所在设备上构造量化层,避免 device 不一致
        fq = FakeQuantLinear(linear, cfg, w_dq=w_dq).to(linear.weight.device)
        setattr(parent, child, fq)          # 原地替换子模块
    n = len(targets)

    if cfg.quant_embedding:
        n += _quantize_embedding(model, cfg)
    return model, n


def target_linear_names(model, cfg: QuantConfig) -> list:
    """返回将被量化的所有 Linear 全名(供 calibration 定位收集哪些层)。"""
    return [name for name, _, _, _ in _iter_linear_names(model, cfg.skip)]


def _quant_module_bits(rows, cols, n_bits, group_size, symmetric):
    """一个量化权重矩阵 [rows, cols] 的比特开销:本体 + 每组 scale(/zero)。

    分组沿 cols(Linear 的 in_features / Embedding 的 embedding_dim)。
    """
    gs = cols if group_size in (-1, None) else group_size
    n = rows * cols
    bits = n * n_bits                                       # 权重本体
    groups = rows * math.ceil(cols / gs)
    bits += groups * 16                                     # scale(FP16)
    if not symmetric:
        bits += groups * 16                                 # 非对称额外 zero
    return n, bits


def compression_report(model) -> dict:
    """统计量化后的理论体积与等效位宽。

    对每块权重分两类计:
    - FakeQuantLinear / FakeQuantEmbedding 权重:按 n_bits + 每组 scale/zero 开销;
    - 其余参数(未量化 embedding/norm/未量化层/bias):按真实 dtype 字节数计。

    绑定(tied)权重:embed_tokens 与 lm_head 共享同一 buffer,按 id() 去重只计一次,
    避免把同一矩阵的体积算两遍。
    """
    quant_bits = 0.0     # 量化权重占的比特总数
    other_bits = 0.0     # 其余参数占的比特总数
    quant_weights = 0    # 被量化的权重元素数(用于算等效 bit)

    quant_ids = set()    # 已计入的量化 buffer id,去重(绑定共享 + 与下方参数统计)
    for module in model.modules():
        w = getattr(module, "weight", None)
        if isinstance(module, FakeQuantLinear):
            if id(w) not in quant_ids:                      # 绑定共享矩阵只计一次
                n, bits = _quant_module_bits(
                    module.out_features, module.in_features,
                    module.n_bits, module.group_size, module.symmetric)
                quant_weights += n
                quant_bits += bits
                quant_ids.add(id(w))
            if module.bias is not None and id(module.bias) not in quant_ids:
                quant_bits += module.bias.numel() * 16      # bias 不量化,FP16
                quant_ids.add(id(module.bias))
        elif isinstance(module, FakeQuantEmbedding):
            if id(w) not in quant_ids:
                n, bits = _quant_module_bits(
                    module.vocab_size, module.embedding_dim,
                    module.n_bits, module.group_size, module.symmetric)
                quant_weights += n
                quant_bits += bits
                quant_ids.add(id(w))

    # 其余参数(未量化 embedding / norm / 未量化 Linear 等):按真实 dtype 计
    for p in model.parameters():
        if id(p) in quant_ids:
            continue
        other_bits += p.numel() * p.element_size() * 8

    total_bits = quant_bits + other_bits
    size_mb = total_bits / 8 / 1024 / 1024
    return {
        "quant_weights": quant_weights,                     # 被量化的权重数
        # 等效位宽:量化部分总比特 / 量化权重数(含 scale/zero 开销,故 > n_bits)
        "effective_bits": round(quant_bits / max(quant_weights, 1), 3),
        "size_mb": round(size_mb, 3),                       # 量化后理论总体积
    }
