"""把量化应用到整个模型,并统计理论压缩体积。

职责:
1) 遍历模型,把符合条件的 nn.Linear 原地替换为 FakeQuantLinear(跳过 skip 列表);
2) 计算量化后的"理论"体积——伪量化权重虽仍以 FP16 存,但真实部署时低 bit 权重
   只占 n_bits + 每组 scale/zero 的开销,据此估算压缩比。
"""
import torch
import torch.nn as nn

from .config import QuantConfig
from .fake_linear import FakeQuantLinear


def _iter_linear_names(model, skip):
    """收集待量化的 (父模块, 子名, Linear) 三元组。

    先收集再替换,避免在遍历 named_modules 的同时修改结构。
    skip 中任一关键字出现在模块全名里,则跳过该层(如 lm_head)。
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in skip):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            child = name.rsplit(".", 1)[-1]
            targets.append((parent, child, module))
    return targets


def apply_quantization(model, cfg: QuantConfig):
    """就地把模型中的 Linear 替换为 FakeQuantLinear。

    参数:
        model: HF 因果 LM。
        cfg:   QuantConfig。
    返回:
        (model, n_replaced):替换后的模型与被替换层数。
    """
    targets = _iter_linear_names(model, cfg.skip)
    for parent, child, linear in targets:
        # 在原模块所在设备上构造量化层,避免 device 不一致
        fq = FakeQuantLinear(linear, cfg).to(linear.weight.device)
        setattr(parent, child, fq)          # 原地替换子模块
    return model, len(targets)


def compression_report(model) -> dict:
    """统计量化后的理论体积与等效位宽。

    对每个参数分两类计:
    - FakeQuantLinear 权重:按 n_bits 计,外加每组 scale/zero 的 FP16 开销;
    - 其余参数(embedding/norm/未量化层/bias):按其真实 dtype 字节数计。
    """
    quant_bits = 0.0     # 量化权重占的比特总数
    other_bits = 0.0     # 其余参数占的比特总数
    quant_weights = 0    # 被量化的权重元素数(用于算等效 bit)

    quant_ids = set()    # 记录已计入的量化层 buffer id,避免与下方参数统计重复
    for module in model.modules():
        if isinstance(module, FakeQuantLinear):
            n = module.out_features * module.in_features
            quant_weights += n
            quant_bits += n * module.n_bits                # 权重本体的低 bit 存储
            # 每行的组数 = ceil(in / group_size);scale 每组 1 个 FP16
            import math
            groups = module.out_features * math.ceil(
                module.in_features / module.group_size)
            quant_bits += groups * 16                       # scale 开销
            if not module.symmetric:
                quant_bits += groups * 16                   # 非对称额外的 zero 开销
            quant_ids.add(id(module.weight))                # 标记该 buffer 已计
            if module.bias is not None:                     # bias 不量化,按 FP16 计
                quant_bits += module.bias.numel() * 16
                quant_ids.add(id(module.bias))

    # 其余参数(embedding / norm / 未量化 Linear 等):按真实 dtype 计
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
