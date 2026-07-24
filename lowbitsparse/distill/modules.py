"""M3 的可训练 fake-quant 模块和导出工具。

M1 的 FakeQuantLinear 是“构造时量化一次、forward 只读”的推理形态；
M3 需要训练 student，所以这里提供带 STE 的可训练 wrapper：
前向看到量化误差，反向仍把梯度传给 FP32 主权重。
"""
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
    """带 STE(straight-through estimator)的分组 fake quant。

    参数:
        w: FP32 主权重，shape 通常为 [out_features, in_features]。
        n_bits / group_size / symmetric: 与 QuantConfig 同义。

    返回:
        数值上等于量化-反量化权重，梯度上等于原始 w 的张量。

    逻辑:
        q = fake_quant(w) 负责制造量化误差；`w + (q - w).detach()` 让 forward
        用 q，backward 却绕过 round/clamp 的不可导部分，把梯度直接传给 w。
    """
    q = fake_quant_groupwise(w, n_bits, group_size, symmetric)
    return w + (q - w).detach()


class DistillLinear(nn.Module):
    """可训练的量化感知 Linear。

    参数:
        linear: 原始 `nn.Linear` 模块。这里读取它的 `in_features`、`out_features`、
                bias 是否存在、权重所在 device 和原始 compute dtype；构造后不会再
                持有原模块引用。
        cfg: `QuantConfig`。使用其中的 `n_bits`、`group_size`、`symmetric` 和
             `method`；`skip` 已在外层筛选完成，进入本类时表示该 Linear 一定要替换。
        weight: 可选初始化权重，shape 必须与 `linear.weight` 一致。默认会用
                `linear.weight` 先做一次 fake-quant 初始化；绑定权重或缓存场景传入
                该参数，可保证多个 wrapper 共享同一份初始量化结果。
        train_mode: 蒸馏消融训练形态。`full` 训练完整权重；`scale` 只训练输出
                    通道 scale；`lora` 只训练低秩 A/B adapter。
        lora_rank: LoRA 低秩维度，仅在 `train_mode="lora"` 时生效。
        lora_alpha: LoRA delta 缩放系数，实际缩放为 `lora_alpha / lora_rank`。

    设计:
        - `weight` / `bias` 使用 FP32 主权重，避免 AMP GradScaler 遇到 FP16 梯度。
        - `compute_dtype` 记录原模型 dtype，forward 时再 cast 到输入 dtype，
          兼容 HF FP16/BF16 模型和 autocast。
        - `scale` / `lora` 都把基础权重注册成 buffer，导出时再折叠训练参数，
          因此部署形态仍是普通 FakeQuantLinear，不额外引入 adapter 模块。
    """

    def __init__(
        self,
        linear: nn.Linear,
        cfg: QuantConfig,
        weight: torch.Tensor | None = None,
        train_mode: str = "full",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_bits = cfg.n_bits
        self.group_size = linear.in_features if cfg.group_size in (-1, None) else cfg.group_size
        self.symmetric = cfg.symmetric
        self.method = getattr(cfg, "method", "rtn")
        self.train_mode = str(train_mode).lower()
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        # 原模型可能是 FP16/BF16；训练主权重保持 FP32，计算时再转回输入 dtype。
        self.compute_dtype = linear.weight.dtype
        init = weight if weight is not None else fake_quant_groupwise(
            linear.weight.data, cfg.n_bits, cfg.group_size, cfg.symmetric)
        if self.train_mode == "full":
            self.weight = nn.Parameter(init.float().clone())
        else:
            self.register_buffer("weight", init.float().clone())
        if linear.bias is not None and self.train_mode == "full":
            self.bias = nn.Parameter(linear.bias.data.float().clone())
        elif linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.float().clone())
        else:
            self.bias = None

        if self.train_mode == "scale":
            self.weight_scale = nn.Parameter(torch.ones(self.out_features, 1))
        elif self.train_mode == "lora":
            if self.lora_rank <= 0:
                raise ValueError("lora_rank 必须 > 0")
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.lora_scaling = self.lora_alpha / self.lora_rank
        elif self.train_mode != "full":
            raise ValueError(f"未知 train_mode: {self.train_mode}")

    def _effective_weight(self) -> torch.Tensor:
        """返回已经折叠训练参数的 FP32 权重。"""
        if self.train_mode == "scale":
            return self.weight * self.weight_scale
        if self.train_mode == "lora":
            delta = torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
            return self.weight + delta
        return self.weight

    def forward(self, x):
        # forward 使用量化后的权重值；STE 让梯度仍流向 FP32 主权重。
        w = _ste_fake_quant(self._effective_weight(), self.n_bits, self.group_size, self.symmetric).to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def export_weight(self) -> torch.Tensor:
        """导出前的 FP32 折叠权重。"""
        return self._effective_weight().detach().clone()

    def export_fake_quant(self, cfg: QuantConfig, w_dq: torch.Tensor | None = None) -> FakeQuantLinear:
        """导出成推理用 FakeQuantLinear。

        参数:
            cfg: 导出时使用的量化配置。通常与训练 cfg 相同；导出时会读取
                 `n_bits`、`group_size`、`symmetric` 和 `method` 来构造
                 `FakeQuantLinear`。
            w_dq: 可选的已量化-反量化权重缓存，shape 与 `self.weight` 一致。
                  tied weight 或重复导出同一参数时传入它，避免数值和压缩统计出现
                  “同一份权重被重新量化多次”的差异。

        返回:
            与 M1 推理路径一致的 FakeQuantLinear。导出后权重不再可训练。
        """
        # FakeQuantLinear 构造函数需要一个 nn.Linear holder，这里只用它承载 shape/dtype/bias。
        holder = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        holder = holder.to(device=self.weight.device, dtype=self.compute_dtype)
        holder.weight = nn.Parameter(self.export_weight().to(self.compute_dtype))
        if self.bias is not None:
            holder.bias = nn.Parameter(self.bias.detach().clone().to(self.compute_dtype))
        return FakeQuantLinear(holder, replace(cfg, method=getattr(cfg, "method", "rtn")), w_dq=w_dq)


class DistillEmbedding(nn.Module):
    """可训练的量化感知 Embedding。

    参数:
        emb: 原始 `nn.Embedding` 模块。这里读取词表大小、embedding 维度、
             padding_idx、权重 device 和原始 dtype；构造后用 FP32 主权重训练。
        cfg: embedding 使用的量化配置。若 YAML 里设置了 `embedding_bits`，
             调用方会先通过 `replace(cfg, n_bits=e_bits)` 把它转成当前 cfg 的
             `n_bits`，因此本类只需要读取 `cfg.n_bits`。
        weight: 可选初始化权重，shape 必须与 `emb.weight` 一致。主要用于输入
                embedding 与 lm_head 绑定权重时，让二者从同一个量化缓存初始化。
        train_mode: embedding 的训练形态。支持 `full` 和 `scale`；`lora` 下
                    embedding 默认保持冻结，因为 LoRA 主要作用在线性层。

    说明:
        Embedding 查表本身没有 matmul，M3 仍用同一套 STE 权重量化逻辑。
        默认配置不量化 embedding，只有 `quant_embedding=True` 时才会走到这里。
    """

    def __init__(self, emb: nn.Embedding, cfg: QuantConfig, weight: torch.Tensor | None = None,
                 train_mode: str = "full"):
        super().__init__()
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        self.n_bits = cfg.n_bits
        self.group_size = emb.embedding_dim if cfg.group_size in (-1, None) else cfg.group_size
        self.symmetric = cfg.symmetric
        self.train_mode = str(train_mode).lower()
        # 记录原始 dtype，导出和 forward 输出都保持和 HF 模型一致。
        self.compute_dtype = emb.weight.dtype
        init = weight if weight is not None else fake_quant_groupwise(
            emb.weight.data, cfg.n_bits, cfg.group_size, cfg.symmetric)
        if self.train_mode == "full":
            self.weight = nn.Parameter(init.float().clone())
        else:
            self.register_buffer("weight", init.float().clone())
        if self.train_mode == "scale":
            self.weight_scale = nn.Parameter(torch.ones(1, self.embedding_dim))
        elif self.train_mode not in ("full", "lora"):
            raise ValueError(f"未知 train_mode: {self.train_mode}")

    def _effective_weight(self) -> torch.Tensor:
        """返回已经折叠训练参数的 FP32 embedding 权重。"""
        if self.train_mode == "scale":
            return self.weight * self.weight_scale
        return self.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # F.embedding 要求权重和后续层 dtype 兼容，输出再转回原模型计算 dtype。
        w = _ste_fake_quant(self._effective_weight(), self.n_bits, self.group_size, self.symmetric)
        return F.embedding(input_ids, w).to(self.compute_dtype)

    def export_weight(self) -> torch.Tensor:
        """导出前的 FP32 折叠 embedding 权重。"""
        return self._effective_weight().detach().clone()

    def export_fake_quant(self, cfg: QuantConfig, w_dq: torch.Tensor | None = None) -> FakeQuantEmbedding:
        """导出成推理用 FakeQuantEmbedding。

        参数:
            cfg: 导出时使用的量化配置。embedding 可用 `embedding_bits` 单独设位宽；
                 若为 None，则沿用 `cfg.n_bits`。
            w_dq: 可选的已量化-反量化权重缓存，shape 与 `self.weight` 一致。
                  绑定权重场景下传入它，可让 embedding 和 lm_head 对同一份权重
                  使用一致的量化结果。
        """
        holder = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=self.padding_idx)
        holder = holder.to(device=self.weight.device, dtype=self.compute_dtype)
        holder.weight = nn.Parameter(self.export_weight().to(self.compute_dtype))
        e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
        return FakeQuantEmbedding(holder, e_bits, cfg.group_size, cfg.symmetric, w_dq=w_dq)


def _iter_replacements(model, skip):
    """枚举需要替换的 Linear / Embedding 模块。

    返回:
        (name, parent, child, module) 列表。parent/child 用于后续 setattr，
        避免只拿到 module 却无法在原模型里替换。
    """
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
    """根据模块对象反查其父模块和属性名。

    tied weight 场景下会先通过 `get_output_embeddings()` 拿到 lm_head 对象，
    再需要回到原模型树里替换它，所以这里按 object identity 查找。
    """
    for name, module in model.named_modules():
        if module is target:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            return parent, name.rsplit(".", 1)[-1]
    return None, None


def prepare_distill_student(model, cfg: QuantConfig, train_mode: str = "full",
                            lora_rank: int = 8, lora_alpha: float = 16.0):
    """把 student 改造成可训练的 fake-quant 版本。

    参数:
        model: 已加载到目标设备的 student 模型。函数会原地修改这个对象：
               冻结全部已有参数，并把命中的 Linear/Embedding 替换成 Distill* wrapper。
        cfg: `QuantConfig`。关键字段包括：
             `skip` 用模块名关键字排除层，例如 `lm_head`；
             `n_bits` / `group_size` / `symmetric` 决定 Linear 权重量化方式；
             `quant_embedding` 决定是否量化输入 embedding；
             `embedding_bits` 可让 embedding 使用不同位宽。
        train_mode: `full` / `scale` / `lora` 三选一，用于 M3 消融。
        lora_rank: LoRA rank，仅在 train_mode 为 lora 时生效。
        lora_alpha: LoRA 缩放系数，仅在 train_mode 为 lora 时生效。

    返回:
        (model, n_replaced)，model 为原地改造后的对象，n_replaced 为替换层数。

    逻辑:
        1. 先冻结原模型所有参数，避免未替换层参与训练。
        2. 把目标 Linear 替换成 DistillLinear，forward 走 STE fake-quant。
        3. 可选替换 input embedding；如果 lm_head 与 embedding 绑定权重，
           需要一起处理，保证数值和压缩统计不重复。
    """
    # 训练只更新蒸馏 wrapper 的 FP32 主权重；原始 FP16/BF16 参数全部冻结。
    for p in model.parameters():
        p.requires_grad_(False)
    targets = _iter_replacements(model, cfg.skip)
    weight_cache = {}
    n = 0
    tied_output = None
    if cfg.quant_embedding:
        get_in = getattr(model, "get_input_embeddings", None)
        if get_in is not None:
            emb = get_in()
            out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
            # Qwen 这类模型常把输入 embedding 和 lm_head 权重绑定；先记住这个关系。
            if isinstance(emb, nn.Embedding) and isinstance(out, nn.Linear) and out.weight is emb.weight:
                tied_output = out

    def cached_weight(tensor: torch.Tensor, n_bits: int):
        # 同一份权重只量化一次，绑定权重时也能共享同一个初始化张量。
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
            fq = DistillLinear(
                module, cfg, weight=cached_weight(module.weight.data, cfg.n_bits),
                train_mode=train_mode, lora_rank=lora_rank, lora_alpha=lora_alpha)
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
                fq_emb = DistillEmbedding(
                    emb, replace(cfg, n_bits=e_bits), weight=emb_w,
                    train_mode=train_mode).to(emb.weight.device)
                model.set_input_embeddings(fq_emb)
                n += 1

                out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
                if tied_output is not None and out is tied_output:
                    fq_lm = DistillLinear(
                        out, replace(cfg, n_bits=e_bits), weight=emb_w,
                        train_mode=train_mode, lora_rank=lora_rank, lora_alpha=lora_alpha)
                    parent, child = _find_module_parent(model, out)
                    if parent is not None:
                        setattr(parent, child, fq_lm)
                        n += 1
    return model, n


def export_distill_student(model, cfg: QuantConfig):
    """把可训练 student 导出成推理用 fake-quant 模型。

    参数:
        model: 训练后的 student，内部仍包含 DistillLinear/DistillEmbedding。
               函数会原地替换这些 wrapper；调用后该模型用于评测/推理，不再用于继续训练。
        cfg: 导出量化配置。通常与训练配置一致；如果导出位宽或 group_size 与训练不一致，
             最终 PPL 可能和训练 history 不匹配，除非是在做专门消融。

    返回:
        原地替换后的 model，内部蒸馏 wrapper 已换成 M1 推理路径的 FakeQuant*。

    逻辑:
        - 先收集目标模块，再统一替换，避免遍历 named_modules 时修改模块树。
        - 导出时重新量化一次训练后的 FP32 主权重，得到最终部署权重。
        - 用 weight_cache 避免 tied weight 被重复量化、重复统计。
    """
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
            # 普通 Linear 按线性层位宽导出。
            key = id(module)
            if key not in weight_cache:
                weight_cache[key] = fake_quant_groupwise(
                    module.export_weight(), cfg.n_bits, cfg.group_size, cfg.symmetric)
            fq = module.export_fake_quant(cfg, w_dq=weight_cache[key])
        else:
            # Embedding 可单独使用 embedding_bits；None 时沿用线性层位宽。
            e_bits = cfg.embedding_bits if cfg.embedding_bits is not None else cfg.n_bits
            key = (id(module), e_bits)
            if key not in weight_cache:
                weight_cache[key] = fake_quant_groupwise(
                    module.export_weight(), e_bits, cfg.group_size, cfg.symmetric)
            fq = module.export_fake_quant(replace(cfg, n_bits=e_bits), w_dq=weight_cache[key])
        setattr(parent, child, fq)
    return model
