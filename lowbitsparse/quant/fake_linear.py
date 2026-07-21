"""伪量化 Linear 层。

与 nn.Linear 接口兼容:权重在构造时做一次量化-反量化并缓存为 FP16 buffer,
forward 走标准 matmul。同时记录量化元数据供压缩比统计。

设计说明:
- RTN:构造时自行调 rtn_quantize_weight 计算 w_dq。
- GPTQ / AWQ:由上层(apply.py)提前算好 w_dq 后,直接传 w_dq 参数跳过内部重算。
这样量化器与层结构解耦:FakeQuantLinear 只负责持有量化权重和统计元数据,
具体用哪种量化算法由 apply.py 决定。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtn import rtn_quantize_weight


class FakeQuantLinear(nn.Module):
    """把一个 nn.Linear 替换为"权重已被量化误差污染"的等价层。

    属性:
        weight/bias: 反量化后的权重与原偏置(bias 不量化)。
        n_bits/group_size/symmetric: 量化元数据,用于压缩比统计。
        in_features/out_features:    维度。
        method:      量化算法名,仅用于 extra_repr 展示。
    """

    def __init__(self, linear: nn.Linear, cfg, w_dq: torch.Tensor = None):
        """
        参数:
            linear: 原始 nn.Linear。
            cfg:    QuantConfig。
            w_dq:   可选——已计算好的反量化权重;若为 None 则用 RTN 就地计算。
                    GPTQ/AWQ 由 apply.py 传入;RTN 走旧路径保持向后兼容。
        """
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_bits = cfg.n_bits
        self.group_size = (linear.in_features if cfg.group_size in (-1, None)
                           else cfg.group_size)
        self.symmetric = cfg.symmetric
        self.method = getattr(cfg, "method", "rtn")

        with torch.no_grad():
            if w_dq is None:
                # RTN 路径:构造时自行计算
                w_dq = rtn_quantize_weight(linear.weight.data, cfg)
        self.register_buffer("weight", w_dq.to(linear.weight.dtype))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        """标准线性变换;权重已是反量化后的 FP16,直接 matmul。"""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.n_bits}, group={self.group_size}, "
                f"sym={self.symmetric}, method={self.method}")
