"""伪量化 Linear 层。

用一个与 nn.Linear 接口一致的模块替换原始 Linear:权重在构造时被 RTN 量化-反量化
一次并缓存为普通 FP16 张量,forward 走标准 matmul(不在每步重复量化,评测更快)。
同时记录量化元数据(n_bits/group_size),供 apply.py 计算理论压缩体积。
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
        in_features/out_features:    维度,复用于统计。
    """

    def __init__(self, linear: nn.Linear, cfg):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # 记录量化配置,供后续体积统计读取
        self.n_bits = cfg.n_bits
        self.group_size = (linear.in_features if cfg.group_size in (-1, None)
                           else cfg.group_size)
        self.symmetric = cfg.symmetric

        # 构造时一次性做 RTN 伪量化,缓存反量化权重
        with torch.no_grad():
            w_dq = rtn_quantize_weight(linear.weight.data, cfg)
        # 注册为 buffer 而非 Parameter:M1 评测无需训练,随 .to(device)/state_dict 迁移
        # (M3 QAT 时会改为可训练 scale,届时另设计)
        self.register_buffer("weight", w_dq)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        """标准线性变换;权重已是反量化后的 FP16,直接 matmul。"""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        """print(model) 时显示量化信息,便于核对替换是否生效。"""
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.n_bits}, group={self.group_size}, "
                f"sym={self.symmetric}")
