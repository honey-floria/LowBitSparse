"""量化配置。

用一个 dataclass 收拢所有量化超参,避免函数间层层传参,也便于从 YAML 构造。
"""
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """权重量化超参。

    字段:
        n_bits:     量化位宽(8/4/3…);越低压缩越狠、误差越大。
        group_size: 分组大小,沿输入维每 group_size 个权重共享一组 scale/zero;
                    -1 表示 per-channel(整行一组)。越小越精细、开销越大。
        symmetric:  True=对称量化(zero 固定为 0,省 1 个 zero 存储);
                    False=非对称(带 zero,能贴合非零均值分布,RTN 精度通常更好)。
        method:     量化算法名,M1 先支持 "rtn";后续扩展 "gptq"/"awq"。
        skip:       不量化的模块名关键字列表(如 lm_head),这些层保持 FP16。
    """
    n_bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    method: str = "rtn"
    skip: tuple = ("lm_head",)

    @classmethod
    def from_dict(cls, d: dict) -> "QuantConfig":
        """从 dict(通常来自 YAML 的 quant 段)构造,忽略未知键。"""
        if not d:
            return cls()
        fields = cls.__dataclass_fields__
        # 只挑 dataclass 已声明的字段,防止 YAML 里多余键导致 TypeError
        return cls(**{k: v for k, v in d.items() if k in fields})
