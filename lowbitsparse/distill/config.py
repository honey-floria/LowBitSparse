"""M3 蒸馏配置。"""
from dataclasses import dataclass, field, asdict

from lowbitsparse.quant.config import QuantConfig


@dataclass
class DistillConfig:
    """量化感知蒸馏的训练超参。"""

    exp_id: str = "m3_distill"
    out_dir: str = "results"
    seed: int = 42

    teacher: dict = field(default_factory=dict)
    student: dict = field(default_factory=dict)
    quant: QuantConfig = field(default_factory=QuantConfig)

    dataset_id: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    seqlen: int = 512
    train_samples: int = 256
    eval_samples: int = 32
    batch_size: int = 2

    max_steps: int = 100
    eval_every: int = 20
    log_every: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    temperature: float = 2.0
    alpha_kd: float = 0.7
    beta_ce: float = 0.3
    gamma_hidden: float = 0.0
    use_amp: bool = True
    gradient_checkpointing: bool = False
    save_student_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DistillConfig":
        """从 YAML / dict 构造配置。"""
        if not d:
            return cls()
        payload = dict(d)
        distill = payload.pop("distill", {})
        if isinstance(distill, dict):
            merged = dict(distill)
            merged.update(payload)
            payload = merged
        if "quant" in payload:
            payload["quant"] = QuantConfig.from_dict(payload["quant"])
        else:
            payload["quant"] = QuantConfig.from_dict({})
        payload.setdefault("teacher", payload.get("model", {}))
        payload.setdefault("student", payload.get("student", payload["teacher"]))
        allowed = cls.__dataclass_fields__
        filtered = {k: v for k, v in payload.items() if k in allowed}
        if isinstance(filtered.get("teacher"), dict) and not filtered["teacher"]:
            filtered["teacher"] = {}
        if isinstance(filtered.get("student"), dict) and not filtered["student"]:
            filtered["student"] = {}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """转成可序列化字典。"""
        payload = asdict(self)
        payload["quant"] = asdict(self.quant)
        return payload
