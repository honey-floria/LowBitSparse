"""M3 蒸馏配置。

这一层只负责把 YAML / dict 里的蒸馏参数整理成统一的 dataclass，
训练逻辑本身不关心配置来源。
"""
from dataclasses import dataclass, field, asdict

from lowbitsparse.quant.config import QuantConfig


@dataclass
class DistillConfig:
    """量化感知蒸馏的训练超参。

    字段分成四类：
    - 运行与落盘：exp_id / out_dir / seed
    - 模型与量化：teacher / student / quant
    - 数据与 batch：dataset_* / seqlen / train_samples / eval_samples / batch_size
    - 优化与损失：max_steps / lr / grad_clip / temperature / alpha_kd / beta_ce / gamma_hidden
    """

    # 实验编号。结果文件会写到 `<out_dir>/<exp_id>.json`，建议每组实验唯一，
    # 例如 `m3_distill_qwen0.5b`，避免多次运行覆盖旧结果。
    exp_id: str = "m3_distill"
    # 结果输出目录。`save_results` 会在这里落 JSON，`save_student_path` 则可单独指定
    # checkpoint 路径；两者没有强绑定。
    out_dir: str = "results"
    # 随机种子。用于固定 torch / numpy / random，以及训练窗口抽样顺序；
    # 只保证本项目可控部分复现，CUDA kernel 和 HF/datasets 仍可能有细小非确定性。
    seed: int = 42

    # teacher 模型加载参数。常用键：
    # - name: HuggingFace 模型名或本地路径，如 `Qwen/Qwen2.5-0.5B-Instruct`
    # - dtype: `float16` / `bfloat16` / `float32`，影响显存和 teacher 前向速度
    # - device: `cuda` / `cpu` / 具体设备字符串；正式蒸馏通常用 `cuda`
    # teacher 默认冻结，只提供 logits / hidden 监督。
    teacher: dict = field(default_factory=dict)
    # student 模型加载参数，键含义与 teacher 相同。通常和 teacher 同架构同权重起步，
    # 然后通过 `prepare_distill_student` 替换为可训练 fake-quant 版本。
    student: dict = field(default_factory=dict)
    # student 的量化参数。M3 训练时不会真的存 INT4，而是在 forward 中施加
    # fake-quant 误差；训练结束再按这份配置导出到 M1 推理路径的 FakeQuant* 模块。
    quant: QuantConfig = field(default_factory=QuantConfig)

    # HuggingFace datasets 数据集 id。默认使用 wikitext 做轻量 sanity check；
    # 正式实验可替换成领域数据，但要保证文本字段能被 datasets 正常读取。
    dataset_id: str = "Salesforce/wikitext"
    # 数据集子配置名。某些数据集没有 config，可在 YAML 中设为 null/空值；
    # wikitext 需要指定 `wikitext-2-raw-v1`。
    dataset_config: str = "wikitext-2-raw-v1"
    # 训练 split 名。会被 `load_token_ids` 拼成长 token 流，再切成固定长度窗口。
    train_split: str = "train"
    # 验证 split 名。用于 teacher/student PPL，对训练梯度没有贡献。
    eval_split: str = "validation"
    # 每个训练/评测窗口的 token 长度。越大越接近长上下文真实分布，但显存按近似
    # batch_size * seqlen 增长；必须小于模型 max_position_embeddings / max length。
    seqlen: int = 512
    # 训练窗口数量。总训练 token 约为 `train_samples * seqlen`；小样本适合调通流程，
    # 正式蒸馏应增大它以减少过拟合到少量窗口。
    train_samples: int = 256
    # 评测窗口数量。越大 PPL 越稳定但评测更慢；正式记录建议固定该值，便于横向比较。
    eval_samples: int = 32
    # batch 内窗口数。主要控制显存峰值；如果 OOM，优先降 batch_size，其次降 seqlen。
    batch_size: int = 2

    # 优化步数。每一步消费一个 batch；数据不足时训练循环会 cycle 复用 train_batches。
    max_steps: int = 100
    # 每隔多少步跑一次 evaluator。评测会额外做完整 forward，太小会明显拖慢训练。
    eval_every: int = 20
    # 每隔多少步打印一次训练 loss。只影响日志，不影响 history 中每步指标记录。
    log_every: int = 10
    # AdamW 学习率。M3 只训练 DistillLinear/DistillEmbedding 的 FP32 主权重，
    # 常见起点是 5e-5；若 PPL 震荡或发散应降低。
    lr: float = 5e-5
    # AdamW weight decay。蒸馏通常设 0，避免对量化主权重额外收缩；需要正则时再开启。
    weight_decay: float = 0.0
    # AdamW beta 参数。第一个控制一阶动量，第二个控制二阶矩平滑；通常不需要改。
    betas: tuple = (0.9, 0.95)
    # 梯度裁剪阈值。>0 时用 clip_grad_norm_ 限制全局梯度范数，降低 FP16/BF16
    # 训练中的偶发尖峰；设 0 或 None 可关闭。
    grad_clip: float = 1.0
    # KD 温度。teacher/student logits 会除以该值再算 KL；温度越高分布越平滑，
    # 但过高会削弱 top token 信号。通常 1~4。
    temperature: float = 2.0
    # 软标签 KL loss 权重。越大越强调贴近 teacher 输出分布，是恢复量化精度的主项。
    alpha_kd: float = 0.7
    # 硬标签 CE loss 权重。越大越强调真实下一个 token，能防止只拟合 teacher 噪声。
    beta_ce: float = 0.3
    # hidden-state MSE 权重。0 表示关闭特征对齐；>0 时 forward 会请求 hidden states，
    # 显存和时间都会增加，且 teacher/student  hidden 维度必须一致。
    gamma_hidden: float = 0.0
    # 蒸馏参数训练形态:
    # - full: 训练每个 fake-quant Linear/Embedding 的完整 FP32 主权重,精度恢复能力最强,
    #         但显存/优化器状态最大;这是 M3 首次实测采用的默认口径。
    # - scale: 冻结量化初始化权重,只训练每个输出通道一个乘性 scale,参数量最小,
    #          用于回答“仅训 scale 能恢复多少量化误差”。
    # - lora: 冻结量化初始化权重,只训练低秩 A/B 适配器,导出时把 delta 折叠回权重,
    #         参数量介于 scale 与 full 之间。
    train_mode: str = "full"
    # LoRA rank,仅在 train_mode="lora" 时生效。rank 越大容量越强、显存越高。
    lora_rank: int = 8
    # LoRA 缩放系数,实际 delta 乘以 `lora_alpha / lora_rank`。
    lora_alpha: float = 16.0
    # 是否启用 CUDA autocast。CUDA 上优先 BF16；若只能 FP16，训练 loop 会启用
    # GradScaler。CPU 下该开关自动失效。
    use_amp: bool = True
    # 是否打开 student 的 gradient checkpointing。可降低显存，但会重算前向、训练变慢；
    # 小模型或短 seqlen 通常不需要。
    gradient_checkpointing: bool = False
    # 可选 student checkpoint 路径。设置后会保存导出后的 FakeQuant* state_dict 和配置；
    # 为空则只保存 results JSON，不额外写模型权重。
    save_student_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DistillConfig":
        """从 YAML / dict 构造配置。

        参数:
            d: 来自配置文件或 CLI 的原始 dict，允许包含顶层字段，也允许把
               蒸馏字段放在 distill 子段里。

        逻辑:
            1. 先把 distill 子段展开到顶层，保证旧式 / 新式 YAML 都兼容。
            2. 把 quant 子段转成 QuantConfig，避免训练时到处写字典访问。
            3. teacher / student 允许从 model 字段回退，便于复用 M0/M1/M2 的配置。
            4. 丢掉未声明字段，避免 YAML 漏写 / 多写时污染 dataclass。
        """
        if not d:
            return cls()
        payload = dict(d)
        distill = payload.pop("distill", {})
        if isinstance(distill, dict):
            # YAML 里若把蒸馏参数收在 distill: 下，这里统一拍平到顶层。
            merged = dict(distill)
            merged.update(payload)
            payload = merged
        if "quant" in payload:
            payload["quant"] = QuantConfig.from_dict(payload["quant"])
        else:
            payload["quant"] = QuantConfig.from_dict({})
        # 兼容 teacher/student 沿用 model 字段的老配置。
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
        """转成可序列化字典。

        dataclass.asdict 会递归展开嵌套对象，这里只额外保证 quant 也被显式转成
        plain dict，便于写入 results/*.json 和 torch.save 的 config 字段。
        """
        payload = asdict(self)
        payload["quant"] = asdict(self.quant)
        return payload
