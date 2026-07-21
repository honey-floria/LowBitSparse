"""模型与分词器加载,以及模型体积统计。

封装 HuggingFace 的加载细节(dtype 映射、设备放置、GPU 降级),
让上层 CLI 只需给一个模型名和精度即可拿到可评测的 eval-mode 模型。
"""
from typing import Tuple

import torch                                              # 张量 / dtype / 设备
from transformers import AutoModelForCausalLM, AutoTokenizer  # HF 通用加载器


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",  # HF Hub 模型 id
    dtype: str = "float16",                          # 权重精度:float16/bfloat16/float32
    device: str = "cuda",                            # 目标设备:cuda/cpu
    trust_remote_code: bool = True,                  # Qwen 自定义代码需为 True
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """加载 HF 因果语言模型与配套分词器。

    参数:
        model_name:        HuggingFace 模型标识。
        dtype:             字符串精度名,经 getattr(torch, dtype) 转为 torch.dtype。
        device:            "cuda" 无 GPU 时自动降级为 "cpu"。
        trust_remote_code: 是否信任仓库自带建模代码(Qwen2.5 需要)。
    返回:
        (model, tokenizer):model 已 .to(device) 且置于 eval 模式。
    """
    # 把 "float16" 这类字符串映射成真正的 torch.float16 dtype 对象
    torch_dtype = getattr(torch, dtype)
    # 分词器:与模型同源加载,保证 vocab / 特殊 token 一致
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    # 按指定精度加载权重,显著省显存(FP16 相比 FP32 减半)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    # 请求 cuda 但环境无 GPU(如本地)时,自动回退 cpu,避免 .to 报错
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)   # 把权重搬到目标设备
    model.eval()       # 评测/量化阶段固定为推理模式:关 dropout、启用缓存路径
    return model, tokenizer


def model_size_report(model: torch.nn.Module) -> dict:
    """统计参数量与理论体积(逐参数按其 dtype 字节数求和)。

    参数:
        model: 任意 nn.Module。
    返回:
        dict:
            params:          参数总数(个)。
            params_millions: 参数量(百万,M),便于口头对比。
            bytes:           理论占用字节数 = Σ(numel × element_size)。
            size_mb:         体积(MB),量化后此值应显著下降。

    说明:按实际 dtype 计,故 FP16 与 INT8 权重会得到不同 size_mb,
    可直接用于计算压缩比。仅统计 parameters,不含激活/KV cache。
    """
    n_params = 0   # 累计参数个数
    n_bytes = 0    # 累计字节数
    for p in model.parameters():
        n_params += p.numel()                 # 该张量元素个数
        n_bytes += p.numel() * p.element_size()  # 元素数 × 每元素字节(dtype 决定)
    return {
        "params": n_params,
        "params_millions": round(n_params / 1e6, 3),
        "bytes": n_bytes,
        "size_mb": round(n_bytes / 1024 / 1024, 3),
    }
