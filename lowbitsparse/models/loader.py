"""模型与分词器加载,以及体积统计。"""
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dtype: str = "float16",
    device: str = "cuda",
    trust_remote_code: bool = True,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """加载 HF 因果语言模型与分词器。

    dtype: float16 / bfloat16 / float32
    device: cuda / cpu (A100 用 cuda)
    """
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def model_size_report(model: torch.nn.Module) -> dict:
    """统计参数量与理论体积(按各参数 dtype 的字节数求和)。"""
    n_params = 0
    n_bytes = 0
    for p in model.parameters():
        n_params += p.numel()
        n_bytes += p.numel() * p.element_size()
    return {
        "params": n_params,
        "params_millions": round(n_params / 1e6, 3),
        "bytes": n_bytes,
        "size_mb": round(n_bytes / 1024 / 1024, 3),
    }
