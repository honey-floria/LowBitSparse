"""M3 蒸馏的数据和评测辅助。"""
from __future__ import annotations

from itertools import cycle
from typing import Iterable

import torch


def load_token_ids(tokenizer, dataset_id: str, dataset_config: str, split: str) -> torch.Tensor:
    """把数据集拼成一条 token 序列。"""
    from datasets import load_dataset

    data = load_dataset(dataset_id, dataset_config, split=split)
    text = "\n\n".join(data["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


def fixed_length_windows(token_ids: torch.Tensor, seqlen: int, max_samples: int | None = None,
                         shuffle: bool = False, seed: int = 42) -> torch.Tensor:
    """切成固定长度的窗口。"""
    if token_ids.dim() != 1:
        token_ids = token_ids.reshape(-1)
    seqlen = max(int(seqlen), 1)
    n_windows = token_ids.numel() // seqlen
    if max_samples is not None:
        n_windows = min(n_windows, int(max_samples))
    if n_windows <= 0:
        raise ValueError("token_ids 太短,无法切出训练窗口")
    windows = token_ids[: n_windows * seqlen].reshape(n_windows, seqlen)
    if shuffle and n_windows > 1:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        order = torch.randperm(n_windows, generator=g)
        windows = windows[order]
    return windows


def make_batches(windows: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """把窗口张量分成 batch 列表。"""
    batch_size = max(int(batch_size), 1)
    batches = []
    for i in range(0, windows.shape[0], batch_size):
        batches.append(windows[i:i + batch_size].clone())
    return batches


def batch_cycle(batches: Iterable[torch.Tensor]):
    """循环取 batch。"""
    return cycle(list(batches))


@torch.no_grad()
def strided_ppl_from_ids(model, token_ids: torch.Tensor, seqlen: int = 2048,
                         stride: int | None = None, device: str | None = None,
                         max_samples: int | None = None) -> dict:
    """在已切好的 token 序列上计算 strided PPL。"""
    if device is None:
        device = next(model.parameters()).device
    if stride is None:
        stride = seqlen
    if token_ids.dim() != 1:
        token_ids = token_ids.reshape(-1)
    n_tokens = token_ids.numel()
    if max_samples is not None:
        n_tokens = min(n_tokens, max_samples * seqlen)

    nll_sum = 0.0
    n_counted = 0
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + seqlen, n_tokens)
        trg_len = end - prev_end
        input_ids = token_ids[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        out = model(input_ids, labels=target_ids)
        valid = trg_len - 1 if trg_len > 1 else 1
        nll_sum += out.loss.float().item() * valid
        n_counted += valid
        prev_end = end
        if end >= n_tokens:
            break
    ppl = float(torch.exp(torch.tensor(nll_sum / max(n_counted, 1))))
    return {
        "ppl": round(ppl, 4),
        "seqlen": seqlen,
        "stride": stride,
        "n_tokens": int(n_tokens),
    }


def build_ppl_evaluator(eval_token_ids: torch.Tensor, seqlen: int, stride: int | None = None,
                        max_samples: int | None = None):
    """构造一个用于蒸馏曲线的 PPL evaluator。"""
    def _eval(model) -> dict:
        return strided_ppl_from_ids(
            model, eval_token_ids, seqlen=seqlen, stride=stride,
            max_samples=max_samples)
    return _eval
