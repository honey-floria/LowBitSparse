"""WikiText-2 困惑度评测(滑窗 strided 方式,标准做法)。"""
import torch
from tqdm import tqdm


@torch.no_grad()
def eval_wikitext2_ppl(
    model,
    tokenizer,
    seqlen: int = 2048,
    stride: int = 2048,
    split: str = "test",
    device: str = None,
    max_samples: int = None,
) -> dict:
    """计算 WikiText-2-raw 的困惑度。

    stride < seqlen 时窗口重叠、只对新 token 计损失,PPL 更准。
    stride == seqlen 为不重叠切分,最快。
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids
    n_tokens = enc.size(1)
    if max_samples is not None:
        n_tokens = min(n_tokens, max_samples * seqlen)

    nll_sum = 0.0
    n_counted = 0
    prev_end = 0
    for begin in tqdm(range(0, n_tokens, stride), desc="ppl"):
        end = min(begin + seqlen, n_tokens)
        trg_len = end - prev_end  # 本窗口真正计损的 token 数
        input_ids = enc[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # 只对新 token 计损

        out = model(input_ids, labels=target_ids)
        # HF 返回的是均值 loss;乘以有效 token 数得到 nll 之和
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
