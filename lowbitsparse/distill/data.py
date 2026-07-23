"""M3 蒸馏的数据和评测辅助。

这里不放训练状态，只放“数据准备”和“蒸馏评测”两类纯函数，
便于单测和 notebook 复用。
"""
from __future__ import annotations

from itertools import cycle
from typing import Iterable

import torch


def load_token_ids(tokenizer, dataset_id: str, dataset_config: str, split: str) -> torch.Tensor:
    """把数据集拼成一条 token 序列。

    参数:
        tokenizer: HuggingFace tokenizer。需要支持批量调用并返回 `input_ids`；
                   本函数不会修改 tokenizer，也不会添加 padding。
        dataset_id: HF datasets 数据集 id 或本地 datasets 路径，例如
                    `Salesforce/wikitext`。
        dataset_config: 数据集 config 名，例如 `wikitext-2-raw-v1`；如果数据集
                        没有 config，可由调用方传 None/空值并确保 datasets 能接受。
        split: 要读取的数据划分名，例如 `train`、`validation` 或 `test`。

    返回:
        一维 LongTensor，包含该 split 的全部 token。

    逻辑:
        - 先按文本 chunk 批量 tokenize，避免一次性喂给 tokenizer 时触发超长序列警告。
        - 不加 special tokens，因为蒸馏训练和 PPL 评测都按原始语料 token 流处理。
        - 这里返回拼接后的长序列，后续再由 fixed_length_windows 切窗。
    """
    from datasets import load_dataset

    data = load_dataset(dataset_id, dataset_config, split=split)
    all_ids = []
    chunk_size = 256
    texts = data["text"]
    for begin in range(0, len(texts), chunk_size):
        # 过滤空串，减少无效编码；chunk 化是为了控制 tokenizer 输入长度。
        batch_text = [t for t in texts[begin: begin + chunk_size] if t]
        if not batch_text:
            continue
        encoded = tokenizer(
            batch_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        for seq in encoded["input_ids"]:
            all_ids.extend(seq)
    return torch.tensor(all_ids, dtype=torch.long)


def fixed_length_windows(token_ids: torch.Tensor, seqlen: int, max_samples: int | None = None,
                         shuffle: bool = False, seed: int = 42) -> torch.Tensor:
    """把一维 token 流切成固定长度窗口。

    参数:
        token_ids: 一维 token 序列，dtype 通常为 torch.long。若传入多维张量，
                   会先 reshape 成一维 token 流。
        seqlen: 每个窗口长度，也是后续模型前向的上下文长度。值越大上下文越完整，
                但显存和计算量越高；必须小于模型可接受的最大序列长度。
        max_samples: 最多保留多少个窗口。None 表示使用 token 流能切出的全部完整窗口；
                     正式训练可增大，快速调试可设小。
        shuffle: 是否打乱窗口顺序。训练集通常打开以减少相邻窗口相关性；验证集通常关闭，
                 让 PPL 更稳定、可复现。
        seed: 打乱窗口时的随机种子，仅在 `shuffle=True` 且窗口数大于 1 时生效。

    返回:
        shape = [n_windows, seqlen] 的窗口张量。

    逻辑:
        - 先把序列裁成 seqlen 的整数倍，避免最后一个短窗口影响 batch 形状。
        - 只在 CPU generator 上打乱，保证和模型设备无关且可复现。
    """
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
    """把窗口张量切成 batch 列表。

    参数:
        windows: `fixed_length_windows` 的输出，shape=[n_windows, seqlen]。
                 函数会按第 0 维切分，不改变每个窗口内部 token 顺序。
        batch_size: 每个 batch 包含的窗口数。最后一个 batch 允许小于 batch_size；
                    训练 loop 会直接消费这些张量。

    返回:
        list[Tensor]，每个元素 shape=[batch, seqlen]，并 clone 一份避免后续原地操作
        影响原 windows。

    这里返回 list 而不是 DataLoader，是为了让蒸馏 loop 能直接 cycle()，
    避免额外的 worker / collate 开销。
    """
    batch_size = max(int(batch_size), 1)
    batches = []
    for i in range(0, windows.shape[0], batch_size):
        batches.append(windows[i:i + batch_size].clone())
    return batches


def batch_cycle(batches: Iterable[torch.Tensor]):
    """把 batch 列表包装成无限循环迭代器。

    参数:
        batches: 有限 batch iterable。函数会先转成 list，再交给 itertools.cycle；
                 因此传入生成器也能重复迭代。

    返回:
        无限迭代器，每次 yield 一个 batch tensor。

    适合蒸馏这种固定步数训练：训练循环只关心 step，不关心 epoch。
    """
    return cycle(list(batches))


@torch.no_grad()
def strided_ppl_from_ids(model, token_ids: torch.Tensor, seqlen: int = 2048,
                         stride: int | None = None, device: str | None = None,
                         max_samples: int | None = None) -> dict:
    """在 token 序列上计算 strided perplexity。

    参数:
        model: 任意支持 `model(input_ids, labels=...)` 并返回 `.loss` 的因果语言模型。
               HF AutoModelForCausalLM 满足该接口；自定义 toy 模型需要兼容 labels。
        token_ids: 一维 token 流，通常来自 `load_token_ids` 或 eval windows reshape。
                   多维输入会被展平成一维。
        seqlen: 每次前向的最大上下文长度。评测显存主要由该值决定；不要超过模型
                最大上下文长度。
        stride: 滑窗步长。None 表示等于 seqlen，即窗口不重叠；小于 seqlen 时可以
                用重叠上下文评测长文本，但前向次数会增加。
        device: 运行设备。None 时从模型参数推断；传入字符串或 torch.device 都可。
        max_samples: 最多评测多少个 seqlen 窗口，用于限制评测耗时。None 表示尽量
                     覆盖传入 token 流。

    返回:
        包含 ppl / seqlen / stride / n_tokens 的 dict。

    实现要点:
        - 和常见 LM eval 一样，只对新进入窗口的 token 计 loss。
        - `trg_len` 记录当前窗口里真正新的 token 数，避免重复统计。
        - 这里直接读 out.loss，用模型自己的 CE 实现，和 HF 口径对齐。
    """
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
    """构造一个用于蒸馏曲线的 PPL evaluator。

    参数:
        eval_token_ids: 评测 token 流。通常是验证窗口 reshape 后的一维 tensor，
                        这样能保证 teacher、student_init、student_final 使用同一批 token。
        seqlen: PPL 前向窗口长度，传给 `strided_ppl_from_ids`。
        stride: PPL 滑窗步长。None 时使用不重叠窗口。
        max_samples: evaluator 每次最多评测多少个窗口；可用于让中途 eval 更快。

    返回:
        `evaluator(model) -> dict` 回调，返回字段包括 ppl / seqlen / stride / n_tokens。

    训练 loop 只需要一个 `evaluator(model) -> dict` 的回调。
    这里把评测 token、窗口大小和 stride 预先绑死，便于在 step 0 / eval_every /
    step_end 复用同一套逻辑。
    """
    def _eval(model) -> dict:
        return strided_ppl_from_ids(
            model, eval_token_ids, seqlen=seqlen, stride=stride,
            max_samples=max_samples)
    return _eval
