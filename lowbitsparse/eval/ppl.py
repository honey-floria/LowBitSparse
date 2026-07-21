"""WikiText-2 困惑度(PPL)评测。

采用社区标准的 strided(滑窗)算法:把整个测试集拼成一条长序列后按窗口滑动,
重叠部分只对"新出现的 token"计损失,避免不重叠切分导致的 PPL 高估。
PPL 是本项目衡量压缩(量化/稀疏)对语言建模能力损伤的核心精度指标。
"""
import torch
from tqdm import tqdm   # 进度条,长序列评测耗时可见


@torch.no_grad()        # 纯评测:禁用梯度,省显存、提速
def eval_wikitext2_ppl(
    model,                       # 已 eval 的因果 LM
    tokenizer,                   # 对应分词器
    seqlen: int = 2048,          # 每个窗口喂给模型的 token 数(上下文长度)
    stride: int = 2048,          # 窗口每次前移的步长
    split: str = "test",         # 数据划分:test/validation
    device: str = None,          # 计算设备,None 时取模型所在设备
    max_samples: int = None,     # 限制评测 token 量做冒烟测试,None 为全量
    dataset_id: str = "Salesforce/wikitext",  # 带命名空间的官方镜像,避免无命名空间旧 id 报错
    dataset_config: str = "wikitext-2-raw-v1",  # 数据集配置(raw 版,PPL 口径与论文一致)
) -> dict:
    """计算 WikiText-2-raw 的困惑度。

    stride < seqlen:窗口重叠,只对新 token 计损,PPL 更准(接近论文口径)。
    stride == seqlen:不重叠切分,最快,但会因缺上下文轻微高估 PPL。

    返回 dict:{ppl, seqlen, stride, n_tokens}。
    """
    from datasets import load_dataset   # 延迟导入,保持顶层轻量

    # 未显式指定设备时,跟随模型第一个参数所在设备(cuda/cpu)
    if device is None:
        device = next(model.parameters()).device

    # 加载 wikitext-2 raw 版(未做 token 归一化,PPL 口径与论文一致)
    # 用 Salesforce/wikitext 而非裸 "wikitext":新版 huggingface_hub 要求
    # 仓库 id 必须为 namespace/name 格式,裸 id 会报 Invalid HF URI。
    data = load_dataset(dataset_id, dataset_config, split=split)
    # 用双换行拼接所有行,还原成一整篇连续文本
    text = "\n\n".join(data["text"])
    # 一次性编码为 token id,形状 [1, 总token数]
    enc = tokenizer(text, return_tensors="pt").input_ids
    n_tokens = enc.size(1)                    # 可评测的总 token 数
    if max_samples is not None:               # 冒烟模式:仅取前若干个窗口的量
        n_tokens = min(n_tokens, max_samples * seqlen)

    nll_sum = 0.0   # 累计负对数似然(nll)之和,分子
    n_counted = 0   # 累计参与计损的 token 数,分母
    prev_end = 0    # 上一窗口的结束位置,用于算本窗口"新 token"数
    # 以 stride 为步长滑动窗口遍历整条序列
    for begin in tqdm(range(0, n_tokens, stride), desc="ppl"):
        end = min(begin + seqlen, n_tokens)   # 本窗口右边界(末窗可能不足 seqlen)
        trg_len = end - prev_end              # 本窗口真正需要计损的新 token 数
        input_ids = enc[:, begin:end].to(device)   # 取窗口 token 送设备
        target_ids = input_ids.clone()             # 标签 = 输入右移(HF 内部处理)
        # 把"非新增"部分标签置 -100,交叉熵会忽略,实现"只对新 token 计损"
        target_ids[:, :-trg_len] = -100

        # 前向:传 labels 时 HF 自动做移位并返回"平均"交叉熵 loss
        out = model(input_ids, labels=target_ids)
        # HF 的 loss 是对有效位置取的均值;有效计损位置数 = 新 token 数 - 1
        # (预测第 t 个 token 需第 t-1 个做输入,首位无标签)
        valid = trg_len - 1 if trg_len > 1 else 1
        nll_sum += out.loss.float().item() * valid   # 还原成 nll 之和(乘回分母)
        n_counted += valid                            # 累加有效 token 数
        prev_end = end                                # 更新上一窗口末位
        if end >= n_tokens:                           # 已覆盖到序列末尾,结束
            break

    # 困惑度 = exp(平均 nll);nll 之和 / token 总数 得平均,再取指数
    ppl = float(torch.exp(torch.tensor(nll_sum / max(n_counted, 1))))
    return {
        "ppl": round(ppl, 4),         # 困惑度,越低越好
        "seqlen": seqlen,             # 记录评测配置,便于复现
        "stride": stride,
        "n_tokens": int(n_tokens),    # 实际参与评测的 token 规模
    }
