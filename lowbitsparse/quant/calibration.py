"""校准数据与逐层激活统计收集(GPTQ / AWQ 用)。

GPTQ 需要每层输入激活的二阶统计 H≈Σ x xᵀ;AWQ 需要每条输入通道的一阶幅度
a=mean(|x|)。二者都只需在原 FP16 模型上做一次前向,用 forward hook 增量累积,
无需缓存全部激活,显存可控。

设计取舍:逐层独立量化(各层用"原模型"的激活统计),不做 GPTQ 原论文的
逐层顺序传播(下游看到上游量化后输出)。对小模型/教学用途,这一近似实现简单、
内存友好,已能体现 GPTQ/AWQ 相对 RTN 的收益;真实收益以参考库佐证。
"""
import torch


def get_calib_inputs(tokenizer, n_samples: int = 128, seqlen: int = 512,
                     dataset_id: str = "Salesforce/wikitext",
                     dataset_config: str = "wikitext-2-raw-v1"):
    """从 WikiText-2 采样校准输入,返回 [n_samples, seqlen] 的 token id。

    把训练集拼成长序列后切成 n_samples 个不重叠窗口;校准只需覆盖典型分布,
    故用 train split、较短 seqlen 控制耗时。
    """
    from datasets import load_dataset

    data = load_dataset(dataset_id, dataset_config, split="train")
    text = "\n\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]   # [total]
    need = n_samples * seqlen
    if enc.numel() < need:                     # 语料不足时缩减样本数
        n_samples = enc.numel() // seqlen
        need = n_samples * seqlen
    enc = enc[:need].reshape(n_samples, seqlen)
    return enc


class _StatHook:
    """挂在单个 Linear 上,增量累积其输入的 GPTQ Hessian 与 AWQ 激活幅度。"""

    def __init__(self, in_features: int, device):
        # H 累积 Σ xxᵀ(GPTQ);act_sum/n 累积 Σ|x| 与计数(AWQ)
        self.H = torch.zeros(in_features, in_features, device=device)
        self.act_sum = torch.zeros(in_features, device=device)
        self.n = 0

    def __call__(self, module, inp, out):
        x = inp[0]                              # [..., in_features]
        x = x.reshape(-1, x.shape[-1]).float()  # 展平成 [tokens, in_features]
        self.H += x.t() @ x                     # 累加二阶统计
        self.act_sum += x.abs().sum(dim=0)      # 累加一阶幅度
        self.n += x.shape[0]


def collect_calib_stats(model, calib_ids: torch.Tensor, target_names: list,
                        batch_size: int = 8) -> dict:
    """对 model 挂 hook,跑校准输入,返回每层的 Hessian 和激活幅度。

    参数:
        model:        原 FP16 模型(eval 模式,不改其权重)。
        calib_ids:    [n_samples, seqlen] 校准 token id 张量。
        target_names: 需要收集统计的子模块全名列表(对应 named_modules 里的 key)。
        batch_size:   一次前向跑几条样本,节省显存。
    返回:
        dict:{子模块全名 → {"H": ..., "act_scales": ...}}
            H [in_f, in_f]:  Hessian 矩阵(GPTQ 用)。
            act_scales [in_f]: 每通道激活幅度均值(AWQ 用)。
    """
    device = next(model.parameters()).device
    name_to_module = {n: m for n, m in model.named_modules()
                      if n in target_names}
    hooks_map = {n: _StatHook(m.in_features, device)
                 for n, m in name_to_module.items()}
    handles = [m.register_forward_hook(hooks_map[n])
               for n, m in name_to_module.items()]
    try:
        with torch.no_grad():
            for i in range(0, calib_ids.shape[0], batch_size):
                batch = calib_ids[i:i + batch_size].to(device)
                model(batch)                    # 纯前向,无需输出
    finally:
        for h in handles:
            h.remove()                          # 无论异常都摘掉 hook

    return {
        n: {
            "H": hooks_map[n].H,               # [in_f, in_f] float32
            "act_scales": (hooks_map[n].act_sum / max(hooks_map[n].n, 1)).clamp(min=1e-6),
        }
        for n in target_names
    }
