"""伪量化 Embedding 层。

与 FakeQuantLinear 对称:权重在构造时做一次 RTN 量化-反量化并缓存,
forward 走标准 F.embedding。用于量化 embed_tokens 以突破压缩地板。

设计说明:
- Embedding 无激活统计(查表操作),故固定用 RTN,不做 GPTQ/AWQ。
  沿 hidden_dim(即 embedding_dim)分组,与 Linear 的 in_features 方向一致。
- 绑定(tied)权重场景:apply.py 把同一 w_dq 张量传给 FakeQuantEmbedding
  和对应的 FakeQuantLinear(lm_head),两者共享 buffer 对象,压缩统计
  通过 id() 去重只计一次,不会重复计算体积。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import fake_quant_groupwise


class FakeQuantEmbedding(nn.Module):
    """把 nn.Embedding 替换为"权重已被量化误差污染"的等价层。

    属性:
        weight:      反量化后的权重 [vocab_size, embedding_dim]。
        vocab_size:  词表大小。
        embedding_dim: 嵌入维度。
        padding_idx: 来自原 Embedding 的 padding 索引(None 则无)。
        n_bits / group_size / symmetric: 量化元数据,供 compression_report 读取。
    """

    def __init__(self, emb: nn.Embedding, n_bits: int, group_size: int,
                 symmetric: bool, w_dq: torch.Tensor = None):
        """
        参数:
            emb:        原始 nn.Embedding。
            n_bits:     量化位宽。
            group_size: 分组大小(沿 embedding_dim 分组);-1 = per-channel。
            symmetric:  是否对称量化。
            w_dq:       可选——已计算好的反量化权重(供绑定场景传入,
                        避免 embed/lm_head 各自独立量化数值不一致);
                        为 None 时在内部用 RTN 计算。
        """
        super().__init__()
        self.vocab_size = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        self.n_bits = n_bits
        self.group_size = (emb.embedding_dim if group_size in (-1, None)
                           else group_size)
        self.symmetric = symmetric

        with torch.no_grad():
            if w_dq is None:
                w_dq = fake_quant_groupwise(
                    emb.weight.data, n_bits, group_size, symmetric)
        self.register_buffer("weight", w_dq.to(emb.weight.dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """标准 embedding 查表;权重已是反量化后的值。"""
        return F.embedding(input_ids, self.weight,
                           padding_idx=self.padding_idx)

    def extra_repr(self):
        return (f"vocab={self.vocab_size}, dim={self.embedding_dim}, "
                f"bits={self.n_bits}, group={self.group_size}, "
                f"sym={self.symmetric}")
