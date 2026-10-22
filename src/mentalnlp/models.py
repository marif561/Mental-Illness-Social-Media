"""Neural models: Transformer encoder with late fusion, plus CNN / LSTM / BiLSTM baselines.

Every encoder maps token ids ``(batch, seq)`` to one pooled vector ``(batch, out_dim)``.
:class:`FusionClassifier` runs one encoder per input text (title, post) and merges the
pooled vectors with one of five late-fusion strategies before a softmax classifier
(paper Fig. 1 and Eq. 1).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

FUSIONS = ("concat", "avg", "wavg", "max", "min")


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class Embedding(nn.Module):
    """Token embeddings, optionally initialised from a pretrained matrix (e.g. RoBERTa)."""

    def __init__(self, vocab_size: int, dim: int, pretrained: Optional[torch.Tensor] = None,
                 freeze: bool = False, pad_id: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        if pretrained is not None:
            self.emb.weight.data.copy_(pretrained)
        self.emb.weight.requires_grad = not freeze

    def forward(self, ids):
        return self.emb(ids)


class TransformerTextEncoder(nn.Module):
    """Encoder part of the vanilla Transformer (Vaswani et al., 2017) with learned positions."""

    def __init__(self, embedding: Embedding, max_len: int, d_model: int, layers: int, heads: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        emb_dim = embedding.emb.embedding_dim
        self.proj = nn.Identity() if emb_dim == d_model else nn.Linear(emb_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d_model

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.drop(self.proj(self.embedding(ids)) + self.pos(pos))
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        return masked_mean(h, mask)  # average of the last encoder layer


class RNNTextEncoder(nn.Module):
    def __init__(self, embedding: Embedding, hidden: int, layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.emb.embedding_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, ids, mask):
        h, _ = self.rnn(self.drop(self.embedding(ids)))
        return masked_mean(h, mask)


class CNNTextEncoder(nn.Module):
    def __init__(self, embedding: Embedding, kernels: List[int], filters: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        dim = embedding.emb.embedding_dim
        self.convs = nn.ModuleList([nn.Conv1d(dim, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.out_dim = filters * len(kernels)

    def forward(self, ids, mask):
        x = self.drop(self.embedding(ids)).transpose(1, 2)
        neg = (~mask.bool()).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[:, :, : ids.size(1)]
            pooled.append(h.masked_fill(neg, float("-inf")).max(-1).values)
        return torch.cat(pooled, -1)


class FusionClassifier(nn.Module):
    """One encoder per input text, late fusion of pooled vectors, then a linear softmax layer."""

    def __init__(self, encoders: Dict[str, nn.Module], n_labels: int, fusion: str = "concat",
                 dropout: float = 0.2):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.fields = list(encoders)
        self.encoders = nn.ModuleDict(encoders)
        dims = {e.out_dim for e in encoders.values()}
        if len(self.fields) > 1 and fusion != "concat" and len(dims) != 1:
            raise ValueError("element-wise fusion needs encoders with equal output size")
        self.fusion = fusion if len(self.fields) > 1 else "single"
        out = sum(e.out_dim for e in encoders.values()) if self.fusion == "concat" else next(iter(dims))
        if self.fusion == "wavg":
            self.fusion_logits = nn.Parameter(torch.zeros(len(self.fields)))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(out, n_labels)

    def fuse(self, hs: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion in ("single",):
            return hs[0]
        if self.fusion == "concat":
            return torch.cat(hs, -1)
        stack = torch.stack(hs, 0)
        if self.fusion == "avg":
            return stack.mean(0)
        if self.fusion == "wavg":
            w = torch.softmax(self.fusion_logits, 0).view(-1, 1, 1)
            return (w * stack).sum(0)
        if self.fusion == "max":
            return stack.max(0).values
        return stack.min(0).values

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        hs = [self.encoders[f](batch[f]["ids"], batch[f]["mask"]) for f in self.fields]
        return self.classifier(self.drop(self.fuse(hs)))


def build_encoder(kind: str, embedding: Embedding, cfg: dict, max_len: int) -> nn.Module:
    if kind == "transformer":
        return TransformerTextEncoder(embedding, max_len, cfg["d_model"], cfg["layers"], cfg["heads"],
                                      cfg["ff_dim"], cfg["dropout"])
    if kind in ("lstm", "bilstm"):
        return RNNTextEncoder(embedding, cfg["hidden"], cfg.get("rnn_layers", 1), cfg["dropout"],
                              bidirectional=kind == "bilstm")
    if kind == "cnn":
        return CNNTextEncoder(embedding, cfg["kernels"], cfg["filters"], cfg["dropout"])
    raise ValueError(kind)
