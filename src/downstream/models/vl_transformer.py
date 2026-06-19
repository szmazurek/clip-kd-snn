"""Vision-language grounding transformer encoder.

6-layer pre-norm transformer encoder that fuses visual and text tokens.
Port of HiVG/models/vl_transformer.py (originally from DETR).

Input token sequence (all in d_model=512 space):
    [REG] [CLS_vis] patch_tokens(196) text_tokens(77)   → 275 tokens

The [REG] token output is read by the MLP box regression head.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def _with_pos(self, t: Tensor, pos: Tensor | None) -> Tensor:
        return t if pos is None else t + pos

    def forward_pre(self, src: Tensor, mask: Tensor | None, pos: Tensor | None) -> Tensor:
        src2 = self.norm1(src)
        q = k = self._with_pos(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        return src + self.dropout2(src2)

    def forward_post(self, src: Tensor, mask: Tensor | None, pos: Tensor | None) -> Tensor:
        q = k = self._with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

    def forward(self, src: Tensor, mask: Tensor | None = None,
                pos: Tensor | None = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, mask, pos)
        return self.forward_post(src, mask, pos)


class VisionLanguageEncoder(nn.Module):
    """6-layer transformer encoder for vision-language fusion.

    Args:
        d_model: Token embedding dimension (512).
        nhead: Attention heads (8).
        num_layers: Encoder depth (6).
        dim_feedforward: FFN hidden size (2048).
        dropout: Dropout rate (0.1).
        normalize_before: Pre-norm if True (recommended).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ) -> None:
        super().__init__()
        layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                         dropout, normalize_before)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            src: [L, B, d_model]
            key_padding_mask: [B, L] bool, True = ignore token.
            pos: [L, B, d_model] positional embeddings.

        Returns:
            [L, B, d_model]
        """
        x = src
        for layer in self.layers:
            x = layer(x, mask=key_padding_mask, pos=pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class MLP(nn.Module):
    """3-layer ReLU MLP used as the box regression head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x
