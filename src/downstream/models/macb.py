"""Multi-layer Adaptive Cross-modal Bridge (MACB).

One MACB instance per injection slot (three total, at block-executions 4/8/12).
Faithful port of HiVG's CLIPEncoderLayer_with_Crossmodal_Bridge, decoupled from
the visual backbone so it can be used with the block-level execution loop.

Forward signature:
    visual   : [B, N_vis, visual_dim]   — visual token sequence at injection point
    text_list: list of K tensors [B, text_seq_len, text_dim]  — one per text layer

Returns the cross-modal residual delta [B, N_vis, visual_dim] to add to visual.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MACB(nn.Module):
    """Multi-layer Adaptive Cross-modal Bridge for one injection slot.

    Args:
        visual_dim: Dimensionality of visual tokens (e.g. 768).
        text_dim:   Dimensionality of each text hidden state (e.g. 512).
        n_text_layers: Number of text layers / loop iterations to aggregate.
        text_seq_len: Token sequence length for text (e.g. 77).
        num_heads: Number of attention heads for cross-attention.
        dropout: Dropout on cross-attention output projection.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        n_text_layers: int,
        text_seq_len: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_text_layers = n_text_layers

        # Per-layer adaptive semantic weights: Embedding(text_seq_len, text_dim)
        # acts as a learnable multiplicative bias over the text sequence.
        self.adaptive_weights = nn.ModuleList(
            [nn.Embedding(text_seq_len, text_dim) for _ in range(n_text_layers)]
        )

        # Project concatenated weighted text features → visual_dim
        self.gate = nn.Linear(text_dim * n_text_layers, visual_dim, bias=True)

        # Pre-norm before cross-attention
        self.cross_norm = nn.LayerNorm(visual_dim)

        # Cross-attention: visual queries, projected-text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Small FFN after cross-attention (same hidden dim as visual)
        hidden = visual_dim * 4
        self.cross_mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, visual_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for emb in self.adaptive_weights:
            nn.init.normal_(emb.weight, std=0.02)
        for m in self.cross_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, visual: Tensor, text_list: list[Tensor]) -> Tensor:
        """Compute cross-modal residual.

        Args:
            visual:    [B, N_vis, visual_dim]
            text_list: K × [B, text_seq_len, text_dim]
                       where K == self.n_text_layers.

        Returns:
            [B, N_vis, visual_dim] residual to add to the visual token sequence.
        """
        assert len(text_list) == self.n_text_layers, (
            f"Expected {self.n_text_layers} text layers, got {len(text_list)}"
        )

        # --- Adaptive semantic weighting ---
        weighted = []
        for i, (emb, txt) in enumerate(zip(self.adaptive_weights, text_list)):
            # emb.weight: [text_seq_len, text_dim], txt: [B, text_seq_len, text_dim]
            w = emb.weight.unsqueeze(0)          # [1, text_seq_len, text_dim]
            weighted.append(txt * w + txt)       # element-wise scale + residual

        # cat along feature dim → [B, text_seq_len, text_dim * K]
        text_agg = torch.cat(weighted, dim=-1)
        # project to visual_dim → [B, text_seq_len, visual_dim]
        text_proj = self.gate(text_agg)

        # --- Cross-attention (visual queries, text kv) ---
        vis_normed = self.cross_norm(visual)
        attn_out, _ = self.cross_attn(
            query=vis_normed,
            key=text_proj,
            value=text_proj,
        )

        # --- FFN ---
        delta = self.cross_mlp(attn_out)
        return delta
