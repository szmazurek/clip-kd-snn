"""Recurrent text encoder mirroring the LoopViT design.

LoopText applies a small shared transformer core (loop_core_depth blocks)
repeatedly for max_loop_steps iterations over token embeddings, then extracts
the EOS token as the sequence representation — matching the standard CLIP
TextTransformer interface.

Key design choices that mirror LoopViT:
  - Shared block(s) applied in a loop (weight-tied recurrence)
  - Optional per-step embeddings added before each iteration
  - Causal (lower-triangular) self-attention, identical to open_clip TextTransformer
  - EOS token extraction at the position of the highest token ID (standard CLIP)
  - RMSNorm as final normalisation (consistent with LoopViT)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..visual_encoders.transformer_utils import RMSNorm, TransformerBlock


class LoopText(nn.Module):
    """Recurrent CLIP-compatible text encoder.

    Satisfies the interface expected by LoopViTCLIPModel:
        encode_text(text, normalize=False) -> Tensor[B, embed_dim]

    Args:
        vocab_size: Vocabulary size (default: 49408, standard CLIP SimpleTokenizer).
        context_length: Maximum sequence length (default: 77).
        embed_dim: Output CLIP embedding dimension (matches text_embed_dim in the
                   CLIP wrapper, e.g. 512 for ViT-B/16 text space).
        width: Internal transformer hidden dimension.
        num_heads: Number of attention heads (width must be divisible by num_heads).
        mlp_ratio: MLP expansion ratio inside each TransformerBlock.
        dropout: Dropout applied inside TransformerBlock FFN and attention projection.
        loop_core_depth: Number of TransformerBlocks in the shared core. All blocks
                         are weight-tied across loop steps.
        max_loop_steps: Number of times the shared core is applied recurrently.
        add_step_embeddings: If True, a learned per-step embedding is added to every
                             token before each loop iteration (same option as LoopViT).
        swiglu: Use SwiGLU activation in the MLP (False → GELU, matching LoopViT default).
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        embed_dim: int = 512,
        width: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        loop_core_depth: int = 1,
        max_loop_steps: int = 12,
        add_step_embeddings: bool = False,
        swiglu: bool = False,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.max_loop_steps = max_loop_steps
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(context_length, width)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=width,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    swiglu=swiglu,
                )
                for _ in range(loop_core_depth)
            ]
        )

        self.step_embed = (
            nn.Embedding(max_loop_steps, width) if add_step_embeddings else None
        )

        self.ln_final = RMSNorm(width)
        self.text_projection = nn.Linear(width, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.01)
        nn.init.normal_(self.text_projection.weight, std=self.embed_dim ** -0.5)
        if self.step_embed is not None:
            nn.init.trunc_normal_(self.step_embed.weight, std=0.02)

    def encode_text(self, text: Tensor, normalize: bool = False) -> Tensor:
        """Encode a batch of token sequences to CLIP embeddings.

        Args:
            text: [B, L] integer token-id tensor (L <= context_length).
            normalize: If True, L2-normalise the output embeddings.

        Returns:
            [B, embed_dim] text feature tensor.
        """
        B, L = text.shape
        positions = torch.arange(L, device=text.device)

        x = self.token_embed(text) + self.pos_embed(positions)  # [B, L, width]

        for step in range(self.max_loop_steps):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step].view(1, 1, -1)
            for block in self.blocks:
                x = block(x, is_causal=True)

        x = self.ln_final(x)  # [B, L, width]

        # Extract EOS token: the position of the highest token ID in each sequence.
        # This matches open_clip's TextTransformer convention exactly.
        eos_positions = text.argmax(dim=-1)  # [B]
        x = x[torch.arange(B, device=x.device), eos_positions]  # [B, width]

        x = self.text_projection(x)  # [B, embed_dim]

        return F.normalize(x, dim=-1) if normalize else x

    def encode_text_with_hidden_states(
        self, text: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode text and return EOS embedding plus per-block-execution hidden states.

        Returns:
            eos: [B, embed_dim] — same as encode_text(normalize=False).
            hidden_states: list of [B, L, width] tensors, one per block execution
                (len = max_loop_steps * loop_core_depth).
        """
        B, L = text.shape
        positions = torch.arange(L, device=text.device)
        x = self.token_embed(text) + self.pos_embed(positions)

        hidden_states: list[Tensor] = []
        for step in range(self.max_loop_steps):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step].view(1, 1, -1)
            for block in self.blocks:
                x = block(x, is_causal=True)
                hidden_states.append(x)

        x = self.ln_final(x)
        eos_positions = text.argmax(dim=-1)
        eos = self.text_projection(x[torch.arange(B, device=x.device), eos_positions])
        return eos, hidden_states
