import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn

from .transformer_utils import TransformerEncoder, PatchEmbed, RMSNorm


@dataclass
class LoopForwardMetadata:

    gate_probs: List[torch.Tensor]
    exit_steps: torch.Tensor
    max_steps: int


def _compute_gate_regularizers(
    metadata: LoopForwardMetadata, disable_exit_gate
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not metadata.gate_probs or disable_exit_gate:
        zero = torch.tensor(
            0.0,
            device=(
                metadata.exit_steps.device
                if isinstance(metadata.exit_steps, torch.Tensor)
                else "cpu"
            ),
        )
        return zero, zero

    gate_stack = torch.stack(metadata.gate_probs, dim=1)  # (B, steps)
    eps = 1e-6
    entropy = -(
        gate_stack * torch.log(gate_stack + eps)
        + (1 - gate_stack) * torch.log(1 - gate_stack + eps)
    )
    neg_entropy = -entropy.mean()
    normalized_steps = metadata.exit_steps.float() / metadata.max_steps
    step_penalty = normalized_steps.mean()
    return neg_entropy, step_penalty


class LoopViT(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        loop_core_depth: int = 2,
        max_loop_steps: int = 8,
        min_loop_steps: int = 1,
        add_step_embeddings: bool = True,
        use_exit_gate: bool = True,
        gate_threshod: float = 0.6,
        swiglu: bool = False,
        loop_mode: str = "global",
        loop_schedule: Optional[List[int]] = None,
    ):
        super().__init__()

        if loop_mode not in ("global", "per_block"):
            raise ValueError(f"loop_mode must be 'global' or 'per_block', got {loop_mode!r}")
        if loop_mode == "per_block":
            if loop_schedule is None:
                raise ValueError("loop_schedule is required when loop_mode='per_block'")
            if len(loop_schedule) != loop_core_depth:
                raise ValueError(
                    f"loop_schedule length {len(loop_schedule)} != loop_core_depth {loop_core_depth}"
                )
            if any(s < 1 for s in loop_schedule):
                raise ValueError("all values in loop_schedule must be >= 1")
            if add_step_embeddings:
                raise ValueError("add_step_embeddings is not supported in loop_mode='per_block'")
            if use_exit_gate:
                raise ValueError("use_exit_gate is not supported in loop_mode='per_block'")

        self.embed_dim = embed_dim
        self.max_loop_steps = max_loop_steps
        self.min_loop_steps = min_loop_steps

        self.add_step_embeddings = add_step_embeddings
        self.use_exit_gate = use_exit_gate
        self.default_gate_threshold = gate_threshod

        self.loop_mode = loop_mode
        self.loop_schedule = loop_schedule

        if loop_mode == "global":
            self.encoder = TransformerEncoder(
                dim=embed_dim,
                depth=loop_core_depth,
                num_heads=num_heads,
                dropout=dropout,
                final_norm=False,
                swiglu=swiglu,
            )
            self.blocks = None
        else:  # per_block
            self.encoder = None
            self.blocks = nn.ModuleList([
                TransformerEncoder(
                    dim=embed_dim,
                    depth=1,
                    num_heads=num_heads,
                    dropout=dropout,
                    final_norm=False,
                    swiglu=swiglu,
                )
                for _ in range(loop_core_depth)
            ])
            self.max_loop_steps = sum(loop_schedule)

        self.patch = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch.num_patches

        self.N_total = 1 + self.num_patches

        self.cls = nn.Parameter(torch.zeros(embed_dim))

        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.N_total, embed_dim))

        self.head_norm = RMSNorm(embed_dim, eps=1e-5, affine=True)

        self.head = nn.Linear(self.embed_dim, num_classes, bias=True)

        if add_step_embeddings:
            self.step_embed = nn.Embedding(max_loop_steps, embed_dim)

        else:
            self.step_embed = None

        self.exit_gate = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )
            if use_exit_gate
            else None
        )

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)
        nn.init.zeros_(self.head.bias)

        if self.step_embed is not None:

            nn.init.trunc_normal_(self.step_embed.weight, std=0.02)

        if self.exit_gate is not None:

            for module in self.exit_gate:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,C,H,W)
        returns: (B, N_total, D)
        """

        B = images.shape[0]

        patches = self.patch(images)  # (B, num_patches, D)

        out = self.cls.view(1, 1, -1).expand(B, -1, -1)  # (B, 1, D)

        x = torch.cat([out, patches], dim=1)  # (B, 1 + num_patches, D)

        x = x + self.patch_pos_embed

        return x

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the recurrent loop and return the CLS token embedding.

        Always runs max_loop_steps (no dynamic exit) for deterministic CLIP training.

        Args:
            images: [B, C, H, W] image tensor.

        Returns:
            [B, embed_dim] CLS token features after head norm, before classification head.
        """
        running_hidden = self.image_tokens(images)

        if self.loop_mode == "global":
            for step in range(self.max_loop_steps):
                if self.step_embed is not None:
                    embed_step = min(step, self.step_embed.num_embeddings - 1)
                    running_hidden = running_hidden + self.step_embed.weight[embed_step].view(1, 1, -1)
                running_hidden = self.encoder(running_hidden)
        else:  # per_block: each block runs its scheduled number of steps sequentially
            for block, n_steps in zip(self.blocks, self.loop_schedule):
                for _ in range(n_steps):
                    running_hidden = block(running_hidden)

        final_states = self.head_norm(running_hidden)
        return final_states[:, 0, :]

    def forward(
        self,
        images,
        dynamic_exit=None,
        gate_threshold=None,
        override_max_steps=None,
        return_intermediates=False,
    ):
        batch_size = images.shape[0]
        hidden_states = self.image_tokens(images)

        running_hidden = hidden_states
        device = images.device

        exit_steps = torch.full(
            (batch_size,), self.max_loop_steps, dtype=torch.long, device=device
        )

        # per_block mode: sequential execution, no dynamic exit or gate
        if self.loop_mode == "per_block":
            for block, n_steps in zip(self.blocks, self.loop_schedule):
                for _ in range(n_steps):
                    running_hidden = block(running_hidden)
            final_states = self.head_norm(running_hidden)
            logits = self.head(final_states[:, 0, :])
            metadata = LoopForwardMetadata(
                gate_probs=[], exit_steps=exit_steps, max_steps=self.max_loop_steps
            )
            if return_intermediates:
                return logits, metadata, []
            return logits, metadata

        # global mode: original loop with optional dynamic exit and step embeddings
        use_dynamic_exit = bool(dynamic_exit) and self.use_exit_gate
        threshold = (
            gate_threshold
            if gate_threshold is not None
            else self.default_gate_threshold
        )
        current_max = (
            override_max_steps
            if override_max_steps is not None
            else self.max_loop_steps
        )

        exit_steps = torch.full(
            (batch_size,), current_max, dtype=torch.long, device=device
        )

        if use_dynamic_exit:
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            cached_final = torch.zeros_like(running_hidden)

        else:
            cached_final = None

        gate_probs: List[torch.Tensor] = []
        intermediate_logits_list: List[torch.Tensor] = []

        for step in range(current_max):

            if self.step_embed is not None:
                embed_step = min(step, self.step_embed.num_embeddings - 1)
                running_hidden = running_hidden + self.step_embed.weight[
                    embed_step
                ].view(1, 1, -1)

            running_hidden = self.encoder(running_hidden)

            if return_intermediates:

                final_states = self.head_norm(running_hidden)
                images_states = final_states[:, 0, :]
                logits = self.head(images_states)
                intermediate_logits_list.append(logits)

            if self.use_exit_gate:

                gate_logit = self.exit_gate(running_hidden[:, 0, :]).squeeze(-1)
                gate_prob = torch.sigmoid(gate_logit)
                gate_probs.append(gate_prob)

            else:
                gate_prob = None

            if use_dynamic_exit and gate_prob is not None:

                eligible = step + 1 >= self.min_loop_steps
                exit_now = (gate_prob >= threshold) & eligible

                if exit_now.any():

                    cached_final[exit_now] = running_hidden[exit_now]

                    exit_steps[exit_now] = step

                    finished_mask = finished_mask | exit_now

                running_hidden = torch.where(
                    finished_mask.view(batch_size, 1, 1),
                    cached_final,
                    running_hidden,
                )

                if finished_mask.all():
                    break

        final_states = self.head_norm(running_hidden)
        images_states = final_states[:, 0, :]
        logits = self.head(images_states)

        metadata = LoopForwardMetadata(
            gate_probs=gate_probs, exit_steps=exit_steps, max_steps=current_max
        )

        if return_intermediates:
            return logits, metadata, intermediate_logits_list

        return logits, metadata
