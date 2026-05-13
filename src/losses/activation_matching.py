"""Activation Matching loss for LoopViT distillation.

Matches per-iteration CLS activations of LoopViT (student) to per-block CLS
activations of ViT-B/16 (teacher), using mean MSE across all steps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CLIPDistillationLoss, KDFeatures


class ActivationMatchingLoss(CLIPDistillationLoss):
    """MSE between student per-iteration and teacher per-block CLS activations.

    For the default LoopViT config (embed_dim=768) vs ViT-B/16 (width=768),
    both have the same dimension and no projection is needed.

    Args:
        student_dim: CLS hidden dimension of the student (pre head_norm).
        teacher_dim: CLS hidden dimension of the teacher (post resblock, pre ln_post).
    """

    def __init__(self, student_dim: int = 768, teacher_dim: int = 768) -> None:
        super().__init__()
        self.proj: nn.Module | None = (
            nn.Linear(student_dim, teacher_dim, bias=False)
            if student_dim != teacher_dim
            else None
        )

    def forward(self, features: KDFeatures) -> torch.Tensor:
        s_acts = features.s_intermediates
        t_acts = features.t_intermediates
        assert s_acts is not None and t_acts is not None, (
            "ActivationMatchingLoss requires s_intermediates and t_intermediates in KDFeatures"
        )
        assert len(s_acts) == len(t_acts), (
            f"Step count mismatch: student has {len(s_acts)} steps, teacher has {len(t_acts)} blocks"
        )
        total = torch.tensor(0.0, device=s_acts[0].device)
        for s, t in zip(s_acts, t_acts):
            if self.proj is not None:
                s = self.proj(s)
            total = total + F.mse_loss(s, t.detach())
        return total / len(s_acts)
