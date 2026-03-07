"""CompositeLoss: combines CLIP task loss with weighted distillation losses.

L_total = alpha_task * L_CLIP + sum(alpha_i * L_KD_i)

Loss call order is fixed to ensure ICLLoss populates cross-modal logits in
KDFeatures before CrossKDLoss reads them:
    task -> ckd -> icl -> cross_kd -> fd -> gd -> afd
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import CLIPDistillationLoss, KDFeatures


# Fixed call order — ICL must precede CrossKD
_LOSS_ORDER = ["task", "ckd", "icl", "cross_kd", "fd", "gd", "afd"]


class CompositeLoss(nn.Module):
    """Config-driven composite loss combining CLIP and KD objectives.

    Args:
        losses: Mapping from loss name to instantiated loss module.
                Names should come from _LOSS_ORDER above.
        weights: Per-loss lambda weights. If a name appears in losses but
                 not in weights, weight defaults to 1.0.
    """

    def __init__(
        self,
        losses: dict[str, CLIPDistillationLoss],
        weights: dict[str, float],
    ) -> None:
        super().__init__()
        # Register as ModuleDict so parameters are tracked by PyTorch
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(
        self, features: KDFeatures
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss and per-component loss dict.

        Args:
            features: KDFeatures (may be mutated by ICLLoss to add
                      cross-modal logits for CrossKDLoss).

        Returns:
            Tuple of:
                total_loss: Weighted sum of all active losses (scalar).
                loss_dict: {name: scalar_tensor} for logging.
        """
        total = torch.tensor(0.0, device=features.labels.device)
        loss_dict: dict[str, torch.Tensor] = {}

        for name in _LOSS_ORDER:
            if name not in self.losses:
                continue
            w = self.weights.get(name, 1.0)
            if w == 0.0:
                continue
            component = self.losses[name](features)
            weighted = w * component
            total = total + weighted
            loss_dict[name] = component.detach()

        loss_dict["total"] = total.detach()
        return total, loss_dict
