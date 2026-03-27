"""Interactive Contrastive Learning (ICL) loss (Equations 14-16 in paper).

ICL aligns student embeddings with teacher embeddings *across* modalities:
student image features are matched against teacher text features and vice versa.

Source: src/open_clip/loss.py lines 244, 298-315.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import CLIPDistillationLoss, KDFeatures


class ICLLoss(CLIPDistillationLoss):
    """Interactive Contrastive Learning loss.

    Paper: Section 3.4, Equations 14-16.
    Source: src/open_clip/loss.py lines 244 (cross_logit_scale init),
            298-315 (icl_loss computation).

    Contains a learnable cross_logit_scale parameter included in the
    optimizer via the CompositeLoss parameter group.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable temperature for cross-modal similarities
        self.cross_logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1 / 0.07)
        )

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute ICL loss.

        Args:
            features: KDFeatures with s_img_proj, s_txt_proj, t_img, t_txt, labels.

        Returns:
            Scalar ICL loss.
        """
        scale = self.cross_logit_scale.exp()

        logits_s_img_to_t_txt = scale * features.s_img_proj @ features.t_txt.T
        logits_s_txt_to_t_img = scale * features.s_txt_proj @ features.t_img.T

        icl_loss = (
            F.cross_entropy(logits_s_img_to_t_txt, features.labels)
            + F.cross_entropy(logits_s_txt_to_t_img, features.labels)
        ) / 2
        return icl_loss
