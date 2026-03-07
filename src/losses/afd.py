"""Augmented Feature Distillation (AFD) loss (Equation 17 in paper).

AFD concatenates student and teacher embeddings, projects the fusion to
student dimension, then applies the standard CLIP InfoNCE loss on the
fused representations. This acts as a teacher-guided augmentation.

Source: src/open_clip/loss.py lines 245-246, 339-352.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import CLIPDistillationLoss, KDFeatures


class AFDLoss(CLIPDistillationLoss):
    """Augmented Feature Distillation loss.

    Paper: Section 3.6, Equation 17.
    Source: src/open_clip/loss.py lines 245-246 (fusion proj init),
            339-352 (afd_loss computation).

    Learnable parameters:
        visual_fusion_proj: Linear(s_dim + t_dim -> out_dim).
        text_fusion_proj:   Linear(s_dim + t_dim -> out_dim).
        fusion_logit_scale: Learnable temperature for fused features.

    Args:
        s_embed_dim: Student embedding dimension.
        t_embed_dim: Teacher embedding dimension.
        out_dim: Output dimension for fused projections (typically s_embed_dim).
    """

    def __init__(self, s_embed_dim: int, t_embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.visual_fusion_proj = nn.Linear(s_embed_dim + t_embed_dim, out_dim)
        self.text_fusion_proj = nn.Linear(s_embed_dim + t_embed_dim, out_dim)
        self.fusion_logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute AFD loss.

        Args:
            features: KDFeatures with s_img, s_txt (pre-projection, D_s),
                      t_img, t_txt (D_t), labels.

        Returns:
            Scalar AFD loss.
        """
        # Concatenate student (pre-projection) and teacher features
        img_fusion = torch.cat([features.s_img, features.t_img], dim=1)
        txt_fusion = torch.cat([features.s_txt, features.t_txt], dim=1)

        img_fusion = F.normalize(self.visual_fusion_proj(img_fusion), dim=1)
        txt_fusion = F.normalize(self.text_fusion_proj(txt_fusion), dim=1)

        scale = self.fusion_logit_scale.exp()
        logits_img = scale * img_fusion @ txt_fusion.T
        logits_txt = logits_img.T

        afd_loss = (
            F.cross_entropy(logits_img, features.labels)
            + F.cross_entropy(logits_txt, features.labels)
        ) / 2
        return afd_loss
