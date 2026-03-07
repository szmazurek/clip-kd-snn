"""Feature Distillation (FD) loss (Equation 11 in paper).

Minimises MSE between L2-normalised student features (after projection to
teacher dimension) and L2-normalised teacher features.

Note: Projection heads are NOT part of this module. They live in
CLIPKDModule and are applied before building KDFeatures so that all losses
share the same projected embeddings without re-computing the projection.

Source: src/open_clip/loss.py lines 289-296.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import CLIPDistillationLoss, KDFeatures


class FDLoss(CLIPDistillationLoss):
    """Feature Distillation loss.

    Paper: Section 3.3, Equation 11.
    Source: src/open_clip/loss.py lines 289-296.

    Operates on s_img_proj and s_txt_proj (already projected + normalised).
    """

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute FD loss.

        Args:
            features: KDFeatures with s_img_proj, s_txt_proj, t_img, t_txt.

        Returns:
            Scalar FD loss (sum of image and text MSE).
        """
        fd_loss = (
            F.mse_loss(features.s_img_proj, features.t_img)
            + F.mse_loss(features.s_txt_proj, features.t_txt)
        )
        return fd_loss
