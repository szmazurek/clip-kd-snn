"""Interactive Contrastive Learning (ICL) loss (Equations 14-16 in paper).

ICL aligns student embeddings with teacher embeddings *across* modalities:
student image features are matched against teacher text features and vice versa.

This module also populates KDFeatures.cross_logits_img2txt and
cross_logits_txt2img so that CrossKDLoss can reuse the same logits without
recomputing them. ICLLoss must run before CrossKDLoss in CompositeLoss.

Source: src/open_clip/loss.py lines 244, 298-315.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import CLIPDistillationLoss, KDFeatures


class ICLLoss(CLIPDistillationLoss):
    """Interactive Contrastive Learning loss.

    Paper: Section 3.4, Equations 14-16.
    Source: src/open_clip/loss.py lines 244 (cross_logit_scale init),
            298-315 (icl_loss computation).

    Contains a learnable cross_logit_scale parameter which is also used by
    CrossKDLoss. Its parameters are included in the optimizer via the
    CompositeLoss parameter group.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable temperature for cross-modal similarities
        self.cross_logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute ICL loss and populate cross-modal logits in features.

        Args:
            features: KDFeatures. Mutates cross_logits_img2txt and
                      cross_logits_txt2img in place for downstream CrossKDLoss.

        Returns:
            Scalar ICL loss.
        """
        scale = self.cross_logit_scale.exp()

        logits_s_img_to_t_txt = scale * features.s_img_proj @ features.t_txt.T
        logits_s_txt_to_t_img = scale * features.s_txt_proj @ features.t_img.T

        # Store for CrossKDLoss reuse (avoids recomputation)
        features.cross_logits_img2txt = logits_s_img_to_t_txt
        features.cross_logits_txt2img = logits_s_txt_to_t_img

        icl_loss = (
            F.cross_entropy(logits_s_img_to_t_txt, features.labels)
            + F.cross_entropy(logits_s_txt_to_t_img, features.labels)
        ) / 2
        return icl_loss


class CrossKDLoss(CLIPDistillationLoss):
    """Cross-KD loss: KL divergence between cross-modal student and teacher logits.

    Source: src/open_clip/loss.py lines 320-321.

    Requires ICLLoss to have run first so that KDFeatures.cross_logits_img2txt
    and cross_logits_txt2img are populated. CompositeLoss enforces this order.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.T = temperature

    def _kl(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute Cross-KD loss.

        Args:
            features: KDFeatures with cross_logits_img2txt,
                      cross_logits_txt2img, t_img, t_txt, t_logit_scale.

        Returns:
            Scalar Cross-KD loss.
        """
        assert features.cross_logits_img2txt is not None, (
            "ICLLoss must run before CrossKDLoss. "
            "Check CompositeLoss ordering."
        )

        t_logits_img = features.t_logit_scale * features.t_img @ features.t_txt.T
        t_logits_txt = t_logits_img.T

        cross_kd_loss = (
            self._kl(features.cross_logits_img2txt, t_logits_img.detach())
            + self._kl(features.cross_logits_txt2img, t_logits_txt.detach())
        ) / 2
        return cross_kd_loss
