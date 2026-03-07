"""Contrastive Representation Distillation / CKD loss (Equations 4-10 in paper).

Naming note: The paper (Section 3.2) calls this CRD (Contrastive Relational
Distillation). The original implementation (src/open_clip/loss.py) uses the
variable name `ckd_loss` and the argument `alpha_ckd_loss`. We use `CKDLoss`
as the class name to match the original code, and document the equivalence.

Source: src/open_clip/loss.py
  - DistillKL: lines 200-210
  - ckd_loss computation: lines 317-318
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import CLIPDistillationLoss, KDFeatures


class _DistillKL(nn.Module):
    """KL divergence loss for knowledge distillation (Hinton et al., 2015).

    Source: src/open_clip/loss.py lines 200-210.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.T = temperature

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)


class CKDLoss(CLIPDistillationLoss):
    """CKD / CRD loss: KL divergence between student and teacher logit distributions.

    Paper: Section 3.2, Equations 4-10 (called CRD in paper).
    Source: src/open_clip/loss.py lines 317-318.

    Minimises KL(student_logits || teacher_logits.detach()) for both
    image-to-text and text-to-image directions, averaged.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.kl = _DistillKL(temperature)

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute CKD loss.

        Args:
            features: KDFeatures with s_img_proj, s_txt_proj, t_img, t_txt,
                      s_logit_scale, t_logit_scale, labels.

        Returns:
            Scalar CKD loss.
        """
        # Student same-modal logits
        s_logits_img = features.s_logit_scale * features.s_img_proj @ features.s_txt_proj.T
        s_logits_txt = s_logits_img.T

        # Teacher same-modal logits (detached — teacher is frozen but we
        # re-compute here to avoid depending on call order)
        t_logits_img = features.t_logit_scale * features.t_img @ features.t_txt.T
        t_logits_txt = t_logits_img.T

        ckd_loss = (
            self.kl(s_logits_img, t_logits_img.detach())
            + self.kl(s_logits_txt, t_logits_txt.detach())
        ) / 2
        return ckd_loss
