"""Gradient Distillation (GD) loss (Equation 13 and Appendix A in paper).

GD matches the analytical gradients of the student's CLIP loss with respect
to embeddings against those of the teacher. The "gradients" are computed
analytically (not via autograd) using the closed-form expression for the
gradient of cross-entropy softmax loss (Eq. 19-24 in Appendix A).

This is the most numerically sensitive loss. get_grad() is ported verbatim
from the original to preserve exact behaviour.

Source: src/open_clip/loss.py
  - get_grad(): lines 129-138
  - gd_loss computation: lines 326-337
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import CLIPDistillationLoss, KDFeatures


def get_grad(
    p: torch.Tensor,
    k: torch.Tensor,
    tau: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical gradients of InfoNCE loss w.r.t. query and key embeddings.

    Computes (d L / d p) and (d L / d k) analytically using the softmax
    gradient formula, as described in Appendix A Equations 19-24 of the paper.

    Ported verbatim from src/open_clip/loss.py lines 129-138.

    Args:
        p: Query embeddings (N, D), e.g. image features.
        k: Key embeddings (N, D), e.g. text features.
        tau: Temperature scalar (logit_scale).
        targets: Integer labels (N,), i.e. torch.arange(N).

    Returns:
        Tuple of (grad_p, grad_k):
            grad_p: (N, D) gradient w.r.t. p.
            grad_k: (N, D) gradient w.r.t. k (diagonal of the full Jacobian).
    """
    logits = p @ k.T / tau
    targets_oh = F.one_hot(targets, num_classes=logits.size(1)).float()
    prob = F.softmax(logits, dim=1)
    grad_p = (prob - targets_oh) @ k / tau / targets.size(0)

    embed_size = p.size(1)
    prob_targets_repeat = (
        (prob - targets_oh).t().repeat(1, embed_size).view(-1, embed_size, p.size(0))
    )
    grad_k = (
        prob_targets_repeat * (p.t() / tau).unsqueeze(0)
    ).sum(-1) / targets.size(0)

    return grad_p, grad_k


class GDLoss(CLIPDistillationLoss):
    """Gradient Distillation loss.

    Paper: Section 3.5, Equation 13, and Appendix A (Equations 19-24).
    Source: src/open_clip/loss.py lines 326-337.

    Minimises MSE between student and teacher analytical gradients for both
    image-to-text and text-to-image directions.

    Teacher gradients are computed under torch.no_grad() since the teacher
    is frozen. Student gradients retain the computation graph so they
    contribute to the student's backward pass.
    """

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute GD loss.

        Args:
            features: KDFeatures with s_img_proj, s_txt_proj, t_img, t_txt,
                      s_logit_scale, t_logit_scale, labels.

        Returns:
            Scalar GD loss.
        """
        with torch.no_grad():
            t_grad_p_img, t_grad_k_txt = get_grad(
                features.t_img, features.t_txt, features.t_logit_scale, features.labels
            )
            t_grad_p_txt, t_grad_k_img = get_grad(
                features.t_txt, features.t_img, features.t_logit_scale, features.labels
            )

        s_grad_p_img, s_grad_k_txt = get_grad(
            features.s_img_proj, features.s_txt_proj, features.s_logit_scale, features.labels
        )
        s_grad_p_txt, s_grad_k_img = get_grad(
            features.s_txt_proj, features.s_img_proj, features.s_logit_scale, features.labels
        )

        gd_loss = (
            F.mse_loss(s_grad_p_img, t_grad_p_img.detach())
            + F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach())
            + F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach())
            + F.mse_loss(s_grad_k_img, t_grad_k_img.detach())
        )
        return gd_loss
