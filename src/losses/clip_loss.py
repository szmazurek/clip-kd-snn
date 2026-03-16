"""Standard CLIP InfoNCE loss (Equations 1-3 in the CLIP-KD paper).

Ported from src/open_clip/loss.py (ClipLoss). The gather step is removed
because gathering happens once upstream in CLIPKDModule.training_step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import CLIPDistillationLoss, KDFeatures


class CLIPInfoNCELoss(CLIPDistillationLoss):
    """Symmetric cross-entropy InfoNCE loss over image-text pairs.

    Paper: Eq. 1-3 (standard CLIP contrastive loss).
    Source: src/open_clip/loss.py ClipLoss.forward() lines 164-196.

    Features are assumed to be gathered across GPUs and L2-normalised
    before being passed in via KDFeatures.
    """

    def forward(self, features: KDFeatures) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            features: KDFeatures with s_img, s_txt, s_logit_scale, labels.

        Returns:
            Scalar loss tensor.
        """
        # Cast to float32 before cross_entropy: in bf16, off-diagonal softmax values
        # underflow to 0 when logit_scale × cos_sim > log(bf16_max) ≈ 88.7, producing
        # a wrong gradient of -1 on the diagonal and exploding the logit_scale update.
        logits_per_image = (features.s_logit_scale * features.s_img @ features.s_txt.T).float()
        logits_per_text = logits_per_image.T

        total_loss = (
            F.cross_entropy(logits_per_image, features.labels)
            + F.cross_entropy(logits_per_text, features.labels)
        ) / 2
        return total_loss
