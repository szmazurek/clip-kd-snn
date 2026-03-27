"""Base classes and shared data structures for CLIP-KD losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class KDFeatures:
    """Container for all pre-computed features passed to distillation losses.

    All embeddings are gathered across GPUs and L2-normalised before being
    placed in this dataclass. Gathering and normalisation happen once in
    CLIPKDModule.training_step to avoid redundant computation.

    Attributes:
        s_img: (N, D_s) Student image embeddings, normalised, pre-projection.
        s_txt: (N, D_s) Student text embeddings, normalised, pre-projection.
        s_img_proj: (N, D_t) Student image embeddings after projection head.
                    Equals s_img when student and teacher embed dims match.
        s_txt_proj: (N, D_t) Student text embeddings after projection head.
                    Equals s_txt when student and teacher embed dims match.
        t_img: (N, D_t) Teacher image embeddings, normalised.
        t_txt: (N, D_t) Teacher text embeddings, normalised.
        s_logit_scale: Scalar logit scale for student (exp of learned param).
        t_logit_scale: Scalar logit scale for teacher.
        labels: (N,) Ground-truth contrastive labels (torch.arange(N)).
    """
    s_img: torch.Tensor
    s_txt: torch.Tensor
    s_img_proj: torch.Tensor
    s_txt_proj: torch.Tensor
    t_img: torch.Tensor
    t_txt: torch.Tensor
    s_logit_scale: torch.Tensor
    t_logit_scale: torch.Tensor
    labels: torch.Tensor


class CLIPDistillationLoss(nn.Module):
    """Abstract base class for all CLIP knowledge distillation losses.

    All concrete losses receive a KDFeatures instance and return a scalar
    tensor. The CompositeLoss sums them with per-loss lambda weights.
    """

    def forward(self, features: KDFeatures) -> torch.Tensor:
        raise NotImplementedError
