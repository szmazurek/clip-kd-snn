"""CLIP-compatible model combining a PseudoSNN SEWResNet image encoder with an open_clip text encoder.

SEWResnetPseudoCLIPModel provides the same interface as open_clip.CLIP so that it can be
wrapped transparently by CLIPWrapper and used with the existing CLIPModule / CLIPKDModule
training pipelines.

Image encoder: SEWResNet (sew_resnet18 / sew_resnet34 / sew_resnet50)
    - Uses PseudoNeuron activations: ReLU proxy during training, real IF neuron at inference.
    - T is managed internally by PseudoNeuron (starts at T=1 after the stem unsqueeze).
    - No explicit SNN reset needed between batches — PseudoNeuron is stateless.
    - forward_features() returns [B, C] pooled features (C=512 for resnet18/34, 2048 for resnet50).

Text encoder: borrowed from an open_clip CLIP model (ViT-B/16 by default).

Projection: when visual_embed_dim != text_embed_dim a learned linear layer projects image
    features into the text embedding space (e.g. ResNet50: 2048 → 512).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SEWResnetPseudoCLIPModel(nn.Module):
    """CLIP-compatible model with a PseudoSNN SEWResNet image encoder.

    Satisfies the interface expected by CLIPWrapper:
        encode_image(image, normalize=False) -> Tensor[B, D]
        encode_text(text, normalize=False)  -> Tensor[B, D]
        logit_scale: nn.Parameter
        forward(image, text) -> (img_norm, txt_norm, logit_scale_exp)

    Args:
        visual: SEWResNet backbone with forward_features() returning [B, visual_embed_dim].
        text_model: Full open_clip CLIP instance used as text encoder (visual deleted).
        visual_embed_dim: Feature dim from the backbone (512 for resnet18/34, 2048 for resnet50).
        text_embed_dim: Embedding dim of the text encoder (512 for ViT-B/16).
        init_logit_scale: Initial value for the learnable log temperature.
    """

    def __init__(
        self,
        visual: nn.Module,
        text_model: nn.Module,
        visual_embed_dim: int,
        text_embed_dim: int,
        init_logit_scale: float = math.log(1 / 0.07),
    ) -> None:
        super().__init__()
        self.visual = visual
        self.text_model = text_model

        if visual_embed_dim != text_embed_dim:
            self.visual_proj: nn.Module = nn.Linear(
                visual_embed_dim, text_embed_dim, bias=False
            )
        else:
            self.visual_proj = None

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    @torch.cuda.nvtx.range("ImageEncode")
    def encode_image(self, image: Tensor, normalize: bool = False) -> Tensor:
        x = self.visual.forward_features(image)  # [B, visual_embed_dim]

        if self.visual_proj is not None:
            x = self.visual_proj(x)

        return F.normalize(x, dim=-1) if normalize else x

    @torch.cuda.nvtx.range("TextEncode")
    def encode_text(self, text: Tensor, normalize: bool = False) -> Tensor:
        return self.text_model.encode_text(text, normalize=normalize)

    @torch.cuda.nvtx.range("Forward")
    def forward(self, image: Tensor, text: Tensor):
        img = self.encode_image(image, normalize=True)
        txt = self.encode_text(text, normalize=True)
        return img, txt, self.logit_scale.exp()
