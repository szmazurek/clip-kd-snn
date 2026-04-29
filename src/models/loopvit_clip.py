"""CLIP-compatible model combining LoopViT image encoder with an open_clip text encoder.

LoopViTCLIPModel provides the same interface as open_clip.CLIP so that it can be
wrapped transparently by CLIPWrapper and used with the existing CLIPModule /
CLIPKDModule training pipelines.

Image encoder: LoopViT (recurrent ViT-B/16-derived, non-SNN)
    - Runs max_loop_steps recurrent transformer passes over patch tokens.
    - Returns CLS token embedding via forward_features() — shape [B, embed_dim].
    - No temporal simulation or SNN state resets required.

Text encoder: borrowed from an open_clip CLIP model (ViT-B/16).
    - Only encode_text() is called; the open_clip visual encoder is discarded.

Projection: when visual_embed_dim != text_embed_dim a learned linear layer
    projects image features into the text embedding space (e.g. 768 → 512).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoopViTCLIPModel(nn.Module):
    """CLIP-compatible model with a LoopViT image encoder.

    Satisfies the interface expected by CLIPWrapper:
        encode_image(image, normalize=False) -> Tensor[B, D]
        encode_text(text, normalize=False)  -> Tensor[B, D]
        logit_scale: nn.Parameter
        forward(image, text) -> (img_norm, txt_norm, logit_scale_exp)

    Args:
        visual: LoopViT backbone with forward_features() returning [B, visual_embed_dim].
        text_model: Full open_clip CLIP instance used as text encoder.
                    Only encode_text() is called on it.
        visual_embed_dim: Channel dimension output by the visual backbone (e.g. 768).
        text_embed_dim: Embedding dimension of the text encoder (512 for ViT-B/16).
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
        """Encode a batch of images to embeddings via LoopViT.

        Args:
            image: [B, C, H, W] image tensor.
            normalize: If True, L2-normalise the output.

        Returns:
            [B, D] image feature tensor where D = text_embed_dim.
        """
        x = self.visual.forward_features(image)  # [B, visual_embed_dim]

        if self.visual_proj is not None:
            x = self.visual_proj(x)

        return F.normalize(x, dim=-1) if normalize else x

    @torch.cuda.nvtx.range("TextEncode")
    def encode_text(self, text: Tensor, normalize: bool = False) -> Tensor:
        """Encode tokenised text via the open_clip text encoder.

        Args:
            text: [B, L] token id tensor.
            normalize: If True, L2-normalise the output.

        Returns:
            [B, D] text feature tensor.
        """
        return self.text_model.encode_text(text, normalize=normalize)

    @torch.cuda.nvtx.range("Forward")
    def forward(self, image: Tensor, text: Tensor):
        """Standard CLIP forward returning L2-normalised features.

        Returns:
            Tuple of (img_feats, txt_feats, logit_scale_exp) where features
            are L2-normalised and logit_scale_exp = exp(logit_scale).
        """
        img = self.encode_image(image, normalize=True)
        txt = self.encode_text(text, normalize=True)
        return img, txt, self.logit_scale.exp()
