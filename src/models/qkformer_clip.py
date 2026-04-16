"""CLIP-compatible model combining QKFormer image encoder with an open_clip text encoder.

QKFormerCLIPModel provides the same interface as open_clip.CLIP so that it can be
wrapped transparently by CLIPWrapper and used with the existing CLIPModule /
CLIPKDModule training pipelines.

Image encoder: hierarchical_spiking_transformer (QKFormer_10_512)
    - Always outputs 512-dim embeddings before any projection.
    - SNN neuron states are reset before every encode_image call via
      reset_lif_states() — this correctly handles our custom LIFNode which
      is not a spikingjelly MemoryModule (unlike functional.reset_net).
    - Input images are tiled T times along a new temporal dimension to create
      the SNN input sequence; outputs are averaged across T.

Text encoder: borrowed from an open_clip CLIP model (ViT-B/16 or ViT-T/16).
    - The open_clip model is created with pretrained=None and the full model
      is stored so that encode_text() works unchanged.

Projection: when visual_embed_dim != text_embed_dim a learned linear layer
    projects image features into the text embedding space.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spikingjelly.activation_based.base import MemoryModule as _SJMemoryModule

from src.models.visual_encoders.lif_node import reset_lif_states


class QKFormerCLIPModel(nn.Module):
    """CLIP-compatible model with a spiking QKFormer image encoder.

    Satisfies the interface expected by CLIPWrapper:
        encode_image(image, normalize=False) -> Tensor[B, D]
        encode_text(text, normalize=False)  -> Tensor[B, D]
        logit_scale: nn.Parameter
        forward(image, text) -> (img_norm, txt_norm, logit_scale_exp)

    Args:
        visual: QKFormer backbone (hierarchical_spiking_transformer with
                num_classes=0).  Its forward_features() returns [T, B, D].
        text_model: Full open_clip CLIP instance used as text encoder.
                    Only encode_text() is called on it.
        visual_embed_dim: Channel dimension output by the visual backbone (512
                          for QKFormer_10_512).
        text_embed_dim: Embedding dimension of the text encoder (512 for ViT-B,
                        256 for ViT-T).
        T: SNN simulation timesteps.  The same input frame is repeated T times.
        init_logit_scale: Initial value for the learnable log temperature.
    """

    def __init__(
        self,
        visual: nn.Module,
        text_model: nn.Module,
        visual_embed_dim: int,
        text_embed_dim: int,
        T: int,
        init_logit_scale: float = math.log(1 / 0.07),
    ) -> None:
        super().__init__()
        self.visual = visual
        self.text_model = text_model
        self.T = T

        # Learnable projection when image and text live in different spaces
        if visual_embed_dim != text_embed_dim:
            self.visual_proj: nn.Module = nn.Linear(
                visual_embed_dim, text_embed_dim, bias=False
            )
        else:
            self.visual_proj = None

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    @torch.cuda.nvtx.range("ImageEncode")
    def encode_image(self, image: Tensor, normalize: bool = False) -> Tensor:
        """Encode a batch of images to embeddings via QKFormer.

        Args:
            image: [B, C, H, W] image tensor.
            normalize: If True, L2-normalise the output.

        Returns:
            [B, D] image feature tensor where D = text_embed_dim.
        """
        # Reset SNN neuron states before each batch.
        reset_lif_states(self.visual)  # custom LIFNode: inplace (CUDA graph safe)
        for m in getattr(self.visual, "_orig_mod", self.visual).modules():
            if isinstance(m, _SJMemoryModule):
                m.reset()  # spikingjelly neurons (sj_lif, plif, glif)

        # Replicate input across T timesteps: [T, B, C, H, W]
        x = image.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # Run backbone → [T, B, visual_embed_dim]
        x = self.visual.forward_features(x)
        # Average over time → [B, visual_embed_dim]
        x = x.mean(0)

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

    # ------------------------------------------------------------------
    # Forward (used by CLIPWrapper in non-distill mode)
    # ------------------------------------------------------------------

    @torch.cuda.nvtx.range("Forward")
    def forward(self, image: Tensor, text: Tensor):
        """Standard CLIP forward returning L2-normalised features.

        Args:
            image: [B, C, H, W] image tensor.
            text: [B, L] tokenised text.

        Returns:
            Tuple of (img_feats, txt_feats, logit_scale_exp) where features
            are L2-normalised and logit_scale_exp = exp(logit_scale).
        """
        img = self.encode_image(image, normalize=True)
        txt = self.encode_text(text, normalize=True)
        return img, txt, self.logit_scale.exp()
