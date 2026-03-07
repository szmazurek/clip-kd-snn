"""CLIP model wrapper and output type.

The actual model implementation is open_clip.CLIP. This module defines
a typed output dataclass for cleaner Lightning module code and re-exports
the factory helpers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CLIPOutput:
    """Typed output from a CLIP model forward pass.

    Attributes:
        image_embeds: (B, D) image embeddings. L2-normalised when
                      distill=False; raw when distill=True.
        text_embeds:  (B, D) text embeddings. L2-normalised when
                      distill=False; raw when distill=True.
        logit_scale:  Scalar temperature (exp of the learned parameter).
    """
    image_embeds: torch.Tensor
    text_embeds: torch.Tensor
    logit_scale: torch.Tensor


class CLIPWrapper(nn.Module):
    """Thin wrapper around open_clip.CLIP adding distill-mode forward.

    open_clip.CLIP.forward() always L2-normalises output features.
    KD training needs raw (un-normalised) features so that losses can
    normalise after gathering across GPUs.

    Attribute access is proxied to the inner model so that callers can
    still reach model.logit_scale, model.visual, etc.

    Args:
        model: An open_clip.CLIP instance.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        distill: bool = False,
        **kwargs,
    ):
        """Forward pass.

        Args:
            image: (B, C, H, W) image tensor.
            text: (B, L) tokenised text tensor.
            distill: If True, return raw (un-normalised) features via
                     encode_image/encode_text with normalize=False.
                     If False (default), delegate to normal forward which
                     returns L2-normalised features.
            **kwargs: Absorbed and ignored (e.g. mask_ratio=0.0 from
                      CLIPKDModule when the masked path is not taken).

        Returns:
            Tuple of (image_features, text_features, logit_scale_exp).
        """
        if distill:
            image_features = self.model.encode_image(image, normalize=False)
            text_features = self.model.encode_text(text, normalize=False)
            return image_features, text_features, self.model.logit_scale.exp()
        return self.model(image, text)
