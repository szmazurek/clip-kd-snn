"""Adapter exposing a real pretrained HuggingFace CLIP visual transformer
through the minimal interface HiVGLoopViT's block-level forward expects
(image_tokens / iter_blocks / head_norm).

Replaces the former open_clip-based backbone for the CLIP baseline: open_clip's
attention is nn.MultiheadAttention, which reads in_proj_weight/out_proj as raw
tensors inside its forward (bypassing any submodule wrapping), so LoRA could
only ever reach the MLP there. HuggingFace's CLIPAttention keeps q_proj/k_proj/
v_proj/out_proj as four separate nn.Linear modules called via ordinary
forward() — exactly what HiVG's own released code relies on (its
target_modules list, ["q_proj", "k_proj", "v_proj", "out_proj", ...], only
matches things shaped like this) — so this is the backbone HiVG's actual code
is built on, not an approximation of it.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


class HFCLIPVisionBlockAdapter:
    """Makes a CLIPEncoderLayer callable as block(x), matching the single-arg
    convention every other block type in this codebase uses (CLIPEncoderLayer
    itself requires an explicit attention_mask argument), while still
    exposing self_attn/mlp so patch_block_with_lora can reach into the real
    underlying layer — LoRA patches mutate the same nn.Linear objects used by
    the real forward pass, this is bookkeeping only, not a copy.
    """

    def __init__(self, layer: CLIPEncoderLayer) -> None:
        self._layer = layer
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp

    def __call__(self, x: Tensor) -> Tensor:
        return self._layer(x, attention_mask=None)

    def modules(self):
        return self._layer.modules()


class HFCLIPVisionBackbone(nn.Module):
    """Wraps a HuggingFace CLIPVisionTransformer (`CLIPModel.vision_model`)
    for use as a HiVGLoopViT visual backbone.
    """

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.embed_dim = vision_model.config.hidden_size
        self._block_adapters = [HFCLIPVisionBlockAdapter(layer) for layer in vision_model.encoder.layers]

    def image_tokens(self, images: Tensor) -> Tensor:
        x = self.vision_model.embeddings(images)
        return self.vision_model.pre_layrnorm(x)

    def iter_blocks(self):
        yield from self._block_adapters

    @property
    def head_norm(self) -> nn.Module:
        return self.vision_model.post_layernorm
