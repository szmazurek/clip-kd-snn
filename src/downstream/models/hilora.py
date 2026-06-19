"""HiLoRA: hierarchical LoRA adapters keyed to weight-matrix identity.

No peft dependency. Adapters are patched in-place onto TransformerBlock objects.
Because each TransformerBlock is a distinct Python object, calling the same block
N times reuses the same adapter weights — the weight-matrix-identity property that
makes LoRA degenerate for depth=1 (good: 1 adapter) and hierarchical for depth=3
(3 adapters, one per block) or depth=12 (12 adapters, matching standard HiVG).

Public API:
    patch_block_with_lora(block, rank, alpha)   — in-place, idempotent
    get_lora_parameters(model)                  — yields only LoRA params
    freeze_non_lora(model)                      — freeze everything except LoRA
    lora_state_dict(model)                      — extract just the LoRA weights
"""
from __future__ import annotations

from typing import Iterator, Union

import torch
import torch.nn as nn
from open_clip.transformer import ResidualAttentionBlock
from torch import Tensor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

from ...models.visual_encoders.transformer_utils import TransformerBlock
from .hf_clip_visual import HFCLIPVisionBlockAdapter

AnyBlock = Union[TransformerBlock, ResidualAttentionBlock, CLIPEncoderLayer, HFCLIPVisionBlockAdapter]


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank delta.

    W_eff = W_frozen + (B @ A) * scale,   scale = alpha / rank
    A is initialised Kaiming-uniform, B is zero → delta starts at zero.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale

    @property
    def weight(self) -> Tensor:
        """Proxy to the wrapped linear's weight — open_clip's
        ResidualAttentionBlock.get_weight_dtype() reads mlp.c_fc.weight.dtype
        directly (bypassing forward()) to pick an autocast dtype; this keeps
        that read-only introspection working without affecting the actual
        forward computation, which already correctly calls LoRALinear.forward()."""
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.linear.bias


def _patch_loopvit_block(block: TransformerBlock, rank: int, alpha: float) -> None:
    """Targets:
        block.attn.qkv   (fused QKV projection)
        block.attn.proj  (output projection)
        block.mlp.w_in / block.mlp.w_out    (GELU FFN)
        block.mlp.w1 / block.mlp.w2 / block.mlp.w_out  (SwiGLU FFN)
    """
    attn = block.attn
    if not isinstance(attn.qkv, LoRALinear):
        attn.qkv = LoRALinear(attn.qkv, rank, alpha)
    if not isinstance(attn.proj, LoRALinear):
        attn.proj = LoRALinear(attn.proj, rank, alpha)

    mlp = block.mlp
    # GELU FFN has w_in / w_out; SwiGLU has w1 / w2 / w_out
    if hasattr(mlp, "w_in") and not isinstance(mlp.w_in, LoRALinear):
        mlp.w_in = LoRALinear(mlp.w_in, rank, alpha)
    if hasattr(mlp, "w1") and not isinstance(mlp.w1, LoRALinear):
        mlp.w1 = LoRALinear(mlp.w1, rank, alpha)
    if hasattr(mlp, "w2") and not isinstance(mlp.w2, LoRALinear):
        mlp.w2 = LoRALinear(mlp.w2, rank, alpha)
    if hasattr(mlp, "w_out") and not isinstance(mlp.w_out, LoRALinear):
        mlp.w_out = LoRALinear(mlp.w_out, rank, alpha)


def _patch_openclip_block(block: ResidualAttentionBlock, rank: int, alpha: float) -> None:
    """Targets block.mlp.c_fc, block.mlp.c_proj only.

    block.attn is a plain nn.MultiheadAttention, whose forward() reads
    in_proj_weight/in_proj_bias and out_proj.weight/.bias as raw tensors
    (it never calls out_proj's own .forward()), so none of its projections
    can be swapped for a LoRALinear wrapper without reimplementing
    nn.MultiheadAttention's forward — skipped as a documented simplification.
    """
    mlp = block.mlp
    if not isinstance(mlp.c_fc, LoRALinear):
        mlp.c_fc = LoRALinear(mlp.c_fc, rank, alpha)
    if not isinstance(mlp.c_proj, LoRALinear):
        mlp.c_proj = LoRALinear(mlp.c_proj, rank, alpha)


def _patch_hf_clip_block(block: CLIPEncoderLayer | HFCLIPVisionBlockAdapter, rank: int, alpha: float) -> None:
    """Targets self_attn.q_proj/k_proj/v_proj/out_proj — these are four
    separate nn.Linear modules called via ordinary forward(), so unlike
    open_clip's fused nn.MultiheadAttention, attention itself is directly
    LoRA-wrappable here. Matches HiVG's actual effective LoRA scope exactly:
    its target_modules list also includes "fc_in"/"fc_out", but those don't
    match CLIPMLP's real attribute names (fc1/fc2), so the released code
    never touches the MLP either — mirrored here by simply not patching it.
    """
    attn = block.self_attn
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        linear = getattr(attn, name)
        if not isinstance(linear, LoRALinear):
            setattr(attn, name, LoRALinear(linear, rank, alpha))


def patch_block_with_lora(block: AnyBlock, rank: int = 32, alpha: float = 16.0) -> None:
    """Add LoRA adapters to a visual block in-place. Idempotent.

    Dispatches on block type: our own TransformerBlock (LoopViT),
    open_clip's ResidualAttentionBlock (legacy open_clip text path), or a
    HuggingFace CLIPEncoderLayer / HFCLIPVisionBlockAdapter (real pretrained
    CLIP backbone, vision or text).
    """
    if isinstance(block, TransformerBlock):
        _patch_loopvit_block(block, rank, alpha)
    elif isinstance(block, ResidualAttentionBlock):
        _patch_openclip_block(block, rank, alpha)
    elif isinstance(block, (CLIPEncoderLayer, HFCLIPVisionBlockAdapter)):
        _patch_hf_clip_block(block, rank, alpha)
    else:
        raise TypeError(f"Don't know how to patch LoRA onto block type {type(block)}")


def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only LoRA A/B parameters from a model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A.weight
            yield module.lora_B.weight


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter that is not a LoRA A/B weight."""
    lora_params = set(id(p) for p in get_lora_parameters(model))
    for p in model.parameters():
        p.requires_grad_(id(p) in lora_params)


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a state dict containing only LoRALinear parameters."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def set_lora_trainable(blocks: list[AnyBlock], trainable: bool) -> None:
    """Toggle requires_grad on the LoRA A/B weights of the given blocks.

    Used to implement HiLoRA's staged curriculum: blocks for stages not yet
    reached have their adapters patched (so checkpoints have stable shapes)
    but frozen (requires_grad=False) until their stage is unlocked.
    """
    for block in blocks:
        for module in block.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.requires_grad_(trainable)
                module.lora_B.weight.requires_grad_(trainable)


def blocks_for_stage(visual: nn.Module, stage: int) -> list[AnyBlock]:
    """Return the cumulative set of unique blocks active at a given HiLoRA stage.

    Cumulative, not an exclusive partition — stage k's set is a strict
    superset of stage k-1's. Matches HiVG/models/HiVG.py's set_HiLoRA, whose
    literal per-stage layer-index conditions for a 12-layer backbone unlock
    layers 0-4, then 0-7, then 0-11 (5/8/12, not an even 4/4/4 partition) —
    consistent with the paper's "parameters of the low-stage HiLoRA are
    included in the high-stage HiLoRA."

    Works for any LoopViT/HFCLIPVisionBackbone configuration via
    iter_blocks(): blocks are deduplicated by identity, preserving
    first-execution order, before slicing — so an uneven per_block schedule
    (e.g. loop_schedule=[1,10,1]) still gets one clean group per unique block,
    unlike slicing the raw (repeated) execution sequence.

    Args:
        visual: backbone with an iter_blocks() method.
        stage: 1, 2, or 3.

    Returns:
        The first N unique blocks (first-execution order), N = the cumulative
        count for this stage. For exactly 12 unique blocks (HiVG's actual
        ViT-B/16 case) N = 5, 8, 12 for stages 1, 2, 3. Other block counts use
        an evenly-split (remainder to earlier groups), cumulative generalization.
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")

    seen: set[int] = set()
    unique_blocks: list[AnyBlock] = []
    for b in visual.iter_blocks():
        if id(b) not in seen:
            seen.add(id(b))
            unique_blocks.append(b)
    total = len(unique_blocks)

    if total == 12:
        cum_counts = {1: 5, 2: 8, 3: 12}   # exact thresholds from HiVG's released code
    else:
        base, rem = divmod(total, 3)
        sizes = [base + (1 if i < rem else 0) for i in range(3)]
        cum_counts = {1: sizes[0], 2: sizes[0] + sizes[1], 3: total}

    return unique_blocks[: cum_counts[stage]]
