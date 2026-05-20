"""SigREG: Sketched Isotropic Gaussian Regularization.

Ported from sigreg/cifar100 and sigreg/nanogpt — forces the empirical
covariance of internal block representations to match the identity matrix.

References:
    sigreg/ repo (weak variant: covariance → identity via Frobenius norm).
"""
from __future__ import annotations

import logging
from typing import List

import torch
from torch import nn


def sigreg_weak_loss(x: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Weak SigREG: forces Cov(x) ≈ I via Frobenius norm.

    Args:
        x: [N, C] tensor (N samples, C features).
        sketch_dim: Random-projection target dimension. When C > sketch_dim,
                    x is compressed to [N, sketch_dim] before covariance
                    estimation, keeping cost at O(N * sketch_dim²).

    Returns:
        Scalar loss tensor (differentiable w.r.t. x).
    """
    N, C = x.shape

    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C

    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    target = torch.eye(sketch_dim, device=x.device)
    return torch.norm(cov - target, p="fro")


def find_visual_blocks(visual: nn.Module) -> List[nn.Module]:
    """Return transformer blocks from a visual encoder, or [] with a warning.

    Tries three layouts in order:
      1. open_clip ViT:       visual.transformer.resblocks
      2. LoopViT global mode: visual.encoder.blocks  (shared encoder, recurred N times)
      3. LoopViT per-block:   visual.blocks[i].blocks (one TransformerEncoder per slot)

    For LoopViT the same physical blocks are called multiple times per
    forward pass — sigREG hooks accumulate a loss each firing, averaged
    over all firings at the end of training_step.

    Returns an empty list (with a warning) for unsupported architectures
    so sigREG silently becomes a no-op rather than crashing.
    """
    # open_clip ViT: visual.transformer.resblocks
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return list(visual.transformer.resblocks)

    # LoopViT global mode: single shared TransformerEncoder with inner TransformerBlocks
    encoder = getattr(visual, "encoder", None)
    if encoder is not None and hasattr(encoder, "blocks"):
        return list(encoder.blocks)

    # LoopViT per-block mode: nn.ModuleList of TransformerEncoders (depth=1 each)
    loop_blocks = getattr(visual, "blocks", None)
    if loop_blocks is not None:
        inner: List[nn.Module] = []
        for blk in loop_blocks:
            if hasattr(blk, "blocks"):
                inner.extend(blk.blocks)
            else:
                inner.append(blk)
        if inner:
            return inner

    logging.warning(
        "SigREG: could not locate transformer blocks in visual encoder "
        "(tried .transformer.resblocks, .encoder.blocks, .blocks[i].blocks) — "
        "hook registration skipped, sigREG will be a no-op."
    )
    return []
