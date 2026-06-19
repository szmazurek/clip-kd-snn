"""Grounding losses: box regression + CLC + RTCC + use_mask_loss.

Four terms matching HiVG's actual loss_utils.trans_vg_loss:
  box       — smooth-L1 (λ=2) + GIoU (λ=2)
  clc       — symmetric CLIP InfoNCE (image-text contrastive)
  rtcc      — Focal (λ=20) + Dice (λ=2) on patch-level region-text similarity
  mask_loss — Focal (λ=20) + Dice (λ=2) on a second, higher-resolution
              soft-segmentation map (HiVG's --use_mask_loss, passed in every
              one of their released training commands, no exceptions) —
              same box-derived target as RTCC, just compared at a different
              (upsampled) resolution against a separately-headed prediction.

All λ values are defaults; they are multiplied in this module, not in the caller.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils.box_utils import generalized_box_iou, xywh2xyxy


# ---------------------------------------------------------------------------
# Primitive loss functions
# ---------------------------------------------------------------------------

def _sigmoid_focal_loss(
    inputs: Tensor, targets: Tensor, num_boxes: int,
    alpha: float = 0.25, gamma: float = 2.0,
) -> Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean(1).sum() / num_boxes


def _dice_loss(inputs: Tensor, targets: Tensor, num_boxes: int) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    num = 2 * (inputs * targets).sum(1)
    den = inputs.sum(-1) + targets.sum(-1)
    return (1 - (num + 1) / (den + 1)).sum() / num_boxes


def _contrastive_loss(logits: Tensor) -> Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def _clip_loss(similarity: Tensor) -> Tensor:
    return (_contrastive_loss(similarity) + _contrastive_loss(similarity.t())) / 2.0


# ---------------------------------------------------------------------------
# Top-level loss function
# ---------------------------------------------------------------------------

def grounding_loss(
    pred_box: Tensor,
    gt_box: Tensor,
    logits_per_text: Tensor | None,
    visu_sim: Tensor | None,
    obj_mask: Tensor | None,
    seg_mask: Tensor | None = None,
    lambda_l1: float = 2.0,
    lambda_giou: float = 2.0,
    lambda_focal: float = 20.0,
    lambda_dice: float = 2.0,
    use_contrastive: bool = True,
    use_rtcc: bool = True,
    use_mask_loss: bool = True,
) -> dict[str, Tensor]:
    """Compute all grounding losses and return a named dict.

    Args:
        pred_box: [B, 4] predicted boxes in normalized (cx,cy,w,h).
        gt_box:   [B, 4] ground-truth boxes in normalized (cx,cy,w,h).
        logits_per_text: [B, B] text→image similarity matrix (from logit_scale).
        visu_sim: [B, N_patches] patch-level text-visual dot-product scores.
        obj_mask: [B, 1, H, W] binary object mask (float, 0/1).
        seg_mask: [B, 1, h, w] second, higher-res soft-segmentation prediction
            (HiVGLoopViT.forward's seg_mask) — upsampled to obj_mask's
            resolution here, mirroring HiVG's trans_vg_loss exactly (RTCC
            downsamples the GT mask to patch resolution; this upsamples the
            prediction to image resolution instead).
        lambda_*: loss weights.
        use_contrastive: whether to compute CLC loss.
        use_rtcc: whether to compute RTCC loss.
        use_mask_loss: whether to compute the use_mask_loss term.

    Returns:
        Dict of named scalar tensors. Sum values for total loss.
    """
    B = pred_box.shape[0]
    losses: dict[str, Tensor] = {}

    # --- Box losses ---
    losses["loss_bbox"] = F.l1_loss(pred_box, gt_box, reduction="sum") / B * lambda_l1
    losses["loss_giou"] = (
        1 - torch.diag(generalized_box_iou(xywh2xyxy(pred_box), xywh2xyxy(gt_box)))
    ).sum() / B * lambda_giou

    # --- Image-text contrastive (CLC) ---
    if use_contrastive and logits_per_text is not None:
        losses["loss_clc"] = _clip_loss(logits_per_text)

    # --- Region-text contrastive (RTCC) ---
    if use_rtcc and visu_sim is not None and obj_mask is not None:
        patch_num = int(visu_sim.shape[-1] ** 0.5)
        # downsample mask to patch grid
        tgt = F.interpolate(
            obj_mask.float(), size=(patch_num, patch_num), mode="nearest"
        )[:, 0]   # [B, patch_num, patch_num]
        tgt = tgt.flatten(1).float()           # [B, N_patches]
        sim = visu_sim.flatten(1)
        losses["loss_rtcc_focal"] = _sigmoid_focal_loss(sim, tgt, B) * lambda_focal
        losses["loss_rtcc_dice"] = _dice_loss(sim, tgt, B) * lambda_dice

    # --- use_mask_loss: second soft-segmentation map, upsampled to obj_mask's resolution ---
    if use_mask_loss and seg_mask is not None and obj_mask is not None:
        src = F.interpolate(seg_mask, size=obj_mask.shape[-2:], mode="bilinear", align_corners=False)
        src = src.flatten(1)
        tgt = obj_mask.flatten(1).float()
        losses["loss_seg_focal"] = _sigmoid_focal_loss(src, tgt, B) * lambda_focal
        losses["loss_seg_dice"] = _dice_loss(src, tgt, B) * lambda_dice

    return losses
