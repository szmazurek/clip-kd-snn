"""Zero-shot ImageNet classification evaluation.

Supports IN-1K, IN-V2, IN-R, IN-Sketch, and IN-A variants.
Variant-specific class index filtering is applied here (not in the dataset).

Ported from src/training/zero_shot.py run() and zero_shot_eval() (lines 40-119).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .imagenet_zeroshot_data import (
    imagenet_classnames,
    imagenet_a,
    imagenet_r_indices,
    openai_imagenet_template,
)
from .zero_shot_classifier import build_zero_shot_classifier


def _accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    """Compute top-k accuracy counts (not rates) for a batch."""
    pred = output.topk(max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run_zero_shot(
    model: nn.Module,
    classifier: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
    variant: str = "imagenet",
) -> tuple[float, float]:
    """Run zero-shot classification on a single ImageNet variant.

    Ported from src/training/zero_shot.py run() (lines 40-79).

    Args:
        model: open_clip CLIP model with encode_image().
        classifier: (D, num_classes) text classifier from build_zero_shot_classifier().
        dataloader: ImageNetDataset dataloader.
        device: Target device.
        variant: One of "imagenet", "imagenet_v2", "imagenet_r", "imagenet_sketch",
                 "imagenet_a".

    Returns:
        Tuple of (top1_accuracy, top5_accuracy) as fractions in [0, 1].
    """
    imagenet_a_indices = (
        [k for k, v in imagenet_a.items() if v != -1] if variant == "imagenet_a" else None
    )

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0
        for images, target in tqdm(dataloader, desc=f"Eval [{variant}]", leave=False):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ classifier  # (B, num_classes)

            if variant == "imagenet_r":
                logits = logits[:, imagenet_r_indices]
            elif variant == "imagenet_a":
                logits = logits[:, imagenet_a_indices]

            acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    return top1 / n, top5 / n


def evaluate_zero_shot(
    model: nn.Module,
    eval_dataloaders: dict[str, DataLoader],
    tokenizer: Callable,
    device: torch.device,
) -> dict[str, float]:
    """Run zero-shot evaluation on all available ImageNet variant dataloaders.

    Args:
        model: open_clip CLIP model.
        eval_dataloaders: Dict mapping variant name to dataloader.
                          Keys should be subset of "imagenet", "imagenet_v2",
                          "imagenet_r", "imagenet_sketch", "imagenet_a".
        tokenizer: Text tokenizer callable.
        device: Target device.

    Returns:
        Dict of metric name → value (e.g. "imagenet/top1": 0.652).
    """
    imagenet_variants = {
        k: v
        for k, v in eval_dataloaders.items()
        if k.startswith("imagenet")
    }
    if not imagenet_variants:
        return {}

    classifier = build_zero_shot_classifier(
        model=model,
        classnames=imagenet_classnames,
        templates=openai_imagenet_template,
        tokenizer=tokenizer,
        device=device,
    )

    results = {}
    for variant, loader in imagenet_variants.items():
        top1, top5 = run_zero_shot(model, classifier, loader, device, variant=variant)
        results[f"{variant}/top1"] = top1
        results[f"{variant}/top5"] = top5
    return results
