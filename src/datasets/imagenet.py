"""ImageNet dataset wrapper supporting IN-1K, IN-V2, IN-R, and IN-Sketch.

Uses torchvision.datasets.ImageFolder for straightforward loading.
Variant-specific class filtering (ImageNet-R, ImageNet-A) is handled in
the evaluation code, not here.
"""
from __future__ import annotations

from typing import Callable

import torchvision.datasets as tvd


_VARIANTS = {"imagenet", "imagenet_v2", "imagenet_r", "imagenet_sketch", "imagenet_a"}


class ImageNetDataset(tvd.ImageFolder):
    """ImageNet validation dataset for zero-shot classification.

    Wraps torchvision.datasets.ImageFolder; the variant is passed as a
    metadata attribute so callers can apply variant-specific index filtering.

    Args:
        root: Root directory of the ImageNet split (e.g. /data/imagenet/val).
        transform: Evaluation image transform.
        variant: Dataset variant name (one of "imagenet", "imagenet_v2",
                 "imagenet_r", "imagenet_sketch", "imagenet_a").
    """

    def __init__(
        self,
        root: str,
        transform: Callable,
        variant: str = "imagenet",
    ) -> None:
        assert variant in _VARIANTS, f"Unknown variant '{variant}'. Choose from {_VARIANTS}."
        super().__init__(root, transform=transform)
        self.variant = variant
