"""Image transform helpers.

Delegates to open_clip.image_transform so that preprocessing is consistent
with the pretrained teacher's preprocessing pipeline.
"""
from __future__ import annotations

from typing import Callable

import open_clip


def get_train_transforms(image_size: int = 224) -> Callable:
    """Return training image transforms (random crop, flip, normalise).

    Args:
        image_size: Target spatial resolution.

    Returns:
        Callable transform pipeline.
    """
    return open_clip.image_transform(image_size, is_train=True)


def get_eval_transforms(image_size: int = 224) -> Callable:
    """Return evaluation image transforms (centre crop, normalise).

    Args:
        image_size: Target spatial resolution.

    Returns:
        Callable transform pipeline.
    """
    return open_clip.image_transform(image_size, is_train=False)
