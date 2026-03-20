"""ImageNet evaluation datasets loaded from WebDataset (WDS) shards.

Format produced by clip_benchmark: shards contain sXXXXXXX.cls (ASCII label)
and sXXXXXXX.webp (image). Used for imagenet, imagenet_v2, imagenet_r,
imagenet_sketch zero-shot evaluation.
"""
from __future__ import annotations

import glob
import os
from typing import Callable

import webdataset as wds

from .cc3m_wds import SizedWebDataset

_VARIANTS = {"imagenet", "imagenet_v2", "imagenet_r", "imagenet_sketch"}

_VARIANT_SIZES: dict[str, int] = {
    "imagenet":        37_200,
    "imagenet_v2":     10_000,
    "imagenet_r":      30_000,
    "imagenet_sketch": 30_100,
}


def build_imagenet_wds(
    wds_dir: str,
    transform: Callable,
    variant: str = "imagenet",
) -> SizedWebDataset:
    """Return a WebDataset iterable for an ImageNet WDS split directory.

    Args:
        wds_dir: Path to the split directory (e.g. .../imagenet-v1-wds/test/)
                 containing 0.tar, 1.tar, ... shards.
        transform: Eval image transform.
        variant: Variant name (used as metadata; actual class filtering done in
                 eval_mixin and imagenet_eval).
    """
    assert variant in _VARIANTS, f"Unknown variant '{variant}'. Choose from {_VARIANTS}."
    shards = sorted(glob.glob(os.path.join(wds_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir}")

    n_samples = _VARIANT_SIZES[variant]
    pipeline = (
        wds.WebDataset(shards, shardshuffle=False, nodesplitter=wds.split_by_node, empty_check=False)
        .decode("pil")
        .to_tuple("jpg;webp", "cls")
        .map_tuple(transform, lambda x: x)
    )
    pipeline.nsamples = n_samples
    sized = SizedWebDataset(pipeline)
    sized.variant = variant
    return sized
