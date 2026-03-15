"""ImageNet-1K validation dataset loaded from HuggingFace parquet cache.

The ILSVRC/imagenet-1k dataset cached by HuggingFace is stored as Parquet files
(not ImageFolder), so torchvision.datasets.ImageFolder cannot be used. This module
provides a drop-in replacement that reads from the HF snapshot's parquet files.

Parquet schema:
    image: struct{bytes: binary, path: str}  — JPEG-encoded image bytes
    label: int64                              — class index (0-999)
"""
from __future__ import annotations

import glob as _glob
import io
import os
from typing import Callable, Optional

import datasets as hf_datasets
import torch.utils.data as tud
from PIL import Image

_VARIANTS = {"imagenet", "imagenet_v2", "imagenet_r", "imagenet_sketch", "imagenet_a"}


class ImageNetHFDataset(tud.Dataset):
    """ImageNet validation dataset backed by HuggingFace parquet cache.

    Reads the validation split from the HF snapshot's parquet files and provides
    lazy random access via the Arrow-backed HF Dataset (mmap, no full load into RAM).

    Args:
        hf_cache_dir: Path to the HF hub cache dir for this dataset,
            e.g. ``$SCRATCH/.cache/hub/datasets--ILSVRC--imagenet-1k``.
            The snapshot hash is discovered automatically from ``refs/main``.
        transform: Eval image transforms (from open_clip).
        variant: Dataset variant name (default ``"imagenet"``).
    """

    def __init__(
        self,
        hf_cache_dir: str,
        transform: Optional[Callable] = None,
        variant: str = "imagenet",
    ) -> None:
        assert variant in _VARIANTS, f"Unknown variant '{variant}'. Choose from {_VARIANTS}."
        refs_main = os.path.join(hf_cache_dir, "refs", "main")
        with open(refs_main) as fh:
            snapshot_hash = fh.read().strip()
        data_dir = os.path.join(hf_cache_dir, "snapshots", snapshot_hash, "data")
        parquet_files = sorted(_glob.glob(os.path.join(data_dir, "validation-*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(
                f"No validation parquet files found in {data_dir}. "
                "Ensure the ILSVRC/imagenet-1k validation split is cached."
            )
        # HF converts parquet → Arrow on first use (cached in HF_HOME/datasets/).
        # Subsequent accesses are mmap'd O(1) random access.
        self._ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"validation": parquet_files},
            split="validation",
        )
        self.transform = transform
        self.variant = variant

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        row = self._ds[idx]
        img_data = row["image"]
        if isinstance(img_data, Image.Image):
            img = img_data.convert("RGB")
        elif isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, row["label"]


def build_imagenet_hfd(
    hf_cache_dir: str,
    transform: Callable,
    variant: str = "imagenet",
) -> ImageNetHFDataset:
    """Construct an ImageNetHFDataset from the HF hub cache directory."""
    return ImageNetHFDataset(hf_cache_dir, transform=transform, variant=variant)
