"""CC12M Arrow-format loader (HuggingFace datasets, Lustre-safe).

Requires one-time conversion before first use:
  python scripts/convert_wds_to_hf.py --dataset cc12m \\
      --hub-cache $SCRATCH/.cache/hub \\
      --output-dir $SCRATCH/cc12m-hf --num-shards 512

After conversion, datasets.load_from_disk() mmap's the Arrow shard files.
See cc3m_hfd.py for full rationale.

Returns a map-style PyTorch Dataset yielding (image_tensor, token_tensor) pairs.
"""
from __future__ import annotations

import io
import sys
from typing import Callable

import datasets as hf_datasets
import torch.utils.data as tud
from PIL import Image, PngImagePlugin

# Raise Pillow's PNG tEXt decompression limit so images with large metadata
# are loaded rather than skipped.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100 MB

CC12M_TRAIN_SAMPLES = 10_968_539


class CC12MHFDataset(tud.Dataset):
    """Map-style PyTorch Dataset wrapping an Arrow-cached HuggingFace Dataset.

    Args:
        hf_ds: datasets.Dataset loaded via load_from_disk() (Arrow-backed).
        transforms: Image preprocessing callable (accepts PIL.Image.RGB).
        tokenizer: Text tokenizer callable; must accept list[str] and return
            a tensor of shape (1, context_length).
    """

    def __init__(
        self,
        hf_ds: hf_datasets.Dataset,
        transforms: Callable,
        tokenizer: Callable,
    ) -> None:
        self._ds = hf_ds
        self.transforms = transforms
        self.tokenizer  = tokenizer

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        for offset in range(len(self)):
            try:
                row = self._ds[(idx + offset) % len(self)]
                img = row["jpg"]
                if isinstance(img, dict):          # HF Image struct: {"bytes": b"...", "path": "..."}
                    img = Image.open(io.BytesIO(img["bytes"]))
                elif not isinstance(img, Image.Image):
                    img = Image.open(io.BytesIO(img))
                img = img.convert("RGB")
                return self.transforms(img), self.tokenizer([str(row["txt"])])[0]
            except Exception as exc:
                if offset == 0:
                    print(
                        f"[hfd/cc12m] Skipping corrupt sample idx={idx}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        raise RuntimeError(f"[hfd/cc12m] No valid sample found starting from idx={idx}")


def build_cc12m_hfd(
    arrow_dir: str,
    transforms: Callable,
    tokenizer: Callable,
) -> CC12MHFDataset:
    """Load Arrow-cached CC12M from a directory created by convert_wds_to_hf.py.

    Args:
        arrow_dir: Path to the directory written by convert_wds_to_hf.py
            (contains Arrow shard files + dataset_info.json).
        transforms: Image preprocessing callable (open_clip train transforms).
        tokenizer: Text tokenizer callable.

    Returns:
        CC12MHFDataset — map-style Dataset with __len__ defined.
        Lightning's DistributedSampler handles DDP sharding automatically.
        Do NOT add a custom sampler.
    """
    ds = hf_datasets.load_from_disk(arrow_dir)
    return CC12MHFDataset(ds, transforms, tokenizer)
