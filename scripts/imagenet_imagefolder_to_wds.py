"""Convert an ImageNet-compatible ImageFolder dataset to WebDataset shards.

Handles any dataset stored in the standard ImageFolder layout:
    <src>/
        n01440764/  ← synset ID (or any class name)
            img_0.JPEG
            img_1.JPEG
            ...
        n01443537/
            ...

The synset subdirectory names are sorted alphabetically by torchvision's
ImageFolder, which matches the standard ImageNet-1K class ordering
(class index 0 = n01440764 tench, ..., 999 = n15075141 toilet paper).
No external label file is needed.

Output shard layout (matches build_imagenet_wds expectations):
    <out>/0.tar, 1.tar, ...
    <out>/nshards.txt

Each sample in a shard:
    <key>.jpg   raw image bytes (no re-encoding)
    <key>.cls   ASCII label integer, e.g. b"42"

Usage:
    # Convert ImageNet-Sketch (~50 889 images → 11 shards)
    python scripts/imagenet_imagefolder_to_wds.py \\
        --src /net/storage/pr3/plgrid/plggwie/plgmazurekagh/sketch \\
        --out /net/storage/pr3/plgrid/plggwie/plgmazurekagh/imagenet-sketch-wds-full

    # Override shard size
    python scripts/imagenet_imagefolder_to_wds.py \\
        --src /path/to/imagenet/val \\
        --out /path/to/imagenet-val-wds \\
        --shard-size 5000
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
from pathlib import Path

from torchvision.datasets import ImageFolder
from tqdm import tqdm


def write_shards(src_dir: str, out_dir: str, shard_size: int) -> None:
    # ImageFolder sorts class names (synset IDs) alphabetically → standard
    # ImageNet 0-999 index ordering, no external mapping needed.
    dataset = ImageFolder(src_dir)
    total = len(dataset)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    samples_in_shard = 0
    total_written = 0
    current_tar: tarfile.TarFile | None = None

    def _open_shard(idx: int) -> tarfile.TarFile:
        return tarfile.open(os.path.join(out_dir, f"{idx}.tar"), "w")

    current_tar = _open_shard(shard_idx)

    with tqdm(total=total, desc="Converting", unit="img") as pbar:
        for global_idx, (img_path, label) in enumerate(dataset.samples):
            if samples_in_shard >= shard_size:
                current_tar.close()
                shard_idx += 1
                current_tar = _open_shard(shard_idx)
                samples_in_shard = 0

            key = f"{global_idx:08d}"

            # Image — read raw bytes, no re-encoding
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_buf = io.BytesIO(img_bytes)
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(img_bytes)
            current_tar.addfile(info, img_buf)

            # Label — ASCII integer
            cls_bytes = str(label).encode()
            cls_buf = io.BytesIO(cls_bytes)
            info = tarfile.TarInfo(name=f"{key}.cls")
            info.size = len(cls_bytes)
            current_tar.addfile(info, cls_buf)

            samples_in_shard += 1
            total_written += 1
            pbar.update(1)

    if current_tar is not None:
        current_tar.close()

    n_shards = shard_idx + 1
    with open(os.path.join(out_dir, "nshards.txt"), "w") as f:
        f.write(str(n_shards))

    print(f"\nDone. {total_written} images → {n_shards} shards in {out_dir}")
    print(f"Classes: {len(dataset.classes)}  "
          f"(index 0 = '{dataset.classes[0]}', "
          f"index {len(dataset.classes)-1} = '{dataset.classes[-1]}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ImageFolder dataset → WebDataset shards"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Root directory of the ImageFolder dataset",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for WDS shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Images per shard (default: 5000)",
    )
    args = parser.parse_args()
    write_shards(args.src, args.out, args.shard_size)


if __name__ == "__main__":
    main()
