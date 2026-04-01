"""Convert HuggingFace ImageNet-1K parquet validation split to WebDataset shards.

Reads the parquet files downloaded with:
    huggingface-cli download --repo-type dataset ILSVRC/imagenet-1k \
        --local-dir <src_dir> --include "data/val*"

Each parquet row has:
    image  — dict with 'bytes' (raw JPEG bytes) and 'path'
    label  — int in [0, 999]

Output shard layout (matching the existing imagenet-v1-wds format):
    <out_dir>/0.tar, 1.tar, ...
    <out_dir>/nshards.txt

Each tar sample:
    <key>.jpg  — raw JPEG bytes (no re-encoding)
    <key>.cls  — ASCII label integer, e.g. b"42"

Usage:
    python scripts/imagenet_parquet_to_wds.py \
        --src  /path/to/imagenet-v1-ilsvrc/data \
        --out  /path/to/imagenet-v1-wds-full \
        --shard-size 5000
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import tarfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def write_shards(
    parquet_dir: str,
    out_dir: str,
    shard_size: int,
) -> None:
    parquet_files = sorted(
        glob.glob(os.path.join(parquet_dir, "validation-*.parquet"))
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No validation-*.parquet files found in {parquet_dir}"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    sample_idx = 0  # global sample counter for unique keys
    current_tar: tarfile.TarFile | None = None
    samples_in_current_shard = 0
    total_written = 0

    def _open_shard(idx: int) -> tarfile.TarFile:
        path = os.path.join(out_dir, f"{idx}.tar")
        return tarfile.open(path, "w")

    def _write_sample(
        tf: tarfile.TarFile,
        key: str,
        image_bytes: bytes,
        label: int,
    ) -> None:
        # Write JPEG
        img_buf = io.BytesIO(image_bytes)
        info = tarfile.TarInfo(name=f"{key}.jpg")
        info.size = len(image_bytes)
        img_buf.seek(0)
        tf.addfile(info, img_buf)

        # Write label as ASCII integer
        cls_bytes = str(label).encode()
        cls_buf = io.BytesIO(cls_bytes)
        info = tarfile.TarInfo(name=f"{key}.cls")
        info.size = len(cls_bytes)
        tf.addfile(info, cls_buf)

    # Count total for progress bar
    total = sum(
        len(pd.read_parquet(f, columns=["label"])) for f in parquet_files
    )

    current_tar = _open_shard(shard_idx)

    with tqdm(total=total, desc="Converting", unit="img") as pbar:
        for parquet_path in parquet_files:
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                if samples_in_current_shard >= shard_size:
                    current_tar.close()
                    shard_idx += 1
                    current_tar = _open_shard(shard_idx)
                    samples_in_current_shard = 0

                key = f"{sample_idx:08d}"
                _write_sample(
                    current_tar,
                    key,
                    row["image"]["bytes"],
                    int(row["label"]),
                )
                sample_idx += 1
                samples_in_current_shard += 1
                total_written += 1
                pbar.update(1)

    if current_tar is not None:
        current_tar.close()

    n_shards = shard_idx + 1
    with open(os.path.join(out_dir, "nshards.txt"), "w") as f:
        f.write(str(n_shards))

    print(
        f"\nDone. {total_written} images → {n_shards} shards in {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ImageNet-1K parquet → WebDataset shards"
    )
    parser.add_argument(
        "--src",
        default="/net/storage/pr3/plgrid/plggwie/plgmazurekagh/imagenet-v1-ilsvrc/data",
        help="Directory containing validation-*.parquet files",
    )
    parser.add_argument(
        "--out",
        default="/net/storage/pr3/plgrid/plggwie/plgmazurekagh/imagenet-v1-wds-full",
        help="Output directory for WDS shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Images per shard (default: 5000 → 10 shards for 50K images)",
    )
    args = parser.parse_args()

    write_shards(
        parquet_dir=args.src,
        out_dir=args.out,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
