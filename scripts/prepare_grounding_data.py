"""Arrange RefCOCO/+/g visual grounding data into the layout RefCOCODataset expects.

This does NOT download anything from Google Drive — HiVG gates its text-box
annotations behind a manual Drive download (see HiVG/README.md, "Text-Box
Anotations"). You must download `ref_data_shuffled.zip` yourself and pass its
path (or its already-extracted directory) via --annotations.

What this script does:
  1. Downloads MSCOCO train2014 images via HiVG/download_mscoco2014.sh
     (direct cocodataset.org URLs — no manual step needed), unless
     --skip-images is given.
  2. Extracts/copies the manually downloaded annotation archive into
     <split_root>/<dataset>/<dataset>_<split>.pth, validating that the
     unc / unc+ / gref_umd splits RefCOCODataset needs are present.

Usage:
    python scripts/prepare_grounding_data.py \
        --data_root $SCRATCH/grounding_data \
        --annotations /path/to/ref_data_shuffled.zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

_REQUIRED_DATASETS = {
    "unc": ["train", "val", "testA", "testB"],
    "unc+": ["train", "val", "testA", "testB"],
    "gref_umd": ["train", "val", "test"],
}


def _download_images(data_root: str) -> None:
    images_dir = os.path.join(data_root, "other", "images", "mscoco", "images")
    if os.path.isdir(os.path.join(images_dir, "train2014")):
        print(f"[images] train2014 already present under {images_dir}, skipping download")
        return
    os.makedirs(images_dir, exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "..", "HiVG", "download_mscoco2014.sh")
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"Expected {script} — run from the repo root, or download train2014 manually "
            f"into {images_dir}/train2014"
        )
    print(f"[images] downloading MSCOCO train2014 into {images_dir} ...")
    subprocess.run(["bash", script, images_dir], check=True)


def _stage_annotations(annotations_path: str, split_root: str, datasets: list[str]) -> None:
    if os.path.isdir(annotations_path):
        src_dir = annotations_path
    elif zipfile.is_zipfile(annotations_path):
        extract_dir = os.path.join(split_root, "_ref_data_shuffled_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[annotations] extracting {annotations_path} -> {extract_dir}")
        with zipfile.ZipFile(annotations_path) as zf:
            zf.extractall(extract_dir)
        # The archive may nest a single top-level directory.
        entries = os.listdir(extract_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            src_dir = os.path.join(extract_dir, entries[0])
        else:
            src_dir = extract_dir
    else:
        raise ValueError(f"--annotations must be a directory or .zip file, got {annotations_path!r}")

    os.makedirs(split_root, exist_ok=True)
    for dataset in datasets:
        src_ds_dir = os.path.join(src_dir, dataset)
        dst_ds_dir = os.path.join(split_root, dataset)
        if not os.path.isdir(src_ds_dir):
            print(f"[annotations] WARNING: {src_ds_dir} not found in archive, skipping")
            continue
        os.makedirs(dst_ds_dir, exist_ok=True)
        for fname in os.listdir(src_ds_dir):
            shutil.copy2(os.path.join(src_ds_dir, fname), os.path.join(dst_ds_dir, fname))
        print(f"[annotations] staged {dataset} -> {dst_ds_dir}")


def _validate(split_root: str, datasets: list[str]) -> None:
    missing = []
    for dataset in datasets:
        for split in _REQUIRED_DATASETS[dataset]:
            p = os.path.join(split_root, dataset, f"{dataset}_{split}.pth")
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        print("[validate] MISSING index files:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)
    print("[validate] all required .pth index files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True,
                         help="Root dir for images; will contain other/images/mscoco/images/train2014/")
    parser.add_argument("--split_root", default=None,
                         help="Root dir for .pth annotation indices (default: <data_root>/data)")
    parser.add_argument("--annotations", default=None,
                         help="Path to the manually downloaded ref_data_shuffled (.zip or extracted dir)")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", default=list(_REQUIRED_DATASETS), choices=list(_REQUIRED_DATASETS),
        help="Restrict staging/validation to these datasets (default: all). "
             "e.g. --datasets unc to set up RefCOCO only.",
    )
    args = parser.parse_args()

    split_root = args.split_root or os.path.join(args.data_root, "data")

    if not args.skip_images:
        _download_images(args.data_root)
    else:
        print("[images] --skip-images set, not downloading")

    if args.annotations:
        _stage_annotations(args.annotations, split_root, args.datasets)
    else:
        print(
            "[annotations] no --annotations given. Download 'ref_data_shuffled.zip' "
            "from the link in HiVG/README.md ('Text-Box Anotations' section) and "
            "re-run with --annotations /path/to/ref_data_shuffled.zip"
        )

    _validate(split_root, args.datasets)


if __name__ == "__main__":
    main()
