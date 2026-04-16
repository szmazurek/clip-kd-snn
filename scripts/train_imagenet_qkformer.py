"""Standalone ImageNet classification training for QKFormer (sanity-check script).

Trains QKFormer_10_512 on ImageNet with a standard 1000-class head using
cross-entropy loss. Use this to verify the SNN backbone is trainable before
committing to the full CLIP pipeline.

Dataset formats accepted (auto-detected by inspecting the directory):
  - HuggingFace Parquet: directory containing train-*.parquet / validation-*.parquet
  - WebDataset:          directory containing *.tar shards
  - ImageFolder:         directory of class subdirectories (fallback)

Usage:
    # HuggingFace Parquet (all splits in one data/ dir)
    python scripts/train_imagenet_qkformer.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --output-dir ./runs/imagenet_qkformer

    # WDS shards
    python scripts/train_imagenet_qkformer.py \\
        --data-dir /storage/imagenet-v1-wds-full \\
        --output-dir ./runs/imagenet_qkformer

    # ImageFolder (separate train/val dirs)
    python scripts/train_imagenet_qkformer.py \\
        --data-dir /data/imagenet \\
        --output-dir ./runs/imagenet_qkformer

    # Multi-GPU + SNN compile
    python scripts/train_imagenet_qkformer.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --gpus 4 --batch-size 128 --compile-snn \\
        --output-dir ./runs/imagenet_qkformer_compiled
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

torch._dynamo.config.assume_static_by_default = True
from src.models.visual_encoders.qkformer import QKFormer_10_512, SNNParams
from src.lightning.imagenet_cls_module import ImageNetClassificationModule

# ---- ImageNet normalisation constants ----
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_TRAIN_SAMPLES = 1_281_167
IMAGENET_VAL_SAMPLES = 50_000


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(data_dir: str) -> str:
    """Return 'parquet', 'wds', or 'imagefolder'."""
    if glob.glob(os.path.join(data_dir, "train-*.parquet")):
        return "parquet"
    if glob.glob(os.path.join(data_dir, "*.tar")):
        return "wds"
    return "imagefolder"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_val_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# HuggingFace Parquet dataset
# ---------------------------------------------------------------------------


class ParquetImageDataset(torch.utils.data.IterableDataset):
    """Iterable dataset over HuggingFace-format ImageNet Parquet shards.

    Each row: {'image': {'bytes': <jpeg bytes>, 'path': <str>}, 'label': <int>}

    Each DataLoader worker owns a disjoint slice of shards and streams through
    them one at a time. Memory footprint: at most one shard (~400 MB) per worker,
    regardless of num_workers — no global random-access cache.

    With shuffle=True the shard order is randomised each epoch and rows within
    each loaded shard are shuffled before yielding.
    """

    def __init__(
        self,
        parquet_files: list[str],
        transform=None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        import pyarrow.parquet as pq

        self.transform = transform
        self._files = sorted(parquet_files)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0  # incremented externally or via set_epoch()

        # Read row counts from file footers only (no image data loaded).
        self._total = sum(pq.read_metadata(f).num_rows for f in self._files)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        import pyarrow.parquet as pq
        import random

        worker_info = torch.utils.data.get_worker_info()
        files = list(self._files)

        # Split shards across workers
        if worker_info is not None:
            wid, nw = worker_info.id, worker_info.num_workers
            files = files[wid::nw]

        # Per-epoch shard shuffle (deterministic across ranks with seed + epoch)
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(files)

        for fpath in files:
            table = pq.read_table(fpath, columns=["image", "label"])
            rows = table.to_pydict()
            images = rows["image"]
            labels = rows["label"]
            n = len(labels)

            indices = list(range(n))
            if self.shuffle:
                rng = random.Random(self.seed + self._epoch + hash(fpath))
                rng.shuffle(indices)

            for i in indices:
                img_bytes = images[i]["bytes"]
                label = int(labels[i])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                yield img, label


def build_parquet_datasets(data_dir: str, train_tf, val_tf):
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet files in {data_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation-*.parquet files in {data_dir}")
    print(
        f"[data] Parquet: {len(train_files)} train shards, {len(val_files)} val shards"
    )
    return ParquetImageDataset(train_files, train_tf), ParquetImageDataset(
        val_files, val_tf
    )


# ---------------------------------------------------------------------------
# WebDataset
# ---------------------------------------------------------------------------


def build_wds_datasets(data_dir: str, train_tf, val_tf):
    import webdataset as wds

    train_shards = sorted(glob.glob(os.path.join(data_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(data_dir, "val", "*.tar")))
    # Fallback: flat directory layout
    if not train_shards:
        train_shards = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
    if not val_shards:
        val_shards = train_shards  # handled by split convention below

    def decode_cls(cls_raw):
        return int(cls_raw.decode() if isinstance(cls_raw, bytes) else cls_raw)

    train_ds = (
        wds.WebDataset(train_shards, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg;webp", "cls")
        .map_tuple(train_tf, decode_cls)
        .with_epoch(IMAGENET_TRAIN_SAMPLES)
    )
    val_ds = (
        wds.WebDataset(
            val_shards,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple("jpg;webp", "cls")
        .map_tuple(val_tf, decode_cls)
        .with_epoch(IMAGENET_VAL_SAMPLES)
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# ImageFolder
# ---------------------------------------------------------------------------


def build_imagefolder_datasets(data_dir: str, train_tf, val_tf):
    from torchvision.datasets import ImageFolder

    train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QKFormer on ImageNet classification"
    )
    # Data — single --data-dir covers all formats
    parser.add_argument(
        "--data-dir",
        required=True,
        help="ImageNet data directory. Auto-detects format: "
        "HF Parquet (train-*.parquet), WDS (*.tar), or ImageFolder.",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Per-GPU batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    # SNN
    parser.add_argument("--T", type=int, default=4, help="SNN timesteps")
    parser.add_argument(
        "--neuron-type", default="lif", choices=["lif", "plif", "nlif", "glif"]
    )
    parser.add_argument(
        "--backend", default="torch", choices=["torch", "triton", "cupy"]
    )
    # Compile
    parser.add_argument(
        "--compile-snn",
        action="store_true",
        help="Wrap SNN forward_features with torch.compile",
    )
    parser.add_argument("--compile-mode", default="default")
    # Hardware
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed", "16-true"],
    )
    # Output
    parser.add_argument("--output-dir", default="./runs/imagenet_qkformer")
    parser.add_argument(
        "--checkpoint", default=None, help="Resume from checkpoint path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    snn_params = SNNParams(
        neuron_type=args.neuron_type,
        tau=2.0,
        v_threshold=1.0,
        backend=args.backend,
    )
    model = QKFormer_10_512(T=args.T, snn=snn_params, num_classes=1000)

    lit_model = ImageNetClassificationModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        compile_snn=args.compile_snn,
        compile_mode=args.compile_mode,
    )

    # ---- Datasets (auto-detected) ----
    train_tf = build_train_transforms()
    val_tf = build_val_transforms()

    fmt = _detect_format(args.data_dir)
    print(f"[data] Detected format: {fmt}  ({args.data_dir})")

    if fmt == "parquet":
        train_ds, val_ds = build_parquet_datasets(args.data_dir, train_tf, val_tf)
    elif fmt == "wds":
        train_ds, val_ds = build_wds_datasets(args.data_dir, train_tf, val_tf)
    else:
        train_ds, val_ds = build_imagefolder_datasets(args.data_dir, train_tf, val_tf)

    from torch.utils.data import IterableDataset

    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(not is_iterable),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0 and not is_iterable),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="qkformer-{epoch:02d}-{val_acc1:.4f}",
                monitor="val_acc1",
                mode="max",
                save_top_k=3,
                save_last=True,
            ),
        ],
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
