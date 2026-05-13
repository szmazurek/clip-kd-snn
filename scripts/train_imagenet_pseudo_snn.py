"""ImageNet classification training for PseudoSNN SEW-ResNet.

Trains a Spatially-Effective-Weight ResNet (18/34) using PseudoNeuron
activations. During training each spiking neuron is a single-pass
ReLU+noise proxy (no BPTT); at inference it deploys a real IF neuron
for its learned number of timesteps T. A T-mean penalty jointly
optimises the accuracy/latency tradeoff.

Based on scripts/train_imagenet_qkformer.py — same dataset auto-detection
(HuggingFace Parquet, WebDataset, ImageFolder) and Lightning training loop.

Usage:
    # Training only (no validation) — recommended for PseudoSNN
    python scripts/train_imagenet_pseudo_snn.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --model sew_resnet18 --epochs 10 --batch-size 128 \\
        --no-val --save-interval 2

    # Then calibrate and evaluate with:
    python scripts/calibrate_pseudo_snn.py \\
        --checkpoint runs/imagenet_pseudo_snn/last.ckpt \\
        --data-dir /storage/imagenet-full-hf/data \\
        --model sew_resnet18

    # Full SEW-ResNet-34 training 
    python scripts/train_imagenet_pseudo_snn.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --model sew_resnet34 --epochs 100 \\
        --batch-size 128 --penalty 1e-2 --noise-type gaussian \\
        --no-val --output-dir ./runs/pseudo_snn_imagenet
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
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from src.models.visual_encoders.sew_resnet_pseudo import (
    sew_resnet18,
    sew_resnet34,
    sew_resnet50,
)
from src.lightning.pseudo_snn_imagenet_module import PseudoSNNImageNetModule

torch.set_float32_matmul_precision("high")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_TRAIN_SAMPLES = 1_281_167
IMAGENET_VAL_SAMPLES = 50_000

_MODEL_REGISTRY = {
    "sew_resnet18": sew_resnet18,
    "sew_resnet34": sew_resnet34,
    "sew_resnet50": sew_resnet50,
}


# ---------------------------------------------------------------------------
# Format detection (shared with train_imagenet_qkformer.py)
# ---------------------------------------------------------------------------


def _detect_format(data_dir: str) -> str:
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
    def __init__(self, parquet_files, transform=None, shuffle=False, seed=42):
        import pyarrow.parquet as pq

        self.transform = transform
        self._files = sorted(parquet_files)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
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
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(files)

        for fpath in files:
            table = pq.read_table(fpath, columns=["image", "label"])
            rows = table.to_pydict()
            images, labels = rows["image"], rows["label"]
            indices = list(range(len(labels)))
            if self.shuffle:
                rng = random.Random(self.seed + self._epoch + hash(fpath))
                rng.shuffle(indices)
            for i in indices:
                img = Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                yield img, int(labels[i])


def build_parquet_datasets(data_dir, train_tf, val_tf):
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet files in {data_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation-*.parquet files in {data_dir}")
    print(
        f"[data] Parquet: {len(train_files)} train shards, {len(val_files)} val shards"
    )
    return ParquetImageDataset(
        train_files, train_tf, shuffle=True
    ), ParquetImageDataset(val_files, val_tf)


# ---------------------------------------------------------------------------
# WebDataset
# ---------------------------------------------------------------------------


def build_wds_datasets(data_dir, train_tf, val_tf):
    import webdataset as wds

    train_shards = sorted(glob.glob(os.path.join(data_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(data_dir, "val", "*.tar")))
    if not train_shards:
        train_shards = sorted(glob.glob(os.path.join(data_dir, "*.tar")))

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


def build_imagefolder_datasets(data_dir, train_tf, val_tf):
    from torchvision.datasets import ImageFolder

    return (
        ImageFolder(os.path.join(data_dir, "train"), transform=train_tf),
        ImageFolder(os.path.join(data_dir, "val"), transform=val_tf),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PseudoSNN SEW-ResNet on ImageNet"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        required=True,
        help="ImageNet data directory (auto-detects Parquet/WDS/ImageFolder).",
    )
    # Model
    parser.add_argument(
        "--model", default="sew_resnet34", choices=list(_MODEL_REGISTRY.keys())
    )
    parser.add_argument(
        "--connect-f",
        default="ADD",
        choices=["ADD", "AND", "IAND"],
        help="Residual connection function in SEW-ResNet.",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Per-GPU batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    # PseudoSNN neuron
    parser.add_argument(
        "--penalty",
        type=float,
        default=1e-2,
        help="Coefficient on T-mean regulariser (set 0 to disable).",
    )
    parser.add_argument(
        "--noise-type", default="gaussian", choices=["uniform", "gaussian"]
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.5,
        help="Probability of applying quantization/clipping noise per batch.",
    )
    parser.add_argument(
        "--init-T",
        type=float,
        default=8.0,
        help="Initial timesteps for each PseudoNeuron.",
    )
    parser.add_argument("--min-T", type=int, default=1)
    parser.add_argument("--max-T", type=int, default=16)
    # Validation
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip validation entirely.  Use calibrate_pseudo_snn.py afterwards "
        "for a proper calibrated evaluation.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs when --no-val is set.",
    )
    # Compile
    parser.add_argument(
        "--compile-snn", action="store_true", help="Wrap model with torch.compile."
    )
    parser.add_argument("--compile-mode", default="default")
    # Hardware
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed", "16-true"],
    )
    # Output
    parser.add_argument("--output-dir", default="./runs/imagenet_pseudo_snn")
    parser.add_argument(
        "--checkpoint", default=None, help="Resume from checkpoint path."
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 train + 1 val batch then exit (tests full train+inference path).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    neuron_kwargs = dict(
        noise_type=args.noise_type,
        noise_prob=args.noise_prob,
        init_T=args.init_T,
        min_T=args.min_T,
        max_T=args.max_T,
    )
    model_fn = _MODEL_REGISTRY[args.model]
    model = model_fn(num_classes=1000, connect_f=args.connect_f, **neuron_kwargs)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.model}  ({n_params:.1f} M params)")

    lit_model = PseudoSNNImageNetModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        penalty=args.penalty,
        compile_snn=args.compile_snn,
        compile_mode=args.compile_mode,
    )

    # ---- Datasets (auto-detected) ----
    train_tf = build_train_transforms()
    val_tf = build_val_transforms()

    fmt = _detect_format(args.data_dir)
    print(f"[data] Detected format: {fmt}  ({args.data_dir})")

    if args.no_val:
        if fmt == "parquet":
            train_files = sorted(
                glob.glob(os.path.join(args.data_dir, "train-*.parquet"))
            )
            train_ds = ParquetImageDataset(train_files, train_tf, shuffle=True)
        elif fmt == "wds":
            train_ds, _ = build_wds_datasets(args.data_dir, train_tf, val_tf)
        else:
            from torchvision.datasets import ImageFolder

            train_ds = ImageFolder(
                os.path.join(args.data_dir, "train"), transform=train_tf
            )
        val_ds = None
    else:
        if fmt == "parquet":
            train_ds, val_ds = build_parquet_datasets(args.data_dir, train_tf, val_tf)
        elif fmt == "wds":
            train_ds, val_ds = build_wds_datasets(args.data_dir, train_tf, val_tf)
        else:
            train_ds, val_ds = build_imagefolder_datasets(
                args.data_dir, train_tf, val_tf
            )

    from torch.utils.data import IterableDataset

    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(not is_iterable),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0 and not is_iterable),
        prefetch_factor=4,
    )
    val_loader = (
        None
        if val_ds is None
        else DataLoader(
            val_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
        )
    )

    # ---- Trainer ----
    if args.no_val:
        ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.model}-{{epoch:02d}}",
            every_n_epochs=args.save_interval,
            save_last=True,
            save_top_k=-1,
        )
    else:
        ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.model}-{{epoch:02d}}-{{val_acc1:.4f}}",
            monitor="val_acc1",
            mode="max",
            save_top_k=3,
            save_last=True,
        )

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        LearningRateMonitor(logging_interval="step"),
        ckpt_callback,
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices="auto",
        strategy="auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks,
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=50,
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
