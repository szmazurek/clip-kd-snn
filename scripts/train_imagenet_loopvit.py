"""Standalone ImageNet classification training for LoopViT (sanity-check script).

Trains LoopViT on ImageNet with a standard 1000-class head using
cross-entropy loss plus optional gate entropy and loop step penalty
regularisers. Use this to verify the backbone is trainable before
committing to the full CLIP pipeline.

Dataset formats accepted (auto-detected by inspecting the directory):
  - HuggingFace Parquet: directory containing train-*.parquet / validation-*.parquet
  - WebDataset:          directory containing *.tar shards
  - ImageFolder:         directory of class subdirectories (fallback)

Usage:
    # HuggingFace Parquet (all splits in one data/ dir)
    python scripts/train_imagenet_loopvit.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --output-dir ./runs/imagenet_loopvit

    # WDS shards
    python scripts/train_imagenet_loopvit.py \\
        --data-dir /storage/imagenet-v1-wds-full \\
        --output-dir ./runs/imagenet_loopvit

    # ImageFolder (separate train/val dirs)
    python scripts/train_imagenet_loopvit.py \\
        --data-dir /data/imagenet \\
        --output-dir ./runs/imagenet_loopvit

    # Multi-GPU with dynamic exit and EMA
    python scripts/train_imagenet_loopvit.py \\
        --data-dir /storage/imagenet-full-hf/data \\
        --gpus 4 --batch-size 128 --use-dynamic-exit --use-ema \\
        --output-dir ./runs/imagenet_loopvit_ema
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

torch._dynamo.config.optimize_ddp = False  # avoids compile/DDP gradient-hook conflict

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.models.visual_encoders.loopvit import LoopViT
from src.models.visual_encoders.transformer_utils import get_mixup_cutmix
from src.lightning.loopvit_imagenet_module import LoopViTImageNetModule

# ---- ImageNet normalisation constants ----
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_TRAIN_SAMPLES = 1_281_167
IMAGENET_VAL_SAMPLES = 50_000
torch._dynamo.config.optimize_ddp = False


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
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        import pyarrow.parquet as pq

        self.transform = transform
        self._files = sorted(parquet_files)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.rank = rank
        self.world_size = world_size

        # Read row counts from file footers only (no image data loaded).
        self._file_rows: dict[str, int] = {
            f: pq.read_metadata(f).num_rows for f in self._files
        }
        self._total = sum(self._file_rows.values())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        # Report per-rank count so Lightning's progress bar and scheduler are correct.
        return max(1, self._total // self.world_size)

    def __iter__(self):
        import pyarrow.parquet as pq
        import random

        files = list(self._files)

        # Rank-based shard split — set at construction time (before workers spawn)
        # so it is visible inside DataLoader worker subprocesses, which never join
        # the distributed process group and cannot call dist.get_rank() themselves.
        if self.world_size > 1:
            files = files[self.rank :: self.world_size]

        if not files:
            return

        # Worker-based shard split within this rank.
        rank_target = self._total // self.world_size

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid, nw = worker_info.id, worker_info.num_workers
            files = files[wid::nw]
            # Equal partition: all workers collectively yield exactly rank_target
            # samples. Proportional rounding (round(rank_target * rows/total)) can
            # accumulate errors so the sum across workers differs from rank_target,
            # causing ranks to exhaust their DataLoaders at different steps and
            # deadlocking DDP allreduce.
            target = rank_target // nw + (1 if wid < rank_target % nw else 0)
        else:
            target = rank_target

        if not files:
            return

        # Stream exactly `target` samples, cycling through the file list if the
        # rank's files contain fewer rows (uneven shard sizes) or stopping early
        # if they contain more.  This keeps all ranks at the same step count and
        # prevents DDP gradient-ALLREDUCE hangs when one rank exhausts its data
        # before the others.
        count = 0
        cycle = 0
        while count < target:
            if self.shuffle:
                rng = random.Random(self.seed + self._epoch + cycle)
                rng.shuffle(files)

            for fpath in files:
                table = pq.read_table(fpath, columns=["image", "label"])
                rows = table.to_pydict()
                images = rows["image"]
                labels = rows["label"]
                n = len(labels)

                indices = list(range(n))
                if self.shuffle:
                    rng_f = random.Random(self.seed + self._epoch + cycle + hash(fpath))
                    rng_f.shuffle(indices)

                for i in indices:
                    img_bytes = images[i]["bytes"]
                    label = int(labels[i])
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if self.transform is not None:
                        img = self.transform(img)
                    yield img, label
                    count += 1
                    if count >= target:
                        return

            cycle += 1


def _parquet_files(data_dir: str) -> tuple[list[str], list[str]]:
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet files in {data_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation-*.parquet files in {data_dir}")
    return train_files, val_files


# ---------------------------------------------------------------------------
# WebDataset
# ---------------------------------------------------------------------------


def build_wds_datasets(data_dir: str, train_tf, val_tf):
    import webdataset as wds

    train_shards = sorted(glob.glob(os.path.join(data_dir, "train", "*.tar")))
    val_shards = sorted(glob.glob(os.path.join(data_dir, "val", "*.tar")))
    if not train_shards:
        train_shards = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
    if not val_shards:
        val_shards = train_shards

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
# LightningDataModule
# ---------------------------------------------------------------------------


class ImageNetDataModule(L.LightningDataModule):
    """DataModule for ImageNet classification.

    Dataset construction is deferred to ``setup()``, which Lightning calls
    after DDP is fully initialised. At that point ``self.trainer.global_rank``
    and ``self.trainer.world_size`` are correct, so we can pass them into the
    ``ParquetImageDataset`` constructor. This is the only way to guarantee that
    DataLoader worker subprocesses (which never join the distributed group) see
    the right rank — they cannot call ``dist.get_rank()`` themselves.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None

        # Build augmentation collate once — same across ranks.
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            num_classes=1000,
        )
        if mixup_cutmix:
            import torchvision.transforms.v2 as Tv2

            _aug = (
                Tv2.RandomChoice(mixup_cutmix)
                if len(mixup_cutmix) > 1
                else mixup_cutmix[0]
            )
            self._train_collate = lambda batch: _aug(*default_collate(batch))
        else:
            self._train_collate = default_collate

    def setup(self, stage=None) -> None:
        args = self.args
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        train_tf = build_train_transforms()
        val_tf = build_val_transforms()

        fmt = _detect_format(args.data_dir)
        if self.trainer.global_rank == 0:
            print(f"[data] Detected format: {fmt}  ({args.data_dir})")

        if fmt == "parquet":
            train_files, val_files = _parquet_files(args.data_dir)
            self.train_ds = ParquetImageDataset(
                train_files, train_tf, shuffle=True, rank=rank, world_size=world_size
            )
            self.val_ds = ParquetImageDataset(
                val_files, val_tf, rank=rank, world_size=world_size
            )
        elif fmt == "wds":
            # WDS uses split_by_node internally — no rank args needed.
            self.train_ds, self.val_ds = build_wds_datasets(
                args.data_dir, train_tf, val_tf
            )
        else:
            # ImageFolder: Lightning wraps sampler with DistributedSampler automatically.
            self.train_ds, self.val_ds = build_imagefolder_datasets(
                args.data_dir, train_tf, val_tf
            )

    def train_dataloader(self) -> DataLoader:
        args = self.args
        from torch.utils.data import IterableDataset

        is_iterable = isinstance(self.train_ds, IterableDataset)
        return DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=(not is_iterable),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.workers > 0 and not is_iterable),
            collate_fn=self._train_collate,
        )

    def val_dataloader(self) -> DataLoader:
        args = self.args
        return DataLoader(
            self.val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoopViT on ImageNet classification"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        default="/net/storage/pr3/plgrid/plggwie/plgmazurekagh/imagenet-full-hf/data",
        help="ImageNet data directory. Auto-detects format: "
        "HF Parquet (train-*.parquet), WDS (*.tar), or ImageFolder. "
        "Defaults to the HuggingFace Parquet shards on storage (fewer LUSTRE opens).",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Per-GPU train batch size"
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Per-GPU val batch size. Defaults to --batch-size (same shape avoids CUDA graph recapture).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch-size * gpus * grad-accum",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--warmup-start-factor", type=float, default=0.033)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--grad-clip", type=float, default=0.0, help="Max gradient norm (0 = disabled)"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction or count of training batches per epoch (1.0 = full, 200 = debug)",
    )
    # LoopViT architecture
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--loop-core-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-loop-steps", type=int, default=8)
    parser.add_argument("--min-loop-steps", type=int, default=1)
    parser.add_argument(
        "--no-step-embeddings",
        action="store_true",
        help="Disable learned step embeddings (enabled by default)",
    )
    parser.add_argument(
        "--no-exit-gate",
        action="store_true",
        help="Disable the learned exit gate (enabled by default)",
    )
    parser.add_argument(
        "--swiglu", action="store_true", help="Use SwiGLU instead of GELU"
    )

    # Gate / loop regularisation
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--loop-penalty-weight", type=float, default=0.01)
    parser.add_argument(
        "--use-dynamic-exit",
        action="store_true",
        help="Enable dynamic early exit during training",
    )
    parser.add_argument("--gate-threshold", type=float, default=0.6)

    # Augmentation
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)

    # EMA
    parser.add_argument(
        "--use-ema", action="store_true", help="Maintain EMA shadow weights"
    )
    parser.add_argument("--ema-rate", type=float, default=0.999)

    # Compile
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap model with torch.compile",
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
    parser.add_argument("--output-dir", default="./runs/imagenet_loopvit")
    parser.add_argument(
        "--checkpoint", default=None, help="Resume from checkpoint path"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size  # same shape → no CUDA graph recapture
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    model = LoopViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        loop_core_depth=args.loop_core_depth,
        max_loop_steps=args.max_loop_steps,
        min_loop_steps=args.min_loop_steps,
        add_step_embeddings=not args.no_step_embeddings,
        use_exit_gate=not args.no_exit_gate,
        gate_threshod=args.gate_threshold,  # note: typo in LoopViT API — must match exactly
        swiglu=args.swiglu,
    )

    lit_model = LoopViTImageNetModule(
        model=model,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        gate_entropy_weight=args.gate_entropy_weight,
        loop_penalty_weight=args.loop_penalty_weight,
        use_dynamic_exit=args.use_dynamic_exit,
        gate_threshold=args.gate_threshold,
        label_smoothing=args.label_smoothing,
        use_ema=args.use_ema,
        ema_rate=args.ema_rate,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )

    # ---- DataModule (DDP-aware dataset construction happens inside setup()) ----
    data_module = ImageNetDataModule(args)

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="loopvit-{epoch:02d}-{val_acc1:.4f}",
                monitor="val_acc1",
                mode="max",
                save_top_k=3,
                save_last=True,
            ),
        ],
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=50,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
    )

    trainer.fit(lit_model, data_module, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
