"""CIFAR-100 classification training for PseudoSNN ResNet18.

Trains a standard ResNet18 (CIFAR-adapted stem, no max-pool) with PseudoNeuron
activations on CIFAR-100. Faster to iterate than ImageNet — useful for checking
whether PseudoSNN spike-mode accuracy rises during training.

Usage:
    # AdamW (default), with validation every epoch:
    python scripts/train_cifar100_pseudo_snn.py \\
        --data-dir ./data \\
        --epochs 200 --batch-size 128

    # SGD (closer to original paper):
    python scripts/train_cifar100_pseudo_snn.py \\
        --data-dir ./data \\
        --epochs 200 --batch-size 128 \\
        --optimizer sgd --lr 0.1 --weight-decay 1e-4

    # Resume from checkpoint:
    python scripts/train_cifar100_pseudo_snn.py \\
        --data-dir ./data --checkpoint runs/cifar100_pseudo_snn/last.ckpt

    # Quick smoke-test (1 train + 1 val batch):
    python scripts/train_cifar100_pseudo_snn.py \\
        --data-dir ./data --fast-dev-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from src.models.visual_encoders.resnet_pseudo_cifar import resnet18_pseudo_cifar
from src.lightning.pseudo_snn_imagenet_module import PseudoSNNImageNetModule

torch.set_float32_matmul_precision("high")

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_transforms():
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    val_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    return train_tf, val_tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PseudoSNN ResNet18 on CIFAR-100"
    )
    # Data
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root directory for CIFAR-100 (downloaded automatically).",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    # Optimizer
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Peak LR (use ~0.1 for SGD, ~1e-3 for AdamW).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay (use ~1e-4 for SGD, ~0.05 for AdamW).",
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--no-sgd-nesterov", action="store_true")
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping (SGD only).",
    )
    # PseudoSNN neuron
    parser.add_argument(
        "--penalty", type=float, default=1e-3, help="T-mean penalty coefficient."
    )
    parser.add_argument(
        "--noise-type", default="gaussian", choices=["uniform", "gaussian"]
    )
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--init-T", type=float, default=8.0)
    parser.add_argument("--min-T", type=int, default=1)
    parser.add_argument("--max-T", type=int, default=16)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Initial value of the learnable scale parameter in each PseudoNeuron.",
    )
    # Hardware / precision
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed", "16-true"],
    )
    # Output
    parser.add_argument("--output-dir", default="./runs/cifar100_pseudo_snn")
    parser.add_argument("--checkpoint", default=None, help="Resume from .ckpt path.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Checkpoint every N epochs (in addition to best val).",
    )
    parser.add_argument("--fast-dev-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    neuron_kwargs = dict(
        noise_type=args.noise_type,
        noise_prob=args.noise_prob,
        init_T=args.init_T,
        min_T=args.min_T,
        max_T=args.max_T,
        scale=args.scale,
    )
    model = resnet18_pseudo_cifar(num_classes=100, **neuron_kwargs)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] resnet18_pseudo_cifar  ({n_params:.1f} M params)")

    lit_model = PseudoSNNImageNetModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        penalty=args.penalty,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        sgd_nesterov=(not args.no_sgd_nesterov),
    )

    # ---- Datasets ----
    train_tf, val_tf = build_transforms()
    train_ds = CIFAR100(args.data_dir, train=True, transform=train_tf, download=True)
    val_ds = CIFAR100(args.data_dir, train=False, transform=val_tf, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    # ---- Callbacks ----
    best_ckpt = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="resnet18-{epoch:03d}-{val_acc1:.4f}",
        monitor="val_acc1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="resnet18-epoch={epoch:03d}",
        every_n_epochs=args.save_interval,
        save_top_k=-1,
    )
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval="step"),
        best_ckpt,
        periodic_ckpt,
    ]

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices="auto",
        strategy="auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks,
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=20,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
