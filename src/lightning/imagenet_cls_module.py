"""Lightning module for standalone ImageNet classification with MSViT.

Used as a sanity check to verify that the SNN backbone can learn at all before
committing to the full CLIP pipeline. Trains MSFormer_10_512 with a standard
1000-class linear head using cross-entropy loss.

Usage (via scripts/train_imagenet_cls.py):
    python scripts/train_imagenet_cls.py \\
        --train-dir /path/to/imagenet/train \\
        --val-dir   /path/to/imagenet/val \\
        --epochs 90 --batch-size 128
"""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from ..utils.misc import cosine_lr_lambda, exclude_weight_decay


class ImageNetClassificationModule(L.LightningModule):
    """LightningModule for MSViT ImageNet classification.

    Args:
        model: A ``hierarchical_spiking_transformer`` with ``num_classes=1000``.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps (step-based, not epoch-based).
        compile_snn: If True, wrap model with torch.compile before training.
        compile_mode: torch.compile mode string (default: "reduce-overhead").
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        compile_snn: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        if compile_snn:
            # Compile the whole model module. Compiling __call__ (not a bound method)
            # handles train→eval mode switching correctly via guard invalidation.
            # "default" mode does kernel fusion without CUDA Graphs.
            model = torch.compile(
                model,
                fullgraph=True,
                mode=compile_mode,
            )
        self.model = model
        self._model_for_reset = model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> None:
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        # Top-1
        acc1 = (logits.argmax(dim=1) == labels).float().mean()
        # Top-5
        top5_preds = logits.topk(5, dim=1).indices  # [B, 5]
        acc5 = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc5", acc5, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        no_wd, wd_params = exclude_weight_decay(list(self.model.named_parameters()))
        optimizer = torch.optim.AdamW(
            [
                {"params": no_wd, "weight_decay": 0.0},
                {"params": wd_params, "weight_decay": self.weight_decay},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        total_steps = self.trainer.estimated_stepping_batches
        lr_lambda = cosine_lr_lambda(
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
