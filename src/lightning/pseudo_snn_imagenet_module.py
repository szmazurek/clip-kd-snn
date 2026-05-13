"""Lightning module for PseudoSNN ImageNet classification.

Wraps a SEWResNet (or any model built on PseudoNeuron) with:
  - Cross-entropy classification loss
  - Optional T-mean penalty: loss += penalty * mean(T across all PseudoNeurons)
    This drives neurons to learn shorter timesteps, reducing inference cost.
  - AdamW + cosine LR schedule with linear warmup (identical to ImageNetClassificationModule)

Usage via scripts/train_imagenet_pseudo_snn.py.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from ..models.visual_encoders.pseudo_neuron import PseudoNeuron
from ..utils.misc import cosine_lr_lambda, exclude_weight_decay


class PseudoSNNImageNetModule(L.LightningModule):
    """LightningModule for PseudoSNN ImageNet classification.

    Args:
        model: SEWResNet with num_classes=1000 (returns plain logits).
        lr: Peak AdamW learning rate.
        weight_decay: Weight decay for non-bias/BN parameters.
        warmup_steps: Linear warmup steps.
        penalty: Coefficient on the T-mean regulariser. Set to 0 to disable.
        compile_snn: If True, wrap model with torch.compile before training.
        compile_mode: torch.compile mode string.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        penalty: float = 1e-2,
        compile_snn: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.penalty = penalty

        if compile_snn:
            model = torch.compile(model, fullgraph=True, mode=compile_mode)
        self.model = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_t_mean(self) -> torch.Tensor:
        """Mean of learned T across all PseudoNeurons in the model."""
        T_list = [
            m.get_timesteps()
            for m in self.model.modules()
            if isinstance(m, PseudoNeuron)
        ]
        if not T_list:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(T_list).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        acc1 = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_acc1", acc1, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.penalty > 0.0:
            t_mean = self._compute_t_mean()
            loss = loss + self.penalty * t_mean
            self.log("train_t_mean", t_mean.detach(), on_step=True, on_epoch=False, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> None:
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        acc1 = (logits.argmax(dim=1) == labels).float().mean()
        top5_preds = logits.topk(5, dim=1).indices
        acc5 = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc5", acc5, on_epoch=True, sync_dist=True)

        # Log average T at validation time (no gradient needed)
        t_vals = [
            m.get_timesteps().item()
            for m in self.model.modules()
            if isinstance(m, PseudoNeuron)
        ]
        if t_vals:
            avg_t = sum(t_vals) / len(t_vals)
            self.log("val_t_mean", avg_t, on_epoch=True, prog_bar=False, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # PseudoNeuron.logit / scale / bias are 0-D or 1-D → caught by exclude_weight_decay
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
