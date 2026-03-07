"""Custom Lightning callbacks for CLIP-KD training."""
from __future__ import annotations

import math

import lightning as L
import torch
from torch.utils.data import DataLoader

from ..evaluation.imagenet_eval import evaluate_zero_shot
from ..evaluation.retrieval_eval import evaluate_retrieval


class ZeroShotEvalCallback(L.Callback):
    """Run zero-shot ImageNet classification and retrieval eval every N epochs.

    Calls evaluate_zero_shot() and evaluate_retrieval() on the val dataloaders
    stored in the LightningModule (populated from CLIPDataModule).

    Args:
        frequency: Evaluate every this many epochs (default: 2).
    """

    def __init__(self, frequency: int = 2) -> None:
        super().__init__()
        self.frequency = frequency

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch % self.frequency != 0 and epoch != trainer.max_epochs - 1:
            return

        model = pl_module.student
        model.eval()
        device = pl_module.device
        tokenizer = pl_module.tokenizer
        datamodule = trainer.datamodule

        if datamodule is None or not hasattr(datamodule, "val_datasets"):
            return

        val_loaders = {
            name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
            for name, ds in datamodule.val_datasets.items()
        }

        with torch.no_grad():
            # Zero-shot ImageNet
            imagenet_metrics = evaluate_zero_shot(
                model=model,
                eval_dataloaders=val_loaders,
                tokenizer=tokenizer,
                device=device,
            )
            # Retrieval
            retrieval_metrics = evaluate_retrieval(
                model=model,
                eval_dataloaders=val_loaders,
                device=device,
            )

        all_metrics = {**imagenet_metrics, **retrieval_metrics}
        for name, val in all_metrics.items():
            pl_module.log(f"val/{name}", val, on_epoch=True, sync_dist=True)

        model.train()


class LogitScaleMonitor(L.Callback):
    """Log student logit_scale after each training step."""

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        scale = pl_module.student.logit_scale.exp().item()
        pl_module.log("train/logit_scale", scale, on_step=True, on_epoch=False)
