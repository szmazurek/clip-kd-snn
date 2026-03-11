"""Custom Lightning callbacks for CLIP-KD training."""
from __future__ import annotations

import lightning as L


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
