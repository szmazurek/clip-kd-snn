"""Lightning module for standalone ImageNet classification with LoopViT.

Used as a sanity check to verify the LoopViT backbone can learn before
committing to the full CLIP pipeline. Trains LoopViT with a standard
1000-class linear head using cross-entropy loss, plus optional gate
entropy and loop step penalty regularisers.

Usage (via scripts/train_imagenet_loopvit.py):
    python scripts/train_imagenet_loopvit.py \\
        --data-dir /path/to/imagenet \\
        --output-dir ./runs/imagenet_loopvit \\
        --epochs 300 --batch-size 128
"""

from __future__ import annotations

import copy

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..models.visual_encoders.loopvit import (
    LoopForwardMetadata,
    _compute_gate_regularizers,
)


# ---------------------------------------------------------------------------
# EMA helper (ported from original loopvit_imagenet_base.py, DDP-free version)
# ---------------------------------------------------------------------------


class _EMAHelper:
    """Exponential moving average of model parameters.

    Lightning unwraps DDP transparently, so named_parameters() always sees
    the underlying model weights rather than DDP module wrappers.
    """

    def __init__(self, mu: float = 0.999) -> None:
        self.mu = mu
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    def register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name]

    def ema(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def store(self, model: nn.Module) -> None:
        self._backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "backup": self._backup}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict["shadow"]
        self._backup = state_dict.get("backup", {})


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class LoopViTImageNetModule(L.LightningModule):
    """LightningModule for LoopViT ImageNet classification.

    Args:
        model: A ``LoopViT`` with ``num_classes=1000``.
        lr: Peak learning rate.
        min_lr: Minimum LR at the end of the cosine schedule.
        weight_decay: AdamW weight decay applied to non-embedding parameters.
        warmup_epochs: Epochs for the linear warmup phase.
        warmup_start_factor: Initial LR multiplier at epoch 0 of warmup.
        gate_entropy_weight: Weight for gate entropy regularisation loss.
        loop_penalty_weight: Weight for loop step penalty regularisation loss.
        use_dynamic_exit: If True, enable dynamic early exit during training.
        gate_threshold: Exit gate confidence threshold.
        label_smoothing: Label smoothing for cross-entropy.
        use_ema: If True, maintain EMA shadow weights and evaluate with them.
        ema_rate: EMA decay rate (default: 0.999).
        compile_model: If True, wrap model with torch.compile before training.
        compile_mode: torch.compile mode string.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        min_lr: float = 0.0,
        weight_decay: float = 0.05,
        warmup_epochs: int = 30,
        warmup_start_factor: float = 0.033,
        gate_entropy_weight: float = 0.01,
        loop_penalty_weight: float = 0.01,
        use_dynamic_exit: bool = False,
        gate_threshold: float = 0.6,
        label_smoothing: float = 0.1,
        use_ema: bool = False,
        ema_rate: float = 0.999,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.gate_entropy_weight = gate_entropy_weight
        self.loop_penalty_weight = loop_penalty_weight
        self.use_dynamic_exit = use_dynamic_exit
        self.gate_threshold = gate_threshold
        self.label_smoothing = label_smoothing
        self.use_ema = use_ema

        if compile_model:
            # Wrap the encoder loop in a compiled function so all max_loop_steps
            # iterations are captured in one CUDA graph instead of 12 separate
            # dispatches with Python overhead between them.
            # We cannot compile the full model.forward because of the gradient
            # shape mismatch on cls.expand() — see cls token comment above.
            _encoder = model.encoder
            _max_steps = model.max_loop_steps
            _step_embed = model.step_embed  # may be None

            @torch.compile(fullgraph=False, mode=compile_mode)
            def _compiled_loop(x: torch.Tensor) -> torch.Tensor:
                for step in range(_max_steps):
                    if _step_embed is not None:
                        idx = min(step, _step_embed.num_embeddings - 1)
                        x = x + _step_embed.weight[idx].view(1, 1, -1)
                    x = _encoder(x)
                return x

            model._compiled_loop = _compiled_loop
            self._use_compiled_loop = True
        else:
            self._use_compiled_loop = False
        self.model = model

        self._ema: _EMAHelper | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        if self.use_ema:
            self._ema = _EMAHelper(mu=self.hparams.ema_rate)
            self._ema.register(self.model)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # All ranks update EMA independently, but DDP all-reduces gradients before
        # the optimizer step, so all ranks have identical parameters at this point.
        # Therefore all EMA shadows are identical — no cross-rank sync is needed.
        if self._ema is not None:
            self._ema.update(self.model)

    def on_validation_epoch_start(self) -> None:
        if self._ema is not None:
            self._ema.store(self.model)
            self._ema.ema(self.model)
        self._val_correct = torch.zeros(1, dtype=torch.long, device=self.device)
        self._val_correct5 = torch.zeros(1, dtype=torch.long, device=self.device)
        self._val_total = torch.zeros(1, dtype=torch.long, device=self.device)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Persist EMA shadow weights in the checkpoint so the saved file contains
        # the smoothed weights (which produced val_acc1) rather than the noisy
        # training weights that are active at save time.
        if self._ema is not None:
            checkpoint["ema_shadow"] = self._ema.shadow

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema_shadow" in checkpoint and self._ema is not None:
            self._ema.shadow = checkpoint["ema_shadow"]

    def on_train_epoch_start(self) -> None:
        # Restore the real training weights now that any checkpointing from
        # the preceding validation epoch is complete.
        if self._ema is not None and self._ema._backup:
            self._ema.restore(self.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _forward(self, images: torch.Tensor, dynamic_exit=None):
        """Forward that uses the compiled loop when available.

        When compiled, all max_loop_steps encoder calls are captured in one
        CUDA graph instead of being driven by a Python for loop.
        Only valid when use_exit_gate=False and dynamic_exit is off; falls
        back to the regular model.forward otherwise.
        """
        if (
            self._use_compiled_loop
            and not self.use_dynamic_exit
            and dynamic_exit is None
        ):
            hidden = self.model.image_tokens(images)
            hidden = self.model._compiled_loop(hidden)
            logits = self.model.head(self.model.head_norm(hidden)[:, 0, :])
            B = images.shape[0]
            exit_steps = torch.full(
                (B,), self.model.max_loop_steps, dtype=torch.long, device=images.device
            )
            metadata = LoopForwardMetadata(
                gate_probs=[],
                exit_steps=exit_steps,
                max_steps=self.model.max_loop_steps,
            )
            return logits, metadata
        return self.model(
            images, dynamic_exit=dynamic_exit, gate_threshold=self.gate_threshold
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        images, labels = batch
        logits, metadata = self._forward(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        gate_entropy_loss, loop_penalty = _compute_gate_regularizers(
            metadata, disable_exit_gate=False
        )
        if self.gate_entropy_weight > 0:
            loss = loss + self.gate_entropy_weight * gate_entropy_loss
        if self.loop_penalty_weight > 0:
            loss = loss + self.loop_penalty_weight * loop_penalty

        mean_steps = metadata.exit_steps.float().mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_gate_entropy",
            gate_entropy_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_loop_penalty",
            loop_penalty,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_mean_exit_steps",
            mean_steps,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> None:
        images, labels = batch
        # Always run all loop steps at validation — no dynamic exit
        logits, metadata = self.model(images, dynamic_exit=None, gate_threshold=None)
        # Convert soft labels (mixup/cutmix) to hard labels for accuracy counting
        hard_labels = labels.argmax(dim=-1) if labels.ndim == 2 else labels
        loss = F.cross_entropy(logits, hard_labels)

        pred = logits.argmax(dim=1)
        self._val_correct += (pred == hard_labels).sum()
        self._val_correct5 += (
            logits.topk(5, dim=1).indices == hard_labels.unsqueeze(1)
        ).any(dim=1).sum()
        self._val_total += hard_labels.shape[0]

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_mean_exit_steps",
            metadata.exit_steps.float().mean(),
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        correct = self._val_correct.clone()
        correct5 = self._val_correct5.clone()
        total = self._val_total.clone()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct5, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        acc1 = (correct / total.clamp(min=1)).float()
        acc5 = (correct5 / total.clamp(min=1)).float()
        # sync_dist=False — we already all_reduced; log same value on every rank
        # so ModelCheckpoint (rank 0) and progress bars (all ranks) both see it
        self.log("val_acc1", acc1, prog_bar=True, sync_dist=False)
        self.log("val_acc5", acc5, sync_dist=False)

        del self._val_correct, self._val_correct5, self._val_total

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        def _no_decay(name: str, param: nn.Parameter) -> bool:
            return (
                param.ndim < 2
                or "bias" in name
                or "norm" in name.lower()
                or "step_embed" in name.lower()
                or "patch_pos_embed" in name
                or "cls" in name
            )

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _no_decay(name, param):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_epochs = self.trainer.max_epochs
        warmup = LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - self.warmup_epochs, 1),
            eta_min=self.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
