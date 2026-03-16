"""Lightning module for baseline CLIP training (no knowledge distillation)."""

from __future__ import annotations

import math
from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ..losses.clip_loss import CLIPInfoNCELoss
from ..models.factory import build_student_model
from ..utils.distributed import gather_features
from ..utils.misc import cosine_lr_lambda, exclude_weight_decay
from .eval_mixin import ZeroShotEvalMixin


class CLIPModule(ZeroShotEvalMixin, L.LightningModule):
    """PyTorch Lightning module for standard CLIP training.

    Trains a CLIP model from scratch using the InfoNCE contrastive loss.
    No knowledge distillation — use CLIPKDModule for that.

    Args:
        cfg: Full Hydra config. Reads cfg.model, cfg.training, cfg.loss.
        tokenizer: Text tokenizer (stored for evaluation callbacks).
    """

    def __init__(self, cfg: DictConfig, tokenizer: Callable) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.student, self.preprocess_train, self.preprocess_val = build_student_model(
            cfg
        )
        self.loss_fn = CLIPInfoNCELoss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        return self.student(images, texts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, texts = batch
        # distill=False → L2-normalised embeddings
        img_feats, txt_feats, logit_scale = self.student(images, texts)

        # Gather across GPUs when using DDP
        if self.trainer.world_size > 1:
            all_img, all_txt = gather_features(
                img_feats,
                txt_feats,
                local_loss=False,
                gather_with_grad=self.cfg.loss.get("gather_with_grad", False),
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        else:
            all_img, all_txt = img_feats, txt_feats

        N = all_img.shape[0]
        labels = torch.arange(N, device=self.device)

        # Build minimal KDFeatures-like container for the loss
        from ..losses.base import KDFeatures

        features = KDFeatures(
            s_img=all_img,
            s_txt=all_txt,
            s_img_proj=all_img,
            s_txt_proj=all_txt,
            t_img=all_img,  # unused in CLIPInfoNCELoss
            t_txt=all_txt,  # unused in CLIPInfoNCELoss
            s_logit_scale=logit_scale,
            t_logit_scale=logit_scale,  # unused
            labels=labels,
        )
        loss = self.loss_fn(features)
        if not loss.isfinite():
            self.log("train/nan_skip", 1.0, on_step=True, sync_dist=False)
            return None
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def on_after_backward(self) -> None:
        """Clamp logit_scale to ln(100) after backward, before optimizer step."""
        self.student.logit_scale.data.clamp_(0, math.log(100))

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        no_wd, wd_params = exclude_weight_decay(list(self.student.named_parameters()))

        optimizer = torch.optim.AdamW(
            [
                {"params": no_wd, "weight_decay": 0.0},
                {"params": wd_params, "weight_decay": self.cfg.training.weight_decay},
            ],
            lr=self.cfg.training.lr,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
            eps=self.cfg.training.eps,
        )

        total_steps = self.trainer.estimated_stepping_batches
        lr_lambda = cosine_lr_lambda(
            warmup_steps=self.cfg.training.warmup_steps,
            total_steps=total_steps,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
