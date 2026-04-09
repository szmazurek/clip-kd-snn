"""Lightning module for CLIP-KD (knowledge distillation) training.

The student is trained to match the teacher via a configurable composite
loss (FD, CKD, ICL, GD, AFD). The teacher is frozen throughout.

Key design choices (matching src/training/main_kd.py):
  - Teacher checkpoint loaded in setup() after Lightning moves models to device.
  - Projection heads (visual_proj, text_proj) live here, not in FDLoss.
  - logit_scale clamping in on_after_backward().
  - AdamW with three param groups: no-wd, wd, loss+proj.
  - Step-based cosine LR with linear warmup.
  - MFD: controlled by cfg.training.mask_ratio > 0 (fixes encode_image bug).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ..losses.base import KDFeatures
from ..losses.factory import build_loss
from ..models.factory import build_student_model, build_teacher_model, get_embed_dim
from ..utils.distributed import gather_features
from ..utils.misc import cosine_lr_lambda, exclude_weight_decay
from .eval_mixin import ZeroShotEvalMixin


class CLIPKDModule(ZeroShotEvalMixin, L.LightningModule):
    """CLIP knowledge distillation Lightning module.

    Args:
        cfg: Full Hydra config. Reads cfg.model, cfg.training, cfg.loss.
        tokenizer: Text tokenizer (stored for evaluation callbacks).
    """

    def __init__(self, cfg: DictConfig, tokenizer: Callable) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "cfg"])
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Build student (trainable)
        self.student, self.preprocess_train, self.preprocess_val = build_student_model(cfg)

        # Build teacher architecture (weights loaded in setup())
        self.teacher = build_teacher_model(cfg)

        if cfg.model.get("compile", False):
            mode = cfg.model.get("compile_mode", "reduce-overhead")
            if hasattr(self.student.model, "text_model"):
                # MSViT: only compile the ANN text encoder; SNN backbone is incompatible with compile
                self.student.model.text_model = torch.compile(self.student.model.text_model, mode=mode)
            else:
                self.student.model = torch.compile(self.student.model, mode=mode)

        # Embedding dimensions
        self.s_dim = get_embed_dim(cfg.model.name)
        self.t_dim = get_embed_dim(cfg.model.teacher_name)

        # Projection heads: align student to teacher dimension when they differ
        self.visual_proj: Optional[nn.Linear] = None
        self.text_proj: Optional[nn.Linear] = None
        if self.s_dim != self.t_dim:
            self.visual_proj = nn.Linear(self.s_dim, self.t_dim)
            self.text_proj = nn.Linear(self.s_dim, self.t_dim)

        # Composite loss (contains AFD fusion projections + ICL cross_logit_scale)
        self.loss_fn = build_loss(cfg.loss, self.s_dim, self.t_dim)

    # ------------------------------------------------------------------
    # Setup: load teacher checkpoint
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Load teacher weights, freeze teacher, and optionally init/freeze student text encoder."""
        # Load teacher from a checkpoint file only when teacher_pretrained is not set
        # (teacher_pretrained means open_clip already loaded the weights in build_teacher_model)
        checkpoint_path = self.cfg.model.get("teacher_checkpoint")
        teacher_pretrained = self.cfg.model.get("teacher_pretrained")
        if teacher_pretrained:
            print(
                f"[Teacher] weights loaded via open_clip pretrained tag: "
                f"'{self.cfg.model.teacher_name}' / '{teacher_pretrained}'"
            )
        elif checkpoint_path:
            print(f"[Teacher] loading weights from checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Support Lightning .ckpt files (have a nested "state_dict" key).
            if "state_dict" in ckpt:
                raw = ckpt["state_dict"]
                # Strip Lightning module prefix ("student." from baseline/KD runs).
                sd = raw
                for prefix in ("student.", "teacher."):
                    candidate = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
                    if candidate:
                        sd = candidate
                        break
            else:
                sd = ckpt
            # Handle DDP-wrapped checkpoints (keys prefixed with "module.").
            if sd and next(iter(sd)).startswith("module."):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            # Strip torch.compile wrapper prefix: checkpoint may have been saved from a compiled
            # model, but we load into the uncompiled teacher here (compilation happens after).
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            self.teacher.load_state_dict(sd)
            print(f"[Teacher] checkpoint loaded successfully")
        else:
            print("[Teacher] WARNING: no checkpoint or pretrained tag specified — random weights")

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Compile teacher after weights are loaded so torch.compile sees the final weights
        # and the checkpoint keys (without _orig_mod.) match the uncompiled model.
        if self.cfg.model.get("compile_teacher", False):
            mode = self.cfg.model.get("compile_mode", "reduce-overhead")
            self.teacher.model = torch.compile(self.teacher.model, mode=mode)

        # Load student from a checkpoint file (optional).
        # Skipped when resume_ckpt is set — the full Lightning checkpoint already contains student weights.
        student_checkpoint_path = self.cfg.model.get("student_checkpoint")
        resume_ckpt = self.cfg.training.get("resume_ckpt")
        if student_checkpoint_path and not resume_ckpt:
            print(f"[Student] loading weights from checkpoint: {student_checkpoint_path}")
            # weights_only=False: user-produced checkpoints from this codebase are trusted.
            ckpt = torch.load(student_checkpoint_path, map_location="cpu", weights_only=False)
            # Support both raw state dicts (.pt) and Lightning .ckpt files.
            if "state_dict" in ckpt:
                # Lightning checkpoint: extract student weights and strip the "student." prefix.
                raw = ckpt["state_dict"]
                sd = {k[len("student."):]: v for k, v in raw.items() if k.startswith("student.")}
            else:
                sd = ckpt
            # Handle DDP-wrapped checkpoints (keys prefixed with "module.")
            if sd and next(iter(sd)).startswith("module."):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            self.student.load_state_dict(sd)
            print("[Student] checkpoint loaded successfully")

        # Optionally copy teacher text-encoder weights into the student.
        # Copies all parameters that are not part of the visual encoder and not
        # logit_scale (i.e. token_embedding, positional_embedding, transformer,
        # ln_final, text_projection). Only copies keys with matching shapes so
        # the same config flag works even when student/teacher differ in visual dim.
        if self.cfg.model.get("init_student_text_from_teacher", False):
            # Strip _orig_mod. from keys: teacher may be compiled, state_dict keys include it.
            teacher_sd = {
                k.replace("_orig_mod.", ""): v
                for k, v in self.teacher.model.state_dict().items()
                if not k.replace("_orig_mod.", "").startswith("visual")
                and k.replace("_orig_mod.", "") != "logit_scale"
            }
            student_sd = self.student.model.state_dict()
            compatible = {
                k: v for k, v in teacher_sd.items()
                if k in student_sd and student_sd[k].shape == v.shape
            }
            student_sd.update(compatible)
            self.student.model.load_state_dict(student_sd)

        # Optionally freeze the student text encoder.
        # Frozen params have requires_grad=False so exclude_weight_decay() (which
        # already filters on requires_grad) will automatically omit them from the
        # optimizer — no changes to configure_optimizers() needed.
        if self.cfg.model.get("freeze_student_text_encoder", False):
            for name, param in self.student.model.named_parameters():
                # Strip _orig_mod. prefix added by torch.compile before checking role.
                clean = name.replace("_orig_mod.", "")
                if not clean.startswith("visual") and clean != "logit_scale":
                    param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward (used during inference / eval)
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        """Run student forward pass (normalised embeddings, no distill mode)."""
        return self.student(images, texts)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, texts = batch
        mask_ratio = float(self.cfg.training.get("mask_ratio", 0.0))

        # ------ Student forward (un-normalised raw features) ------
        # distill=True sets normalize=False in encode_image/encode_text
        # Bug fix: when mask_ratio > 0, call mask_forward explicitly to
        # avoid the overwrite bug in the original encode_image (model.py L195).
        if mask_ratio > 0.0:
            s_img_raw = self.student.visual.mask_forward(images, mask_ratio)
            s_txt_raw = self.student.encode_text(texts, normalize=False)
            s_logit_scale = self.student.logit_scale.exp()
        else:
            s_img_raw, s_txt_raw, s_logit_scale = self.student(
                images, texts, distill=True, mask_ratio=0.0
            )

        # ------ Teacher forward (normalised, no grad) ------
        with torch.no_grad():
            t_img_norm, t_txt_norm, t_logit_scale = self.teacher(images, texts)

        # ------ Gather across GPUs ------
        world_size = self.trainer.world_size
        gather_with_grad = bool(self.cfg.loss.get("gather_with_grad", False))
        if world_size > 1:
            all_s_img, all_s_txt = gather_features(
                s_img_raw, s_txt_raw,
                local_loss=False,
                gather_with_grad=gather_with_grad,
                rank=self.global_rank,
                world_size=world_size,
            )
            all_t_img, all_t_txt = gather_features(
                t_img_norm, t_txt_norm,
                local_loss=False,
                gather_with_grad=False,  # teacher never needs grad
                rank=self.global_rank,
                world_size=world_size,
            )
        else:
            all_s_img, all_s_txt = s_img_raw, s_txt_raw
            all_t_img, all_t_txt = t_img_norm, t_txt_norm

        # ------ Normalise student features ------
        all_s_img_norm = F.normalize(all_s_img, dim=1)
        all_s_txt_norm = F.normalize(all_s_txt, dim=1)

        # ------ Project student → teacher dim (if needed) ------
        if self.visual_proj is not None:
            all_s_img_proj = F.normalize(self.visual_proj(all_s_img_norm), dim=1)
            all_s_txt_proj = F.normalize(self.text_proj(all_s_txt_norm), dim=1)
        else:
            all_s_img_proj = all_s_img_norm
            all_s_txt_proj = all_s_txt_norm

        # ------ Build KDFeatures ------
        N = all_s_img_norm.shape[0]
        labels = torch.arange(N, device=self.device)
        features = KDFeatures(
            s_img=all_s_img_norm,
            s_txt=all_s_txt_norm,
            s_img_proj=all_s_img_proj,
            s_txt_proj=all_s_txt_proj,
            t_img=all_t_img,
            t_txt=all_t_txt,
            s_logit_scale=s_logit_scale,
            t_logit_scale=t_logit_scale,
            labels=labels,
        )

        # ------ Compute composite loss ------
        total_loss, loss_dict = self.loss_fn(features)
        if not total_loss.isfinite():
            self.log("train_nan_skip", 1.0, on_step=True, sync_dist=False)
            return None

        # ------ Log ------
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True)
        for name, val in loss_dict.items():
            if name != "total":
                self.log(f"train_{name}_loss", val, on_step=True, on_epoch=False)

        return total_loss

    def on_after_backward(self) -> None:
        """Clamp student logit_scale to ln(100) after backward."""
        self.student.logit_scale.data.clamp_(0, math.log(100))

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Three param groups matching main_kd.py lines 212-228:
        # 1. Student no-wd params (bias, norm, logit_scale)
        # 2. Student wd params (weight matrices)
        # 3. Loss params (cross_logit_scale, fusion_proj, etc.) + projection heads

        no_wd, wd_params = exclude_weight_decay(list(self.student.named_parameters()))

        proj_params = list(self.loss_fn.parameters())
        if self.visual_proj is not None:
            proj_params += (
                list(self.visual_proj.parameters())
                + list(self.text_proj.parameters())
            )

        optimizer = torch.optim.AdamW(
            [
                {"params": no_wd, "weight_decay": 0.0},
                {"params": wd_params, "weight_decay": self.cfg.training.weight_decay},
                {"params": proj_params, "weight_decay": self.cfg.training.weight_decay},
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
