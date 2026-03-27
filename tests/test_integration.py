"""Integration test: single training step end-to-end.

Runs one training step for both CLIPModule (baseline) and CLIPKDModule (KD)
on dummy data without actual datasets or checkpoints.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import open_clip
from omegaconf import OmegaConf

from src.losses.base import KDFeatures
from src.losses.factory import build_loss
from src.models.factory import build_student_model
import torch.nn.functional as F


def _make_dummy_cfg(student_name="ViT-B-16", teacher_name="ViT-B-16"):
    return OmegaConf.create(
        {
            "model": {
                "name": student_name,
                "pretrained": None,
                "teacher_name": teacher_name,
                "teacher_checkpoint": None,
            },
            "training": {
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.98,
                "eps": 1e-6,
                "weight_decay": 0.1,
                "warmup_steps": 100,
                "epochs": 1,
                "batch_size": 4,
                "mask_ratio": 0.0,
                "precision": "32",
            },
            "loss": {
                "alpha_task": 1.0,
                "alpha_ckd": 1.0,
                "alpha_icl": 1.0,
                "alpha_fd": 1.0,
                "alpha_gd": 0.0,
                "alpha_afd": 0.0,
                "gather_with_grad": False,
                "local_loss": False,
            },
            "dataset": {},
        }
    )


class TestCompositeLossEndToEnd:
    """Verify a full forward+backward pass with all non-expensive losses."""

    def test_forward_backward_unified(self):
        B, D = 4, 512
        cfg = _make_dummy_cfg()

        student, _, _ = build_student_model(cfg)
        student.train()

        images = torch.randn(B, 3, 224, 224)
        texts = open_clip.tokenize(["a cat"] * B)

        with torch.no_grad():
            teacher_img = F.normalize(torch.randn(B, D), dim=1)
            teacher_txt = F.normalize(torch.randn(B, D), dim=1)
            t_logit_scale = torch.tensor(14.29)

        s_img_raw, s_txt_raw, s_logit_scale = student(images, texts, distill=True)
        s_img = F.normalize(s_img_raw, dim=1)
        s_txt = F.normalize(s_txt_raw, dim=1)

        feats = KDFeatures(
            s_img=s_img,
            s_txt=s_txt,
            s_img_proj=s_img,
            s_txt_proj=s_txt,
            t_img=teacher_img,
            t_txt=teacher_txt,
            s_logit_scale=s_logit_scale,
            t_logit_scale=t_logit_scale,
            labels=torch.arange(B),
        )

        loss_fn = build_loss(cfg.loss, s_embed_dim=D, t_embed_dim=D)
        total, loss_dict = loss_fn(feats)

        assert not torch.isnan(total), "NaN in total loss!"
        assert total.item() > 0

        total.backward()

        # Verify student params received gradients
        has_grad = any(
            p.grad is not None for p in student.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients flowed to student parameters!"
