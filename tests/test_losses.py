"""Unit tests for CLIP-KD loss modules.

Tests verify:
  1. Each loss produces a scalar output.
  2. Total loss == weighted sum of components (CompositeLoss).
  3. Gradients flow to student params but NOT teacher params.
  4. Numerical equivalence to original KDClipLoss implementation.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses.base import KDFeatures
from src.losses.clip_loss import CLIPInfoNCELoss
from src.losses.crd import CKDLoss
from src.losses.fd import FDLoss
from src.losses.gd import GDLoss, get_grad
from src.losses.icl import ICLLoss, CrossKDLoss
from src.losses.afd import AFDLoss
from src.losses.composite import CompositeLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 8    # batch size
D_S = 192  # student embed dim (ViT-T/16)
D_T = 512  # teacher embed dim (ViT-B/16)


def _make_features(s_dim=D_S, t_dim=D_T, batch=B) -> KDFeatures:
    """Build a KDFeatures instance with random normalised embeddings."""
    s_img = F.normalize(torch.randn(batch, s_dim), dim=1)
    s_txt = F.normalize(torch.randn(batch, s_dim), dim=1)
    s_img_proj = F.normalize(torch.randn(batch, t_dim), dim=1)
    s_txt_proj = F.normalize(torch.randn(batch, t_dim), dim=1)
    t_img = F.normalize(torch.randn(batch, t_dim), dim=1)
    t_txt = F.normalize(torch.randn(batch, t_dim), dim=1)
    s_logit_scale = torch.tensor(np.log(1 / 0.07)).exp()
    t_logit_scale = torch.tensor(np.log(1 / 0.07)).exp()
    labels = torch.arange(batch)
    return KDFeatures(
        s_img=s_img, s_txt=s_txt,
        s_img_proj=s_img_proj, s_txt_proj=s_txt_proj,
        t_img=t_img, t_txt=t_txt,
        s_logit_scale=s_logit_scale, t_logit_scale=t_logit_scale,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Individual loss tests
# ---------------------------------------------------------------------------

class TestCLIPInfoNCELoss:
    def test_output_is_scalar(self):
        loss_fn = CLIPInfoNCELoss()
        feats = _make_features(s_dim=D_T, t_dim=D_T)
        # CLIPInfoNCE uses s_img, s_txt (D_T when no projection needed)
        feats.s_img = feats.s_img_proj
        feats.s_txt = feats.s_txt_proj
        loss = loss_fn(feats)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_positive(self):
        loss_fn = CLIPInfoNCELoss()
        feats = _make_features(s_dim=D_T, t_dim=D_T)
        feats.s_img = feats.s_img_proj
        feats.s_txt = feats.s_txt_proj
        loss = loss_fn(feats)
        assert loss.item() > 0

    def test_matches_original_clip_loss(self):
        """Verify numerical equivalence with original ClipLoss."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
        try:
            from open_clip.loss import ClipLoss
        except ImportError:
            pytest.skip("open_clip not importable from src/")

        torch.manual_seed(42)
        feats = _make_features(s_dim=D_T, t_dim=D_T)
        feats.s_img = feats.s_img_proj
        feats.s_txt = feats.s_txt_proj

        # Original
        orig = ClipLoss(world_size=1)
        orig_loss = orig(feats.s_img, feats.s_txt, feats.s_logit_scale)

        # New
        new_fn = CLIPInfoNCELoss()
        new_loss = new_fn(feats)

        assert abs(orig_loss.item() - new_loss.item()) < 1e-5


class TestCKDLoss:
    def test_output_is_scalar(self):
        loss_fn = CKDLoss()
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.shape == ()

    def test_loss_positive(self):
        loss_fn = CKDLoss()
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.item() >= 0


class TestICLLoss:
    def test_output_is_scalar(self):
        loss_fn = ICLLoss()
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.shape == ()

    def test_populates_cross_logits(self):
        """ICLLoss must populate cross_logits for CrossKDLoss downstream."""
        loss_fn = ICLLoss()
        feats = _make_features()
        assert feats.cross_logits_img2txt is None
        _ = loss_fn(feats)
        assert feats.cross_logits_img2txt is not None
        assert feats.cross_logits_img2txt.shape == (B, B)

    def test_cross_logit_scale_is_learnable(self):
        loss_fn = ICLLoss()
        params = list(loss_fn.parameters())
        assert len(params) == 1, "ICLLoss should have exactly one learnable param"


class TestCrossKDLoss:
    def test_requires_icl_to_run_first(self):
        loss_fn = CrossKDLoss()
        feats = _make_features()
        with pytest.raises(AssertionError, match="ICLLoss must run before"):
            _ = loss_fn(feats)

    def test_output_is_scalar(self):
        icl = ICLLoss()
        cross_kd = CrossKDLoss()
        feats = _make_features()
        _ = icl(feats)
        loss = cross_kd(feats)
        assert loss.shape == ()


class TestFDLoss:
    def test_output_is_scalar(self):
        loss_fn = FDLoss()
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.shape == ()

    def test_zero_when_identical(self):
        """FD loss should be 0 when student projection equals teacher."""
        loss_fn = FDLoss()
        feats = _make_features()
        feats.s_img_proj = feats.t_img.clone()
        feats.s_txt_proj = feats.t_txt.clone()
        loss = loss_fn(feats)
        assert loss.item() < 1e-6


class TestGDLoss:
    def test_get_grad_shapes(self):
        p = F.normalize(torch.randn(B, D_T), dim=1)
        k = F.normalize(torch.randn(B, D_T), dim=1)
        tau = torch.tensor(14.29)
        labels = torch.arange(B)
        grad_p, grad_k = get_grad(p, k, tau, labels)
        assert grad_p.shape == p.shape
        assert grad_k.shape == (B, D_T)       # (B, D) — same shape as k

    def test_output_is_scalar(self):
        loss_fn = GDLoss()
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.shape == ()


class TestAFDLoss:
    def test_output_is_scalar(self):
        loss_fn = AFDLoss(s_embed_dim=D_S, t_embed_dim=D_T, out_dim=D_S)
        feats = _make_features()
        loss = loss_fn(feats)
        assert loss.shape == ()

    def test_learnable_params(self):
        loss_fn = AFDLoss(s_embed_dim=D_S, t_embed_dim=D_T, out_dim=D_S)
        params = list(loss_fn.parameters())
        # visual_fusion_proj, text_fusion_proj (weight + bias), fusion_logit_scale
        assert len(params) == 5


# ---------------------------------------------------------------------------
# CompositeLoss tests
# ---------------------------------------------------------------------------

class TestCompositeLoss:
    def test_total_equals_weighted_sum(self):
        w_task = 1.0
        w_ckd = 2.0
        losses = {
            "task": CLIPInfoNCELoss(),
            "ckd": CKDLoss(),
        }
        weights = {"task": w_task, "ckd": w_ckd}
        composite = CompositeLoss(losses=losses, weights=weights)

        feats = _make_features(s_dim=D_T, t_dim=D_T)
        feats.s_img = feats.s_img_proj
        feats.s_txt = feats.s_txt_proj

        total, loss_dict = composite(feats)
        expected = w_task * loss_dict["task"] + w_ckd * loss_dict["ckd"]
        assert abs(total.item() - expected.item()) < 1e-5

    def test_zero_weight_skips_loss(self):
        losses = {
            "task": CLIPInfoNCELoss(),
            "ckd": CKDLoss(),
        }
        weights = {"task": 1.0, "ckd": 0.0}
        composite = CompositeLoss(losses=losses, weights=weights)

        feats = _make_features(s_dim=D_T, t_dim=D_T)
        feats.s_img = feats.s_img_proj
        feats.s_txt = feats.s_txt_proj

        _, loss_dict = composite(feats)
        assert "ckd" not in loss_dict


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_to_student_not_teacher(self):
        """After backward, student leaf tensors receive gradients; teacher do not."""
        # Use leaf tensors (pre-normalisation) to check grads via retain_grad()
        s_img_raw = torch.randn(B, D_T, requires_grad=True)
        s_txt_raw = torch.randn(B, D_T, requires_grad=True)
        s_img_proj = F.normalize(s_img_raw, dim=1)
        s_txt_proj = F.normalize(s_txt_raw, dim=1)
        s_img_proj.retain_grad()
        s_txt_proj.retain_grad()

        t_img = F.normalize(torch.randn(B, D_T), dim=1)  # no grad (teacher)
        t_txt = F.normalize(torch.randn(B, D_T), dim=1)

        feats = _make_features()
        feats.s_img_proj = s_img_proj
        feats.s_txt_proj = s_txt_proj
        feats.t_img = t_img
        feats.t_txt = t_txt

        loss_fn = FDLoss()
        loss = loss_fn(feats)
        loss.backward()

        # Leaf tensors receive gradients
        assert s_img_raw.grad is not None, "Student img leaf grad should exist"
        assert s_txt_raw.grad is not None, "Student txt leaf grad should exist"
        # Non-leaf retain_grad also works
        assert s_img_proj.grad is not None, "Student img_proj retain_grad should exist"
        # Teacher tensors have no grad (requires_grad=False)
        assert t_img.grad is None, "Teacher img grad should be None"
        assert t_txt.grad is None, "Teacher txt grad should be None"
