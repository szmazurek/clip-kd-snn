"""Masked Feature Distillation (MFD) loss (Equation 12 in paper).

MFD is identical to FD in its loss formulation — MSE between student and
teacher embeddings — but differs in the *input*: the student receives
masked image patches (following MAE), while the teacher sees the full image.

The masking is applied at the model level by setting mask_ratio > 0 in
CLIPKDModule.training_step. When MFD is active, the same FDLoss is used;
this module serves as a named alias so the Hydra config can select it
explicitly and training logs can distinguish it from standard FD.

Source: src/open_clip/loss.py lines 289-296 (same as FD).
        src/open_clip/model.py lines 192-197 (encode_image with mask_ratio).

Note: The original encode_image has a bug (mask_forward result is
overwritten by the standard visual forward on line 195). This is fixed in
CLIPKDModule by patching the encode_image method or calling mask_forward
directly. See CLIPKDModule for the fix.
"""
from __future__ import annotations

from .fd import FDLoss

# MFD uses the same MSE computation as FD; masking is applied at model level.
MFDLoss = FDLoss
