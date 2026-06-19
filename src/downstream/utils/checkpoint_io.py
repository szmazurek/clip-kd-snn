"""Loads weights for one submodule out of a CLIP-KD Lightning checkpoint.

Training runs in this repo wrap models as CLIPWrapper(model) and assign the
wrapper to self.student (see src/lightning/clip_module.py,
src/lightning/clip_kd_module.py). The resulting Lightning .ckpt therefore
stores weights under "student.model.<submodule>.*", e.g.
"student.model.visual.*" for the LoopViT backbone and
"student.model.text_model.*" for either an open_clip CLIP instance or a
LoopText instance (both are assigned to `model.text_model` by the factory).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def load_submodule_from_lightning_ckpt(
    ckpt_path: str,
    module: nn.Module,
    submodule_key: str,
    strict: bool = True,
) -> None:
    """Load `module`'s weights from the `model.<submodule_key>.*` keys of a checkpoint.

    Args:
        ckpt_path: Path to a Lightning .ckpt file (or a raw state-dict .pt file).
        module: Target module to load weights into, e.g. a LoopViT or LoopText
                instance, or an open_clip CLIP instance (for "text_model").
        submodule_key: "visual" or "text_model".
        strict: Forwarded to module.load_state_dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    raw = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}

    prefix = f"model.{submodule_key}."
    sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("student." + prefix):
            sd[k[len("student." + prefix):]] = v
        elif k.startswith(prefix):
            sd[k[len(prefix):]] = v

    if not sd:
        raise RuntimeError(
            f"No keys matching '{prefix}' (optionally 'student.'-prefixed) "
            f"found in checkpoint {ckpt_path}"
        )
    module.load_state_dict(sd, strict=strict)
