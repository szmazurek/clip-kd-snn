"""Standalone zero-shot ImageNet evaluation for any CLIP model.

Supports three loading modes (mutually exclusive, checked in this order):
  1. lightning_ckpt — a Lightning .ckpt saved by this training codebase.
                      The student is extracted automatically.
  2. pretrained     — an open_clip pretrained tag (e.g. "laion2b_s34b_b88k").
                      open_clip downloads / loads the weights directly.
  3. checkpoint     — path to a raw open_clip state-dict .pt file.

Usage examples:

  # Evaluate the best public ViT-B/16 teacher (auto-download)
  python scripts/eval_open_clip.py \\
      model_name=ViT-B-16 pretrained=laion2b_s34b_b88k \\
      imagenet_val=/data/imagenet/val

  # Evaluate a raw open_clip checkpoint
  python scripts/eval_open_clip.py \\
      model_name=ViT-B-16 checkpoint=/path/to/weights.pt \\
      imagenet_val=/data/imagenet/val imagenet_v2=/data/imagenet-v2

  # Evaluate a trained Lightning checkpoint (student extracted automatically)
  python scripts/eval_open_clip.py \\
      model_name=ViT-B-16 lightning_ckpt=/path/to/last.ckpt \\
      imagenet_val=/data/imagenet/val

  # Evaluate multiple ImageNet variants
  python scripts/eval_open_clip.py \\
      model_name=ViT-B-16 pretrained=laion2b_s34b_b88k \\
      imagenet_val=/data/imagenet/val \\
      imagenet_v2=/data/imagenet-v2 \\
      imagenet_r=/data/imagenet-r \\
      imagenet_sketch=/data/imagenet-sketch

Config keys (all passed as CLI overrides):
  model_name      open_clip model name (required)
  pretrained      open_clip pretrained tag (optional)
  checkpoint      path to raw state-dict .pt (optional)
  lightning_ckpt  path to Lightning .ckpt (optional)
  imagenet_val    path to ImageNet-1K val folder   (optional, skip if null)
  imagenet_v2     path to ImageNet-V2 folder       (optional)
  imagenet_r      path to ImageNet-R folder        (optional)
  imagenet_sketch path to ImageNet-Sketch folder   (optional)
  imagenet_a      path to ImageNet-A folder        (optional)
  batch_size      default 256
  num_workers     default 8
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import open_clip
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.imagenet import ImageNetDataset
from src.datasets.imagenet_wds import build_imagenet_wds
from src.datasets.tokenizer import get_tokenizer
from src.datasets.transforms import get_eval_transforms
from src.evaluation.imagenet_eval import evaluate_zero_shot
from src.models.clip_model import CLIPWrapper
from typing import Callable


# Mapping from CLI key → variant name used in evaluate_zero_shot
_VARIANT_KEYS = {
    "imagenet_val": "imagenet",
    "imagenet_v2": "imagenet_v2",
    "imagenet_r": "imagenet_r",
    "imagenet_sketch": "imagenet_sketch",
    "imagenet_a": "imagenet_a",
}

_DEFAULTS = {
    "model_name": None,
    "pretrained": None,
    "checkpoint": None,
    "lightning_ckpt": None,
    "imagenet_val": None,
    "imagenet_v2": None,
    "imagenet_r": None,
    "imagenet_sketch": None,
    "imagenet_a": None,
    "batch_size": 256,
    "num_workers": 8,
}


def _load_model(cfg: DictConfig, device: torch.device) -> tuple[CLIPWrapper, Callable]:
    """Load model from one of three sources.

    Returns:
        Tuple of (model, preprocess_val) where preprocess_val is the
        model-specific eval transform (correct resize/normalization for that
        exact pretrained checkpoint).
    """
    if cfg.lightning_ckpt:
        # Mirror the pattern from CLIPKDModule.setup(): torch.load + load_state_dict.
        # Avoids load_from_checkpoint which would rebuild the teacher, call setup(),
        # and require a tokenizer — none of which are needed for eval.
        ckpt = torch.load(cfg.lightning_ckpt, map_location="cpu", weights_only=False)

        saved_cfg = ckpt.get("hyper_parameters", {}).get("cfg", {})
        # saved_cfg may be an OmegaConf DictConfig or a plain dict
        def _get(d, *keys):
            for k in keys:
                d = d.get(k) if isinstance(d, dict) else getattr(d, k, None)
                if d is None:
                    return None
            return d

        saved_model_name = _get(saved_cfg, "model", "name")
        if saved_model_name and not cfg.get("model_name"):
            cfg["model_name"] = saved_model_name

        # Build a fresh student using the saved config, then load weights
        from src.models.factory import build_student_model as _build_student
        student, _, preprocess_val = _build_student(saved_cfg)

        # Extract student weights from the Lightning state_dict (keys: "student.model.*").
        # torch.compile wraps the inner model, saving keys as "student.model._orig_mod.*".
        # Strip both prefixes so the uncompiled fresh model can load them.
        full_sd = ckpt["state_dict"]
        student_sd = {}
        for k, v in full_sd.items():
            if not k.startswith("student.model."):
                continue
            k = k[len("student.model."):]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            student_sd[k] = v
        student.model.load_state_dict(student_sd, strict=True)
        return student.to(device).eval(), preprocess_val

    if cfg.pretrained:
        clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        return CLIPWrapper(clip_model).to(device).eval(), preprocess_val

    if cfg.checkpoint:
        clip_model, _, preprocess_val = open_clip.create_model_and_transforms(cfg.model_name)
        sd = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
        if sd and next(iter(sd)).startswith("module"):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        clip_model.load_state_dict(sd)
        return CLIPWrapper(clip_model).to(device).eval(), preprocess_val

    raise ValueError(
        "Specify exactly one of: pretrained=<tag>, checkpoint=<path>, lightning_ckpt=<path>"
    )


def main() -> None:
    # Parse CLI overrides manually (avoids Hydra dependency for this lightweight script)
    cfg_dict = dict(_DEFAULTS)
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            if key in cfg_dict:
                cfg_dict[key] = val

    cfg = OmegaConf.create(cfg_dict)

    if not cfg.model_name and not cfg.lightning_ckpt:
        print(__doc__)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, transforms = _load_model(cfg, device)
    tokenizer = get_tokenizer(cfg.model_name)

    eval_dataloaders: dict[str, DataLoader] = {}
    for cfg_key, variant_name in _VARIANT_KEYS.items():
        path = cfg_dict.get(cfg_key)
        if not path:
            continue
        # Auto-detect format: WDS (.tar shards) vs ImageFolder
        is_wds = bool(glob.glob(os.path.join(path, "*.tar")))
        if is_wds:
            ds = build_imagenet_wds(path, transform=transforms, variant=variant_name)
            # WDS is iterable; use num_workers=1 to avoid shard-splitting issues
            eval_dataloaders[variant_name] = DataLoader(
                ds,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
        else:
            ds = ImageNetDataset(path, transform=transforms, variant=variant_name)
            eval_dataloaders[variant_name] = DataLoader(
                ds,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )

    if not eval_dataloaders:
        print("No evaluation datasets specified. Pass imagenet_val=<path> or similar.")
        sys.exit(1)

    print(f"Evaluating {cfg.model_name} on: {list(eval_dataloaders.keys())}")
    with torch.no_grad():
        metrics = evaluate_zero_shot(model, eval_dataloaders, tokenizer, device)

    print("\n--- Results ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v * 100:.2f}%")


if __name__ == "__main__":
    main()
