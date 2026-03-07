"""Model factory: build student/teacher CLIP models from Hydra config.

Thin wrappers around open_clip.create_model_and_transforms(). The heavy
lifting (architecture definitions, weight loading) is handled by open_clip.
"""
from __future__ import annotations

from typing import Callable

import open_clip
import torch
from omegaconf import DictConfig
from torch import nn

from .clip_model import CLIPWrapper


def build_student_model(
    cfg: DictConfig,
) -> tuple[nn.Module, Callable, Callable]:
    """Create student CLIP model with train/eval image transforms.

    The student is always initialised from scratch (pretrained=None)
    unless cfg.model.pretrained is explicitly set.

    Args:
        cfg: Hydra config with cfg.model.name and optional cfg.model.pretrained.

    Returns:
        Tuple of (model, preprocess_train, preprocess_val).
    """
    pretrained = cfg.model.get("pretrained", None)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        cfg.model.name,
        pretrained=pretrained,
    )
    return CLIPWrapper(model), preprocess_train, preprocess_val


def build_teacher_model(cfg: DictConfig) -> nn.Module:
    """Create teacher CLIP model (architecture only; weights loaded separately).

    Teacher weights are loaded in CLIPKDModule.setup() after Lightning has
    moved the model to the target device, to avoid CPU-GPU copy timing issues.

    Args:
        cfg: Hydra config with cfg.model.teacher_name.

    Returns:
        Teacher model with random weights (to be replaced in setup()).
    """
    model, _, _ = open_clip.create_model_and_transforms(cfg.model.teacher_name)
    return CLIPWrapper(model)


def get_embed_dim(model_name: str) -> int:
    """Return the embedding dimension for a given open_clip model name.

    Reads from open_clip's model config registry.

    Args:
        model_name: open_clip model name, e.g. "ViT-B-16".

    Returns:
        Embedding dimension (e.g. 512 for ViT-B-16).

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    cfg = open_clip.get_model_config(model_name)
    if cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in open_clip registry. "
            f"Available: {open_clip.list_models()[:10]}..."
        )
    return cfg["embed_dim"]
