"""Model factory: build student/teacher CLIP models from Hydra config.

Thin wrappers around open_clip.create_model_and_transforms(). The heavy
lifting (architecture definitions, weight loading) is handled by open_clip.

MSViT models (name prefix "MSViT-") are handled by a separate builder that
combines an MSFormer spiking image encoder with an open_clip text encoder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import open_clip
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .clip_model import CLIPWrapper

# Register custom model configs (e.g. timm-based models not in open_clip built-ins)
_MODEL_CONFIGS_DIR = Path(__file__).parent.parent.parent / "model_configs"
if _MODEL_CONFIGS_DIR.is_dir():
    open_clip.add_model_config(_MODEL_CONFIGS_DIR)

# ---------------------------------------------------------------------------
# MSViT model registry
# ---------------------------------------------------------------------------

_MSVIT_PREFIX = "MSViT-"

# Maps MSViT model name → effective CLIP embedding dimension (= text encoder dim)
_MSVIT_EMBED_DIMS: dict[str, int] = {
    "MSViT-ViT-B-16": 512,
    "MSViT-ViT-T-16": 256,
}

_MSVIT_VISUAL_DIM = 512  # MSFormer_10_512 always produces 512-dim image features


def _is_msvit(name: str) -> bool:
    return name.startswith(_MSVIT_PREFIX)


# ---------------------------------------------------------------------------
# QKFormer model registry
# ---------------------------------------------------------------------------

_QKFORMER_PREFIX = "QKFormer-"

# Maps QKFormer model name → effective CLIP embedding dimension (= text encoder dim)
_QKFORMER_EMBED_DIMS: dict[str, int] = {
    "QKFormer-ViT-B-16": 512,
    "QKFormer-ViT-T-16": 256,
}

_QKFORMER_VISUAL_DIM = 512  # QKFormer_10_512 always produces 512-dim image features


def _is_qkformer(name: str) -> bool:
    return name.startswith(_QKFORMER_PREFIX)


# ---------------------------------------------------------------------------
# LoopViT model registry
# ---------------------------------------------------------------------------

_LOOPVIT_PREFIX = "LoopViT-"

# Maps LoopViT model name → effective CLIP embedding dimension (= text encoder dim)
_LOOPVIT_EMBED_DIMS: dict[str, int] = {
    "LoopViT-ViT-B-16": 512,
}


def _is_loopvit(name: str) -> bool:
    return name.startswith(_LOOPVIT_PREFIX)


def _build_loopvit_student_model(
    cfg: DictConfig,
) -> tuple[nn.Module, Callable, Callable]:
    """Create student model with LoopViT image encoder + open_clip text encoder.

    Args:
        cfg: Hydra config.  Reads:
            cfg.model.text_encoder_name  – open_clip model name for the text side.
            cfg.model.embed_dim          – LoopViT embedding dimension.
            cfg.model.num_heads          – number of attention heads.
            cfg.model.loop_core_depth    – transformer depth per loop step.
            cfg.model.mlp_ratio          – MLP expansion ratio.
            cfg.model.dropout            – dropout rate.
            cfg.model.max_loop_steps     – maximum recurrent iterations.
            cfg.model.min_loop_steps     – minimum recurrent iterations.
            cfg.model.add_step_embeddings – whether to use per-step embeddings.
            cfg.model.use_exit_gate      – whether to use dynamic early exit.

    Returns:
        Tuple of (CLIPWrapper(LoopViTCLIPModel), preprocess_train, preprocess_val).
    """
    from .loopvit_clip import LoopViTCLIPModel
    from .visual_encoders.loopvit import LoopViT

    # ---- Text encoder (open_clip) -----------------------------------------
    text_encoder_name = cfg.model.get("text_encoder_name", "ViT-B-16")
    text_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        text_encoder_name, pretrained=None
    )
    # Drop the open_clip visual encoder — only encode_text() is used here, and
    # keeping text_clip.visual would register its parameters without ever using
    # them, causing DDP to raise unused-parameter errors.
    del text_clip.visual

    text_embed_dim = open_clip.get_model_config(text_encoder_name)["embed_dim"]

    # ---- Visual encoder (LoopViT) -----------------------------------------
    visual_embed_dim = int(cfg.model.get("embed_dim", 768))
    visual = LoopViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=visual_embed_dim,
        num_heads=int(cfg.model.get("num_heads", 12)),
        mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        loop_core_depth=int(cfg.model.get("loop_core_depth", 1)),
        max_loop_steps=int(cfg.model.get("max_loop_steps", 12)),
        min_loop_steps=int(cfg.model.get("min_loop_steps", 1)),
        add_step_embeddings=bool(cfg.model.get("add_step_embeddings", False)),
        use_exit_gate=bool(cfg.model.get("use_exit_gate", False)),
    )

    # ---- Assemble CLIP model ----------------------------------------------
    model = LoopViTCLIPModel(
        visual=visual,
        text_model=text_clip,
        visual_embed_dim=visual_embed_dim,
        text_embed_dim=text_embed_dim,
    )
    return CLIPWrapper(model), preprocess_train, preprocess_val


def _build_qkformer_student_model(
    cfg: DictConfig,
) -> tuple[nn.Module, Callable, Callable]:
    """Create student model with QKFormer image encoder + open_clip text encoder.

    Args:
        cfg: Hydra config.  Reads:
            cfg.model.text_encoder_name  – open_clip model name for the text side.
            cfg.model.snn.*              – SNN hyperparameters (T, neuron_type, …).

    Returns:
        Tuple of (CLIPWrapper(QKFormerCLIPModel), preprocess_train, preprocess_val).
    """
    from .qkformer_clip import QKFormerCLIPModel
    from .visual_encoders.qkformer import QKFormer_10_512, SNNParams

    # ---- SNN config -------------------------------------------------------
    snn_cfg = cfg.model.get("snn", {})
    if isinstance(snn_cfg, DictConfig):
        snn_cfg = OmegaConf.to_container(snn_cfg, resolve=True)

    snn = SNNParams(
        neuron_type=snn_cfg.get("neuron_type", "lif"),
        v_threshold=float(snn_cfg.get("v_threshold", 1.0)),
        tau=float(snn_cfg.get("tau", 2.0)),
        backend=snn_cfg.get("backend", "torch"),
    )
    T = int(snn_cfg.get("T", 4))

    # ---- Text encoder (open_clip) -----------------------------------------
    text_encoder_name = cfg.model.get("text_encoder_name", "ViT-B-16")
    text_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        text_encoder_name, pretrained=None
    )
    # Drop the open_clip visual encoder — only encode_text() is used here, and
    # keeping text_clip.visual would register its parameters without ever touching
    # them in the forward pass, causing DDP to raise unused-parameter errors.
    del text_clip.visual

    text_embed_dim = open_clip.get_model_config(text_encoder_name)["embed_dim"]

    # ---- Visual encoder (QKFormer) ----------------------------------------
    visual = QKFormer_10_512(T=T, snn=snn)

    # ---- Assemble CLIP model ----------------------------------------------
    model = QKFormerCLIPModel(
        visual=visual,
        text_model=text_clip,
        visual_embed_dim=_QKFORMER_VISUAL_DIM,
        text_embed_dim=text_embed_dim,
        T=T,
    )
    return CLIPWrapper(model), preprocess_train, preprocess_val


def _build_msvit_student_model(
    cfg: DictConfig,
) -> tuple[nn.Module, Callable, Callable]:
    """Create student model with MSViT image encoder + open_clip text encoder.

    Args:
        cfg: Hydra config.  Reads:
            cfg.model.text_encoder_name  – open_clip model name for the text side.
            cfg.model.snn.*              – SNN hyperparameters (T, neuron_type, …).

    Returns:
        Tuple of (CLIPWrapper(MSViTCLIPModel), preprocess_train, preprocess_val).
    """
    from .msvit_clip import MSViTCLIPModel
    from .visual_encoders.msformer import MSFormer_10_512, SNNParams

    # ---- SNN config -------------------------------------------------------
    snn_cfg = cfg.model.get("snn", {})
    # OmegaConf DictConfig → plain dict for .get() calls
    if isinstance(snn_cfg, DictConfig):
        snn_cfg = OmegaConf.to_container(snn_cfg, resolve=True)

    snn = SNNParams(
        neuron_type=snn_cfg.get("neuron_type", "lif"),
        v_threshold=float(snn_cfg.get("v_threshold", 1.0)),
        tau=float(snn_cfg.get("tau", 2.0)),
        backend=snn_cfg.get("backend", "torch"),
    )
    T = int(snn_cfg.get("T", 4))

    # ---- Text encoder (open_clip) -----------------------------------------
    text_encoder_name = cfg.model.get("text_encoder_name", "ViT-B-16")
    text_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        text_encoder_name, pretrained=None
    )
    # Drop the open_clip visual encoder — only encode_text() is used here.
    del text_clip.visual

    text_embed_dim = open_clip.get_model_config(text_encoder_name)["embed_dim"]

    # ---- Visual encoder (MSFormer) ----------------------------------------
    visual = MSFormer_10_512(T=T, snn=snn)

    # ---- Assemble CLIP model ----------------------------------------------
    model = MSViTCLIPModel(
        visual=visual,
        text_model=text_clip,
        visual_embed_dim=_MSVIT_VISUAL_DIM,
        text_embed_dim=text_embed_dim,
        T=T,
    )
    return CLIPWrapper(model), preprocess_train, preprocess_val


def build_student_model(
    cfg: DictConfig,
) -> tuple[nn.Module, Callable, Callable]:
    """Create student CLIP model with train/eval image transforms.

    Dispatches to the MSViT builder when cfg.model.name starts with "MSViT-",
    otherwise delegates to open_clip.

    The student is always initialised from scratch (pretrained=None)
    unless cfg.model.pretrained is explicitly set.

    Args:
        cfg: Hydra config with cfg.model.name and optional cfg.model.pretrained.

    Returns:
        Tuple of (model, preprocess_train, preprocess_val).
    """
    if _is_msvit(cfg.model.name):
        return _build_msvit_student_model(cfg)
    if _is_qkformer(cfg.model.name):
        return _build_qkformer_student_model(cfg)
    if _is_loopvit(cfg.model.name):
        return _build_loopvit_student_model(cfg)

    pretrained = cfg.model.get("pretrained", None)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        cfg.model.name,
        pretrained=pretrained,
    )
    return CLIPWrapper(model), preprocess_train, preprocess_val


def build_teacher_model(cfg: DictConfig) -> nn.Module:
    """Create teacher CLIP model, optionally loading pretrained weights via open_clip.

    If cfg.model.teacher_pretrained is set, open_clip downloads/loads the weights
    directly (no separate checkpoint file needed). Otherwise the model is created
    with random weights and CLIPKDModule.setup() will load from teacher_checkpoint.

    Args:
        cfg: Hydra config with cfg.model.teacher_name and optional
             cfg.model.teacher_pretrained.

    Returns:
        Teacher model (weights either random or open_clip pretrained).
    """
    pretrained = cfg.model.get("teacher_pretrained", None)
    model, _, _ = open_clip.create_model_and_transforms(
        cfg.model.teacher_name, pretrained=pretrained
    )
    return CLIPWrapper(model)


def get_embed_dim(model_name: str) -> int:
    """Return the CLIP embedding dimension for a given model name.

    For MSViT models (prefix "MSViT-") returns the effective CLIP space
    dimension (text encoder dim) from the hardcoded registry.  For all other
    models reads from open_clip's config registry.

    Args:
        model_name: Model name, e.g. "ViT-B-16" or "MSViT-ViT-B-16".

    Returns:
        Embedding dimension.

    Raises:
        ValueError: If the model name is not found in any registry.
    """
    if model_name in _MSVIT_EMBED_DIMS:
        return _MSVIT_EMBED_DIMS[model_name]
    if model_name in _QKFORMER_EMBED_DIMS:
        return _QKFORMER_EMBED_DIMS[model_name]
    if model_name in _LOOPVIT_EMBED_DIMS:
        return _LOOPVIT_EMBED_DIMS[model_name]

    cfg = open_clip.get_model_config(model_name)
    if cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in open_clip registry. "
            f"Available: {open_clip.list_models()[:10]}..."
        )
    return cfg["embed_dim"]
