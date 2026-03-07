"""Loss factory: instantiate CompositeLoss from Hydra config.

Reads alpha_* weights from cfg.loss and instantiates only the losses
with alpha > 0. Passes embed dims to losses that require projection
parameters (AFD only; FD projections live in CLIPKDModule).
"""
from __future__ import annotations

from omegaconf import DictConfig

from .afd import AFDLoss
from .clip_loss import CLIPInfoNCELoss
from .composite import CompositeLoss
from .crd import CKDLoss
from .fd import FDLoss
from .gd import GDLoss
from .icl import CrossKDLoss, ICLLoss
from .mfd import MFDLoss


def build_loss(
    cfg: DictConfig,
    s_embed_dim: int,
    t_embed_dim: int,
) -> CompositeLoss:
    """Build a CompositeLoss from a Hydra loss config.

    Args:
        cfg: Hydra loss config with alpha_* fields and optional flags.
        s_embed_dim: Student embedding dimension (used for AFD proj).
        t_embed_dim: Teacher embedding dimension (used for AFD proj).

    Returns:
        Configured CompositeLoss with only active (alpha > 0) components.
    """
    losses: dict = {}
    weights: dict = {}

    def _add(name: str, alpha_key: str, module):
        alpha = float(cfg.get(alpha_key, 0.0))
        if alpha > 0.0:
            losses[name] = module
            weights[name] = alpha

    # Task (CLIP InfoNCE) — always included
    alpha_task = float(cfg.get("alpha_task", 1.0))
    losses["task"] = CLIPInfoNCELoss()
    weights["task"] = alpha_task

    # CKD / CRD (paper: CRD, code: CKD)
    _add("ckd", "alpha_ckd", CKDLoss())

    # ICL — must come before cross_kd
    _add("icl", "alpha_icl", ICLLoss())

    # CrossKD — depends on ICL populating KDFeatures.cross_logits_*
    _add("cross_kd", "alpha_cross_kd", CrossKDLoss())

    # FD or MFD (same loss formulation; MFD uses mask_ratio > 0 at model level)
    alpha_fd = float(cfg.get("alpha_fd", 0.0))
    alpha_mfd = float(cfg.get("alpha_mfd", 0.0))
    if alpha_fd > 0.0:
        losses["fd"] = FDLoss()
        weights["fd"] = alpha_fd
    elif alpha_mfd > 0.0:
        # MFD is FD with masking applied by CLIPKDModule; treated as "fd" slot
        losses["fd"] = MFDLoss()
        weights["fd"] = alpha_mfd

    # GD — expensive, disabled unless explicitly requested
    _add("gd", "alpha_gd", GDLoss())

    # AFD — requires fusion projection layers
    alpha_afd = float(cfg.get("alpha_afd", 0.0))
    if alpha_afd > 0.0:
        losses["afd"] = AFDLoss(
            s_embed_dim=s_embed_dim,
            t_embed_dim=t_embed_dim,
            out_dim=s_embed_dim,
        )
        weights["afd"] = alpha_afd

    return CompositeLoss(losses=losses, weights=weights)
