"""Miscellaneous training utilities."""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
from torch import nn


def exclude_weight_decay(named_params: list[tuple[str, nn.Parameter]]) -> tuple[list, list]:
    """Split parameters into those that should/should not have weight decay.

    Mirrors the pattern from src/training/main_kd.py lines 212-213:
    exclude bias, 1-D tensors (LayerNorm gains, logit_scale, etc.).

    Args:
        named_params: List of (name, param) pairs.

    Returns:
        Tuple of (no_wd_params, wd_params).
    """
    _exclude = lambda n, p: (
        p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    no_wd = [p for n, p in named_params if _exclude(n, p) and p.requires_grad]
    wd = [p for n, p in named_params if not _exclude(n, p) and p.requires_grad]
    return no_wd, wd


def cosine_lr_lambda(warmup_steps: int, total_steps: int, base_lr: float = 1.0):
    """Return a LambdaLR-compatible function implementing cosine LR with linear warmup.

    Ported from src/training/scheduler.py. Step-based (not epoch-based).

    Args:
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (used as the cosine period end).
        base_lr: Reference LR; the lambda returns a *multiplier* on this value.
                 When used with LambdaLR, set the optimizer LR to base_lr and
                 this function will scale it appropriately.

    Returns:
        Callable(step) -> float multiplier in [0, 1].
    """
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: reaches 1.0 at step == warmup_steps
            return (step + 1) / warmup_steps
        else:
            e = step - warmup_steps
            es = max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * e / es))

    return _lr_lambda
