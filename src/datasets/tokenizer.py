"""Text tokenization wrapper.

Delegates to open_clip's tokenizer registry so the rest of the codebase
does not import open_clip directly.
"""
from __future__ import annotations

from typing import Callable

import open_clip


def get_tokenizer(model_name: str) -> Callable:
    """Return the tokenizer for a given open_clip model.

    Args:
        model_name: open_clip model name, e.g. "ViT-B-16" or
                    "timm-vit_tiny_patch16_224".

    Returns:
        Callable tokenizer (either SimpleTokenizer or HFTokenizer).
    """
    return open_clip.get_tokenizer(model_name)
