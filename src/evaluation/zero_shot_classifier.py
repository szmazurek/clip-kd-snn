"""Build zero-shot text classifier for ImageNet classification.

Encodes all 1000 ImageNet class names using the 80 CLIP prompt templates,
averages the text embeddings per class, and returns a (D, 1000) classifier
matrix ready for cosine similarity scoring.

Ported from src/training/zero_shot.py zero_shot_classifier() (lines 12-31).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def build_zero_shot_classifier(
    model: nn.Module,
    classnames: list[str],
    templates: list[Callable],
    tokenizer: Callable,
    device: torch.device,
) -> torch.Tensor:
    """Build a zero-shot text classifier.

    Encodes each class name with all templates, averages the L2-normalised
    embeddings, then re-normalises the mean.

    Args:
        model: open_clip CLIP model with encode_text().
        classnames: List of 1000 ImageNet class name strings.
        templates: List of callables f(classname) -> prompt string.
        tokenizer: Text tokenizer callable.
        device: Target device.

    Returns:
        Float tensor of shape (D, num_classes), i.e. each column is the
        normalised mean text embedding for one class. Ready for
        image_features @ classifier to produce (N, num_classes) logits.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Building zero-shot classifier"):
            texts = [template(classname) for template in templates]
            tokens = tokenizer(texts).to(device)
            class_embeds = model.encode_text(tokens)  # (num_templates, D)
            class_embeds = F.normalize(class_embeds, dim=-1)
            class_embedding = class_embeds.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        classifier = torch.stack(zeroshot_weights, dim=1).to(device)  # (D, num_classes)
    return classifier
