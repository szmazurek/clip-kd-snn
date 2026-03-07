"""Cross-modal retrieval evaluation (R@1, R@5, R@10).

Evaluates both image-to-text (I2T) and text-to-image (T2I) retrieval on
MS-COCO or Flickr30K test splits.

Ported from src/training/train.py get_metrics() (lines 374-391).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def encode_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all images and captions in a retrieval dataset.

    Assumes dataloader returns (image, tokenised_captions, image_id) tuples
    where tokenised_captions is shape (num_captions, 77).

    Args:
        model: open_clip CLIP model.
        dataloader: MSCOCODataset or Flickr30KDataset dataloader.
        device: Target device.

    Returns:
        Tuple of (all_image_features, all_text_features), both L2-normalised.
        all_image_features: (num_images, D)
        all_text_features: (num_images * num_captions_per_image, D)
    """
    all_image_feats = []
    all_text_feats = []

    for images, texts, _ in tqdm(dataloader, desc="Encoding retrieval dataset", leave=False):
        images = images.to(device)
        image_feats = F.normalize(model.encode_image(images), dim=-1)
        all_image_feats.append(image_feats.cpu())

        # texts: (B, num_captions, 77) — flatten to (B*num_captions, 77)
        B, num_caps, L = texts.shape
        texts_flat = texts.view(B * num_caps, L).to(device)
        text_feats = F.normalize(model.encode_text(texts_flat), dim=-1)
        # Reshape back and average across captions → (B, D) for T2I ground truth
        # Keep per-caption for I2T evaluation
        all_text_feats.append(text_feats.cpu())

    return torch.cat(all_image_feats, dim=0), torch.cat(all_text_feats, dim=0)


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = 1.0,
    num_captions_per_image: int = 5,
) -> dict[str, float]:
    """Compute R@1, R@5, R@10 for I2T and T2I retrieval.

    Ported from src/training/train.py get_metrics() (lines 374-391).

    Args:
        image_features: (N, D) L2-normalised image embeddings.
        text_features: (N * num_captions_per_image, D) L2-normalised text embeddings.
        logit_scale: Temperature scalar (default 1.0, cosine similarity).
        num_captions_per_image: Number of captions per image (5 for COCO/Flickr).

    Returns:
        Dict of metric names → values.
    """
    N = image_features.shape[0]
    # For each image, its ground-truth captions are at indices
    # [i*num_caps, (i+1)*num_caps)
    logits_i2t = logit_scale * image_features @ text_features.T  # (N, N*caps)
    logits_t2i = logits_i2t.T  # (N*caps, N)

    metrics = {}

    # I2T: for image i, find its captions
    i2t_gt = torch.arange(N).view(-1, 1)  # (N, 1)
    i2t_gt = i2t_gt * num_captions_per_image + torch.arange(num_captions_per_image).view(1, -1)
    # i2t_gt[i] = [i*5, i*5+1, ..., i*5+4]

    # For each image, rank all texts by similarity
    ranking = torch.argsort(logits_i2t, dim=1, descending=True)  # (N, N*caps)
    for k in [1, 5, 10]:
        top_k = ranking[:, :k]  # (N, k)
        hit = torch.any(
            top_k.unsqueeze(-1).eq(i2t_gt.unsqueeze(1)), dim=(1, 2)
        ).float()
        metrics[f"i2t_R@{k}"] = hit.mean().item()

    # T2I: for text j (caption of image i = j // num_caps), find image i
    t2i_gt = torch.arange(N * num_captions_per_image) // num_captions_per_image  # (N*caps,)
    ranking_t2i = torch.argsort(logits_t2i, dim=1, descending=True)  # (N*caps, N)

    for k in [1, 5, 10]:
        preds = torch.where(
            ranking_t2i == t2i_gt.view(-1, 1)
        )[1]  # position of GT in ranked list
        metrics[f"t2i_R@{k}"] = float((preds < k).float().mean().item())

    return metrics


def evaluate_retrieval(
    model: nn.Module,
    eval_dataloaders: dict[str, DataLoader],
    device: torch.device,
    num_captions_per_image: int = 5,
) -> dict[str, float]:
    """Run retrieval evaluation on COCO and/or Flickr30K.

    Args:
        model: open_clip CLIP model.
        eval_dataloaders: Dict mapping "mscoco" or "flickr30k" to DataLoader.
        device: Target device.
        num_captions_per_image: 5 for both COCO and Flickr30K.

    Returns:
        Dict of "{dataset}/{metric}" → value.
    """
    retrieval_keys = [k for k in eval_dataloaders if k in ("mscoco", "flickr30k")]
    results = {}
    for key in retrieval_keys:
        img_feats, txt_feats = encode_dataset(model, eval_dataloaders[key], device)
        metrics = compute_retrieval_metrics(
            img_feats, txt_feats, num_captions_per_image=num_captions_per_image
        )
        results.update({f"{key}/{m}": v for m, v in metrics.items()})
    return results
