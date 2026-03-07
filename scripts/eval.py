"""Standalone evaluation script.

Loads a trained student checkpoint and runs zero-shot ImageNet
classification and retrieval evaluation.

Usage:
    python scripts/eval.py \
        model.name=timm-vit_tiny_patch16_224 \
        checkpoint=/path/to/student.ckpt \
        dataset.imagenet_val_root=/data/imagenet/val
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig

from src.datasets.tokenizer import get_tokenizer
from src.datasets.transforms import get_eval_transforms
from src.datasets.imagenet import ImageNetDataset
from src.evaluation.imagenet_eval import evaluate_zero_shot
from src.lightning.clip_kd_module import CLIPKDModule
from src.lightning.clip_module import CLIPModule
from src.models.factory import build_student_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_kd = cfg.model.get("teacher_name") is not None
    if is_kd:
        module = CLIPKDModule.load_from_checkpoint(cfg.checkpoint, cfg=cfg, tokenizer=None)
    else:
        module = CLIPModule.load_from_checkpoint(cfg.checkpoint, cfg=cfg, tokenizer=None)

    model = module.student.to(device).eval()
    tokenizer = get_tokenizer(cfg.model.name)
    transforms = get_eval_transforms()

    from torch.utils.data import DataLoader
    eval_dataloaders = {}
    if cfg.dataset.get("imagenet_val_root"):
        eval_dataloaders["imagenet"] = DataLoader(
            ImageNetDataset(cfg.dataset.imagenet_val_root, transforms),
            batch_size=256, shuffle=False, num_workers=8,
        )

    with torch.no_grad():
        metrics = evaluate_zero_shot(model, eval_dataloaders, tokenizer, device)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
