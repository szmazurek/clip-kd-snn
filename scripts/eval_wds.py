"""Standalone zero-shot evaluation script using WDS imagenet datasets.

Loads a trained student checkpoint and runs zero-shot ImageNet classification
on all four variants whose WDS paths are set in the dataset config.

Usage:
    python scripts/eval_wds.py \\
        model.name=ViT-B-16 \\
        checkpoint=/path/to/student.ckpt \\
        dataset=combined_wds_dali_pretok
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.imagenet_wds import build_imagenet_wds
from src.datasets.tokenizer import get_tokenizer
from src.datasets.transforms import get_eval_transforms
from src.evaluation.imagenet_eval import evaluate_zero_shot
from src.lightning.clip_kd_module import CLIPKDModule
from src.lightning.clip_module import CLIPModule


_WDS_KEYS = {
    "imagenet_wds_dir": "imagenet",
    "imagenet_v2_wds_dir": "imagenet_v2",
    "imagenet_r_wds_dir": "imagenet_r",
    "imagenet_sketch_wds_dir": "imagenet_sketch",
}


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

    eval_dataloaders = {}
    for cfg_key, variant_name in _WDS_KEYS.items():
        path = cfg.dataset.get(cfg_key)
        if path:
            dataset = build_imagenet_wds(wds_dir=path, transform=transforms, variant=variant_name)
            eval_dataloaders[variant_name] = DataLoader(
                dataset,
                batch_size=cfg.training.get("eval_batch_size", 256),
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

    with torch.no_grad():
        metrics = evaluate_zero_shot(model, eval_dataloaders, tokenizer, device)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
