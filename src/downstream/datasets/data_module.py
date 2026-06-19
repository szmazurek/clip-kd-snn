"""GroundingDataModule: PyTorch Lightning DataModule for visual grounding.

Wraps RefCOCODataset for train/val/test, following the same cfg-driven
pattern as src/datasets/factory.py's CLIPDataModule.
"""
from __future__ import annotations

from typing import Callable, Optional

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .refcoco import RefCOCODataset
from .transforms import make_train_transforms, make_val_transforms


class GroundingDataModule(L.LightningDataModule):
    """Data module for RefCOCO-family visual grounding.

    cfg.dataset.test_splits lists every real test split for the dataset
    (e.g. ["testA", "testB"] for unc/unc+, ["test"] for gref_umd) — unlike
    val, RefCOCO datasets have multiple disjoint test splits, so
    test_dataloader() returns a list, one DataLoader per split, in the same
    order as test_splits (GroundingModule.test_step uses dataloader_idx to
    look up the split name for logging).

    Args:
        cfg: Full Hydra config (reads cfg.dataset and cfg.training).
        tokenizer: Text tokenizer callable (open_clip.tokenize-style).
    """

    def __init__(self, cfg: DictConfig, tokenizer: Callable) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.train_dataset: Optional[RefCOCODataset] = None
        self.val_dataset: Optional[RefCOCODataset] = None
        self.test_datasets: list[RefCOCODataset] = []

    def setup(self, stage: Optional[str] = None) -> None:
        ds_cfg = self.cfg.dataset
        img_size = ds_cfg.get("img_size", 224)
        train_tf = make_train_transforms(img_size)
        val_tf = make_val_transforms(img_size)

        common = dict(
            data_root=ds_cfg.data_root,
            split_root=ds_cfg.split_root,
            dataset=ds_cfg.name,
            tokenizer=self.tokenizer,
        )

        if stage in (None, "fit"):
            self.train_dataset = RefCOCODataset(
                **common, split="train", transform=train_tf,
            )
            self.val_dataset = RefCOCODataset(
                **common, split=ds_cfg.get("val_split", "val"), transform=val_tf,
            )
        if stage in (None, "test"):
            test_splits = list(ds_cfg.get("test_splits", ["test"]))
            self.test_datasets = [
                RefCOCODataset(**common, split=split, transform=val_tf)
                for split in test_splits
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.get("workers", 8),
            prefetch_factor=self.cfg.training.get("prefetch_factor", 2),
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.get("eval_batch_size", 64),
            shuffle=False,
            num_workers=self.cfg.training.get("eval_workers", 4),
            pin_memory=True,
        )

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.cfg.training.get("eval_batch_size", 64),
                shuffle=False,
                num_workers=self.cfg.training.get("eval_workers", 4),
                pin_memory=True,
            )
            for ds in self.test_datasets
        ]
