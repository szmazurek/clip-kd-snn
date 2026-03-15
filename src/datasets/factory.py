"""CLIPDataModule: PyTorch Lightning DataModule for CLIP training/evaluation.

Wraps CC3M, CC12M, ImageNet, MS-COCO, and Flickr30K datasets and exposes
them as train/val DataLoaders through Lightning's DataModule interface.
"""
from __future__ import annotations

from typing import Callable, Optional

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset

import torch.distributed as _dist


def _wds_num_samples(total: int) -> int:
    """Return per-rank sample count for a WebDataset epoch.

    wds.split_by_node distributes shards by global rank, so each GPU sees
    roughly total/world_size samples per epoch. Passing this value to
    with_epoch() and __len__() keeps Lightning's progress bar and LR
    scheduler step counts correct under any number of GPUs.
    """
    try:
        world_size = _dist.get_world_size() if _dist.is_initialized() else 1
    except Exception:
        world_size = 1
    return max(1, total // world_size)


from .cc3m import CC3MDataset
from .cc12m import CC12MDataset
from .cc3m_wds import build_cc3m_wds, CC3M_TRAIN_SAMPLES
from .cc12m_wds import build_cc12m_wds, CC12M_TRAIN_SAMPLES
from .combined import build_combined_dataset
from .combined_wds import build_combined_wds, CC3M_CC12M_TRAIN_SAMPLES
from .cc3m_hfd import build_cc3m_hfd
from .cc12m_hfd import build_cc12m_hfd
from .combined_hfd import build_combined_hfd
from .dali_wds import DALILoader, build_dali_train_loader
from .flickr30k import Flickr30KDataset
from .imagenet import ImageNetDataset
from .imagenet_hfd import ImageNetHFDataset
from .mscoco import MSCOCODataset


class CLIPDataModule(L.LightningDataModule):
    """Data module for CLIP training and evaluation.

    Supports the following training dataset types:
        "cc3m"     — CC3M only (CSV + local image files)
        "cc12m"    — CC12M only (CSV + local image files)
        "combined" — CC3M + CC12M concatenated (CSV + local image files)
        "cc3m_wds" — CC3M WebDataset shards (pixparse/cc3m-wds HuggingFace format)

    Evaluation dataloaders are added for whichever paths are set in config.

    Args:
        cfg: Full Hydra config (reads cfg.dataset and cfg.training).
        preprocess_train: Training image transforms from open_clip.
        preprocess_val: Evaluation image transforms from open_clip.
        tokenizer: Text tokenizer callable.
    """

    def __init__(
        self,
        cfg: DictConfig,
        preprocess_train: Callable,
        preprocess_val: Callable,
        tokenizer: Callable,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.tokenizer = tokenizer

        self.train_dataset = None
        self.val_datasets: dict = {}

    def setup(self, stage: Optional[str] = None) -> None:
        ds_cfg = self.cfg.dataset

        # ---- Training dataset ----
        dtype = ds_cfg.get("type", "cc3m")
        if dtype == "cc3m":
            self.train_dataset = CC3MDataset(
                data_root=ds_cfg.train_root,
                csv_path=ds_cfg.train_csv,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
                img_key=ds_cfg.get("img_key", "filepath"),
                caption_key=ds_cfg.get("caption_key", "title"),
                sep=ds_cfg.get("sep", "\t"),
            )
        elif dtype == "cc12m":
            self.train_dataset = CC12MDataset(
                data_root=ds_cfg.train_root,
                csv_path=ds_cfg.train_csv,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
                img_key=ds_cfg.get("img_key", "filepath"),
                caption_key=ds_cfg.get("caption_key", "title"),
                sep=ds_cfg.get("sep", "\t"),
            )
        elif dtype == "combined":
            self.train_dataset = build_combined_dataset(
                cc3m_root=ds_cfg.cc3m_root,
                cc3m_csv=ds_cfg.cc3m_csv,
                cc12m_root=ds_cfg.cc12m_root,
                cc12m_csv=ds_cfg.cc12m_csv,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
            )
        elif dtype == "cc3m_wds":
            # WebDataset shards (pixparse/cc3m-wds format).
            # shard_pattern: brace-expansion pattern or list of shard paths/URLs.
            # e.g. "/data/cc3m-wds/cc3m-train-{0000..0575}.tar"
            self.train_dataset = build_cc3m_wds(
                shard_pattern=ds_cfg.shard_pattern,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
                num_samples=_wds_num_samples(ds_cfg.get("num_samples", 2_905_954)),
                resampled=ds_cfg.get("resampled", False),
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        elif dtype == "cc12m_wds":
            self.train_dataset = build_cc12m_wds(
                shard_pattern=ds_cfg.shard_pattern,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
                num_samples=_wds_num_samples(ds_cfg.get("num_samples", 10_968_539)),
                resampled=ds_cfg.get("resampled", False),
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        elif dtype == "combined_wds":
            self.train_dataset = build_combined_wds(
                cc3m_pattern=ds_cfg.cc3m_shard_pattern,
                cc12m_pattern=ds_cfg.cc12m_shard_pattern,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
                num_samples=_wds_num_samples(ds_cfg.get("num_samples", 13_874_493)),
                resampled=ds_cfg.get("resampled", False),
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        elif dtype == "cc3m_hfd":
            # Arrow-cached CC3M loaded via datasets.load_from_disk().
            # Requires prior conversion: scripts/convert_wds_to_hf.py --dataset cc3m
            self.train_dataset = build_cc3m_hfd(
                arrow_dir=ds_cfg.arrow_dir,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
            )
        elif dtype == "cc12m_hfd":
            # Arrow-cached CC12M loaded via datasets.load_from_disk().
            # Requires prior conversion: scripts/convert_wds_to_hf.py --dataset cc12m
            self.train_dataset = build_cc12m_hfd(
                arrow_dir=ds_cfg.arrow_dir,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
            )
        elif dtype == "combined_hfd":
            # Arrow-cached CC3M + CC12M via ConcatDataset.
            # Requires both Arrow dirs to exist (run convert_wds_to_hf.py for each).
            self.train_dataset = build_combined_hfd(
                cc3m_arrow_dir=ds_cfg.cc3m_arrow_dir,
                cc12m_arrow_dir=ds_cfg.cc12m_arrow_dir,
                transforms=self.preprocess_train,
                tokenizer=self.tokenizer,
            )
        elif dtype == "cc3m_wds_dali":
            # DALI-accelerated CC3M WebDataset.
            # nvJPEG decode + GPU augmentation → zero large HtoD memcpy for images.
            # Requires: pip install nvidia-dali-cudaXXX  (CUDA version must match).
            self.train_dataset = build_dali_train_loader(
                shard_pattern=ds_cfg.shard_pattern,
                tokenizer=self.tokenizer,
                preprocess_train=self.preprocess_train,
                num_samples=_wds_num_samples(ds_cfg.get("num_samples", CC3M_TRAIN_SAMPLES)),
                shard_id=self.trainer.global_rank,
                num_shards=self.trainer.world_size,
                batch_size=self.cfg.training.batch_size,
                num_threads=self.cfg.training.get("dali_threads", 4),
                device_id=self.trainer.local_rank,
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        elif dtype == "cc12m_wds_dali":
            # DALI-accelerated CC12M WebDataset.
            self.train_dataset = build_dali_train_loader(
                shard_pattern=ds_cfg.shard_pattern,
                tokenizer=self.tokenizer,
                preprocess_train=self.preprocess_train,
                num_samples=_wds_num_samples(ds_cfg.get("num_samples", CC12M_TRAIN_SAMPLES)),
                shard_id=self.trainer.global_rank,
                num_shards=self.trainer.world_size,
                batch_size=self.cfg.training.batch_size,
                num_threads=self.cfg.training.get("dali_threads", 4),
                device_id=self.trainer.local_rank,
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        elif dtype == "combined_wds_dali":
            # DALI-accelerated combined CC3M + CC12M WebDataset.
            # Shard lists from both patterns are merged and shuffled together.
            from .dali_wds import _expand_paths
            all_paths = (
                _expand_paths(ds_cfg.cc3m_shard_pattern)
                + _expand_paths(ds_cfg.cc12m_shard_pattern)
            )
            self.train_dataset = build_dali_train_loader(
                shard_pattern=all_paths,
                tokenizer=self.tokenizer,
                preprocess_train=self.preprocess_train,
                num_samples=_wds_num_samples(
                    ds_cfg.get("num_samples", CC3M_CC12M_TRAIN_SAMPLES)
                ),
                shard_id=self.trainer.global_rank,
                num_shards=self.trainer.world_size,
                batch_size=self.cfg.training.batch_size,
                num_threads=self.cfg.training.get("dali_threads", 4),
                device_id=self.trainer.local_rank,
                shuffle_buffer=ds_cfg.get("shuffle_buffer", 1000),
                seed=ds_cfg.get("seed", 42),
            )
        else:
            raise ValueError(f"Unknown dataset type '{dtype}'.")

        # ---- Evaluation datasets ----
        if ds_cfg.get("imagenet_val_root"):
            self.val_datasets["imagenet"] = ImageNetDataset(
                root=ds_cfg.imagenet_val_root,
                transform=self.preprocess_val,
                variant="imagenet",
            )
        if ds_cfg.get("imagenet_hf_cache_dir"):
            self.val_datasets["imagenet"] = ImageNetHFDataset(
                hf_cache_dir=ds_cfg.imagenet_hf_cache_dir,
                transform=self.preprocess_val,
                variant="imagenet",
            )
        if ds_cfg.get("imagenet_v2_root"):
            self.val_datasets["imagenet_v2"] = ImageNetDataset(
                root=ds_cfg.imagenet_v2_root,
                transform=self.preprocess_val,
                variant="imagenet_v2",
            )
        if ds_cfg.get("imagenet_r_root"):
            self.val_datasets["imagenet_r"] = ImageNetDataset(
                root=ds_cfg.imagenet_r_root,
                transform=self.preprocess_val,
                variant="imagenet_r",
            )
        if ds_cfg.get("imagenet_sketch_root"):
            self.val_datasets["imagenet_sketch"] = ImageNetDataset(
                root=ds_cfg.imagenet_sketch_root,
                transform=self.preprocess_val,
                variant="imagenet_sketch",
            )
        if ds_cfg.get("mscoco_root"):
            self.val_datasets["mscoco"] = MSCOCODataset(
                data_path=ds_cfg.mscoco_root,
                transform=self.preprocess_val,
                tokenizer=self.tokenizer,
            )
        if ds_cfg.get("flickr30k_root"):
            self.val_datasets["flickr30k"] = Flickr30KDataset(
                data_path=ds_cfg.flickr30k_root,
                transform=self.preprocess_val,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self) -> DataLoader:
        # DALI loaders are self-contained iterators (DDP-sharded, GPU-normalised).
        # Wrapping them in DataLoader would break their protocol; return directly.
        if isinstance(self.train_dataset, DALILoader):
            return self.train_dataset

        # For IterableDataset (e.g. WebDataset): DDP sharding is handled
        # internally via split_by_node/split_by_worker — omit shuffle.
        # For map-style datasets: Lightning automatically wraps with
        # DistributedSampler under DDP (use_distributed_sampler=True by default),
        # so shuffle=True is correct here and Lightning will replace it as needed.
        is_iterable = isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=(not is_iterable),
            num_workers=self.cfg.training.get("workers", 8),
            prefetch_factor=self.cfg.training.get("prefetch_factor", 2),
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        eval_workers = self.cfg.training.get("eval_workers", 4)
        loaders = []
        for dataset in self.val_datasets.values():
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.cfg.training.get("eval_batch_size", 256),
                    shuffle=False,
                    num_workers=eval_workers,
                    pin_memory=True,
                    persistent_workers=(eval_workers > 0),
                )
            )
        return loaders
