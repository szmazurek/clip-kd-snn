"""RefCOCO / RefCOCO+ / RefCOCOg visual grounding dataset.

Port of HiVG/datasets/data_loader.py's TransVGDataset, restricted to the
MSCOCO-backed splits (unc, unc+, gref_umd) and with two simplifications:
  - CLIP tokenizer (open_clip.tokenize) instead of BERT WordPiece.
  - Box-rectangle object mask instead of COCO polygon decoding, so there is
    no pycocotools dependency (matches HiVG's use_seg_mask=False path).

Expects the same `.pth` index files HiVG uses, named
`<dataset>_<split>.pth`, each holding a list of
`(img_file, img_size, bbox_xywh, phrase, _seg)` tuples. Download/prepare
these with scripts/prepare_grounding_data.py.
"""
from __future__ import annotations

import os.path as osp
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.box_utils import bbox_to_mask

_SUPPORTED = {
    "unc": "refcoco",
    "unc+": "refcoco+",
    "gref_umd": "refcocog",
}


class RefCOCODataset(Dataset):
    """RefCOCO-family referring expression grounding dataset.

    Args:
        data_root: Root directory containing `other/images/mscoco/images/train2014`.
        split_root: Directory containing `<dataset>/<dataset>_<split>.pth` index files.
        dataset: One of "unc", "unc+", "gref_umd".
        split: "train", "val", "testA", "testB" (testA/B only for unc/unc+); "test" for gref_umd.
        transform: Callable dict-transform (see datasets/transforms.py).
        tokenizer: Callable mapping str -> LongTensor[context_length] (open_clip tokenizer).
    """

    def __init__(
        self,
        data_root: str,
        split_root: str,
        dataset: str,
        split: str,
        transform: Callable,
        tokenizer: Callable,
    ) -> None:
        if dataset not in _SUPPORTED:
            raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(_SUPPORTED)}")
        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.im_dir = osp.join(data_root, "other", "images", "mscoco", "images", "train2014")

        index_path = osp.join(split_root, dataset, f"{dataset}_{split}.pth")
        if not osp.exists(index_path):
            raise FileNotFoundError(
                f"Annotation index not found: {index_path}. "
                "Run scripts/prepare_grounding_data.py first."
            )
        self.images: list = torch.load(index_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_file, img_size, bbox, phrase, _seg = self.images[idx]
        height, width = img_size["height"], img_size["width"]

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox_xywh = torch.as_tensor(bbox, dtype=torch.float32)
        bbox_xyxy = bbox_xywh.clone()
        bbox_xyxy[2] = bbox_xywh[0] + bbox_xywh[2]
        bbox_xyxy[3] = bbox_xywh[1] + bbox_xywh[3]

        obj_mask = bbox_to_mask(bbox_xywh, height, width)

        sample = {
            "img": img,
            "box": bbox_xyxy,
            "text": phrase.lower(),
            "obj_mask": obj_mask,
        }
        sample = self.transform(sample)

        token_ids = self.tokenizer([sample["text"]])[0]  # [context_length]

        return (
            sample["img"],            # [3, H, W]
            token_ids,                 # [context_length]
            sample["box"],             # [4] normalized (cx,cy,w,h)
            sample["obj_mask"],        # [1, H, W]
            img_file,
            phrase,
        )
