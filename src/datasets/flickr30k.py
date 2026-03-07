"""Flickr30K retrieval evaluation dataset.

Expects the same directory layout as the original repo's RetrievalDataset
(non-COCO branch):
  data_path/
    test_captions.pt     — dict {image_id: [caption1, ..., caption5]}
    test_img_keys.tsv    — one image_id per line
    images/              — <image_id>.jpg files

Ported from src/training/data.py RetrievalDataset (lines 98-139),
Flickr30K branch (line 122).
"""
from __future__ import annotations

import json
import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class Flickr30KDataset(Dataset):
    """Flickr30K test split for image-text retrieval evaluation.

    Returns one image and all its associated captions per __getitem__,
    tokenised and stacked as a single tensor.

    Args:
        data_path: Root directory containing test_captions.pt and images/.
        transform: Evaluation image transform.
        tokenizer: Text tokenizer callable.
    """

    def __init__(
        self,
        data_path: str,
        transform: Callable,
        tokenizer: Callable,
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer

        captions_all: dict = torch.load(os.path.join(data_path, "test_captions.pt"))
        with open(os.path.join(data_path, "test_img_keys.tsv")) as f:
            self.img_keys = [int(k.strip()) for k in f.readlines()]

        self.captions = {k: captions_all[k] for k in self.img_keys}
        if not isinstance(self.captions[self.img_keys[0]], list):
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

    def __len__(self) -> int:
        return len(self.img_keys)

    def __getitem__(self, idx: int):
        img_key = self.img_keys[idx]
        img_path = os.path.join(self.data_path, "images", f"{img_key}.jpg")
        image = self.transform(Image.open(img_path).convert("RGB"))

        raw_captions = self.captions[img_key]
        texts = self.tokenizer(raw_captions)  # (num_captions, 77)
        return image, texts, img_key
