"""CC3M (Conceptual Captions 3M) dataset.

Expects a tab-separated CSV file with image path and caption columns.
Images are assumed to be pre-downloaded locally.

Ported from src/training/data.py CsvDataset (lines 34-56).
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CC3MDataset(Dataset):
    """CC3M image-text dataset backed by a TSV/CSV file.

    Each row maps a local image path to a caption. Tokenisation is
    applied at __getitem__ time so collation is trivial.

    Args:
        data_root: Root directory prepended to image paths.
        csv_path: Path to the TSV/CSV file.
        transforms: Image transform callable (train or eval).
        tokenizer: Text tokenizer callable.
        img_key: Column name for image paths (default "filepath").
        caption_key: Column name for captions (default "title").
        sep: CSV separator (default tab).
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        transforms: Callable,
        tokenizer: Callable,
        img_key: str = "filepath",
        caption_key: str = "title",
        sep: str = "\t",
    ) -> None:
        logging.debug(f"Loading CC3M data from {csv_path}.")
        df = pd.read_csv(csv_path, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.data_root = data_root
        self.transforms = transforms
        self.tokenizer = tokenizer
        logging.debug(f"CC3M dataset loaded: {len(self.captions)} samples.")

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.data_root, str(self.images[idx]))
        image = self.transforms(Image.open(img_path).convert("RGB"))
        text = self.tokenizer([str(self.captions[idx])])[0]
        return image, text
