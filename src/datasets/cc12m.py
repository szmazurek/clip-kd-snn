"""CC12M (Conceptual Captions 12M) dataset.

Identical structure to CC3MDataset but uses cc12m-specific default column
names. Kept as a separate file so Hydra configs can target it by name.
"""
from __future__ import annotations

from .cc3m import CC3MDataset


class CC12MDataset(CC3MDataset):
    """CC12M image-text dataset.

    Thin subclass of CC3MDataset; the only difference is the default
    column names used in CC12M CSV files.
    """

    def __init__(self, data_root, csv_path, transforms, tokenizer,
                 img_key="filepath", caption_key="title", sep="\t"):
        super().__init__(
            data_root=data_root,
            csv_path=csv_path,
            transforms=transforms,
            tokenizer=tokenizer,
            img_key=img_key,
            caption_key=caption_key,
            sep=sep,
        )
