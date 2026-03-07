"""Unit tests for dataset classes.

Tests use synthetic data (temporary CSV files and images) so they
run without actual CC3M/ImageNet data.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.cc3m import CC3MDataset
from src.datasets.combined import build_combined_dataset
from src.datasets.transforms import get_train_transforms


def _dummy_tokenizer(texts):
    """Return zero tensors of shape (len(texts), 77)."""
    return torch.zeros(len(texts), 77, dtype=torch.long)


def _create_dummy_csv_and_images(tmp_dir: Path, n: int = 4):
    """Create n dummy JPEG images and a TSV file pointing to them."""
    img_dir = tmp_dir / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        img_path = img_dir / f"img_{i:04d}.jpg"
        Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(img_path)
        rows.append({"filepath": f"images/img_{i:04d}.jpg", "title": f"caption {i}"})

    csv_path = tmp_dir / "data.tsv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "title"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return str(tmp_dir), str(csv_path)


class TestCC3MDataset:
    def test_length(self, tmp_path):
        data_root, csv_path = _create_dummy_csv_and_images(tmp_path, n=6)
        dataset = CC3MDataset(
            data_root=data_root,
            csv_path=csv_path,
            transforms=get_train_transforms(64),
            tokenizer=_dummy_tokenizer,
        )
        assert len(dataset) == 6

    def test_item_shapes(self, tmp_path):
        data_root, csv_path = _create_dummy_csv_and_images(tmp_path, n=4)
        dataset = CC3MDataset(
            data_root=data_root,
            csv_path=csv_path,
            transforms=get_train_transforms(64),
            tokenizer=_dummy_tokenizer,
        )
        image, text = dataset[0]
        assert image.shape == (3, 64, 64), f"Unexpected image shape: {image.shape}"
        assert text.shape == (77,), f"Unexpected text shape: {text.shape}"

    def test_item_types(self, tmp_path):
        data_root, csv_path = _create_dummy_csv_and_images(tmp_path, n=2)
        dataset = CC3MDataset(
            data_root=data_root,
            csv_path=csv_path,
            transforms=get_train_transforms(64),
            tokenizer=_dummy_tokenizer,
        )
        image, text = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(text, torch.Tensor)
        assert text.dtype == torch.long


class TestCombinedDataset:
    def test_combined_length_is_sum(self, tmp_path):
        cc3m_dir = tmp_path / "cc3m"
        cc3m_dir.mkdir()
        cc12m_dir = tmp_path / "cc12m"
        cc12m_dir.mkdir()

        _, cc3m_csv = _create_dummy_csv_and_images(cc3m_dir, n=3)
        _, cc12m_csv = _create_dummy_csv_and_images(cc12m_dir, n=5)

        combined = build_combined_dataset(
            cc3m_root=str(cc3m_dir),
            cc3m_csv=cc3m_csv,
            cc12m_root=str(cc12m_dir),
            cc12m_csv=cc12m_csv,
            transforms=get_train_transforms(64),
            tokenizer=_dummy_tokenizer,
        )
        assert len(combined) == 8  # 3 + 5
