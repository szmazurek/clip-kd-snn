"""Combined CC3M + CC12M dataset (ConcatDataset wrapper)."""
from __future__ import annotations

from typing import Callable

from torch.utils.data import ConcatDataset

from .cc3m import CC3MDataset
from .cc12m import CC12MDataset


def build_combined_dataset(
    cc3m_root: str,
    cc3m_csv: str,
    cc12m_root: str,
    cc12m_csv: str,
    transforms: Callable,
    tokenizer: Callable,
) -> ConcatDataset:
    """Build a concatenated CC3M + CC12M training dataset.

    Args:
        cc3m_root: Root directory for CC3M images.
        cc3m_csv: Path to CC3M CSV/TSV file.
        cc12m_root: Root directory for CC12M images.
        cc12m_csv: Path to CC12M CSV/TSV file.
        transforms: Shared image transform callable.
        tokenizer: Shared text tokenizer callable.

    Returns:
        ConcatDataset of CC3M and CC12M.
    """
    cc3m = CC3MDataset(cc3m_root, cc3m_csv, transforms, tokenizer)
    cc12m = CC12MDataset(cc12m_root, cc12m_csv, transforms, tokenizer)
    return ConcatDataset([cc3m, cc12m])
