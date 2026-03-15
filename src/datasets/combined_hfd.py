"""Combined CC3M + CC12M Arrow-format loader (HuggingFace datasets, Lustre-safe).

Concatenates the two Arrow-cached datasets using torch.utils.data.ConcatDataset.
ConcatDataset is map-style with a correct __len__, so Lightning's DistributedSampler
handles DDP sharding uniformly across the combined index space.

Requires both Arrow directories to exist (run convert_wds_to_hf.py for each):
  python scripts/convert_wds_to_hf.py --dataset cc3m  --output-dir $SCRATCH/cc3m-hf  --num-shards 128
  python scripts/convert_wds_to_hf.py --dataset cc12m --output-dir $SCRATCH/cc12m-hf --num-shards 512
"""
from __future__ import annotations

from typing import Callable

from torch.utils.data import ConcatDataset

from .cc3m_hfd  import build_cc3m_hfd
from .cc12m_hfd import build_cc12m_hfd

CC3M_TRAIN_SAMPLES  = 2_905_954
CC12M_TRAIN_SAMPLES = 10_968_539
COMBINED_TRAIN_SAMPLES = CC3M_TRAIN_SAMPLES + CC12M_TRAIN_SAMPLES  # 13_874_493


def build_combined_hfd(
    cc3m_arrow_dir: str,
    cc12m_arrow_dir: str,
    transforms: Callable,
    tokenizer: Callable,
) -> ConcatDataset:
    """Build a concatenated CC3M + CC12M Arrow-cached dataset.

    Args:
        cc3m_arrow_dir: Directory created by convert_wds_to_hf.py for CC3M.
        cc12m_arrow_dir: Directory created by convert_wds_to_hf.py for CC12M.
        transforms: Shared image preprocessing callable.
        tokenizer: Shared text tokenizer callable.

    Returns:
        ConcatDataset([CC3MHFDataset, CC12MHFDataset]) — map-style, __len__ defined.
        Lightning's DistributedSampler handles DDP sharding automatically.
    """
    cc3m  = build_cc3m_hfd(cc3m_arrow_dir,   transforms, tokenizer)
    cc12m = build_cc12m_hfd(cc12m_arrow_dir,  transforms, tokenizer)
    return ConcatDataset([cc3m, cc12m])
