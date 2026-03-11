"""Combined CC3M + CC12M WebDataset loader.

Joins both shard patterns with "::" into a single string passed to
wds.WebDataset, which expands each brace pattern and concatenates all shards.
With shardshuffle=True, shards from both datasets are shuffled together per
epoch, giving uniform data mixing.

Returns a SizedWebDataset (IterableDataset with __len__) yielding
(image_tensor, token_tensor) pairs.
"""
from __future__ import annotations

from typing import Callable, Union

import webdataset as wds

from .cc3m_wds import CC3M_TRAIN_SAMPLES, SizedWebDataset, _decode_warn_and_continue, _shard_warn_and_continue
from .cc12m_wds import CC12M_TRAIN_SAMPLES

CC3M_CC12M_TRAIN_SAMPLES = CC3M_TRAIN_SAMPLES + CC12M_TRAIN_SAMPLES  # 13_874_493


def build_combined_wds(
    cc3m_pattern: Union[str, list[str]],
    cc12m_pattern: Union[str, list[str]],
    transforms: Callable,
    tokenizer: Callable,
    num_samples: int = CC3M_CC12M_TRAIN_SAMPLES,
    resampled: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> SizedWebDataset:
    """Build a combined CC3M + CC12M WebDataset pipeline.

    Shards from both datasets are interleaved and shuffled together,
    providing uniform mixing across the full ~15.3M sample corpus.

    Args:
        cc3m_pattern: Brace-expansion pattern for CC3M shards.
            e.g. "/data/cc3m-wds/cc3m-train-{0000..0575}.tar"
        cc12m_pattern: Brace-expansion pattern for CC12M shards.
            e.g. "/data/cc12m-wds/cc12m-train-{0000..1242}.tar"
        transforms: Image preprocessing callable.
        tokenizer: Text tokeniser callable.
        num_samples: Total epoch length (defaults to CC3M + CC12M combined).
        resampled: If True, use ResampledShards for infinite DDP streams.
        shuffle_buffer: In-pipeline sample shuffle buffer size.
        seed: Random seed for shard shuffling.

    Returns:
        SizedWebDataset (IterableDataset with __len__) yielding
        (image_tensor, text_tensor) tuples.
    """
    # wds.WebDataset expands brace patterns only for string input, not for list
    # input. Join with "::" so expand_urls processes both patterns with braceexpand.
    combined_pattern = f"{cc3m_pattern}::{cc12m_pattern}"
    dataset = (
        wds.WebDataset(
            combined_pattern,
            resampled=resampled,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            seed=seed,
            handler=_shard_warn_and_continue,
        )
        .shuffle(shuffle_buffer)
        .decode("pil", handler=_decode_warn_and_continue)
        .to_tuple("jpg", "txt")
        .map_tuple(transforms, lambda t: tokenizer([t])[0])
        .with_epoch(num_samples)
    )
    return SizedWebDataset(dataset)
