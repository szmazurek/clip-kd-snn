"""CC12M WebDataset loader for the pixparse/cc12m-wds HuggingFace format.

Shard layout (https://huggingface.co/datasets/pixparse/cc12m-wds):
  Training: cc12m-train-{0000..1242}.tar — 1243 shards, ~12,423,374 samples

Each sample inside a shard contains:
  jpg  — raw JPEG bytes decoded to a PIL Image
  txt  — caption string

Returns a SizedWebDataset (IterableDataset with __len__) yielding
(image_tensor, token_tensor) pairs.
DDP-awareness is handled via webdataset's split_by_node/split_by_worker.
"""
from __future__ import annotations

from typing import Callable, Union

import webdataset as wds

from .cc3m_wds import SizedWebDataset, _decode_warn_and_continue, _shard_warn_and_continue

# CC12M shard counts / sample counts (local copy — shards 0000..1101)
CC12M_TRAIN_SHARDS = 1102
CC12M_TRAIN_SAMPLES = 10_968_539  # pixparse/cc12m-wds HF docs (train split)


def build_cc12m_wds(
    shard_pattern: Union[str, list[str]],
    transforms: Callable,
    tokenizer: Callable,
    num_samples: int,
    resampled: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> SizedWebDataset:
    """Build a WebDataset pipeline compatible with the pixparse/cc12m-wds format.

    Args:
        shard_pattern: Brace-expansion pattern or list of shard URLs / local paths.
            Local example: "/data/cc12m-wds/cc12m-train-{0000..1101}.tar"
        transforms: Image preprocessing callable (open_clip train or eval transforms).
        tokenizer: Text tokeniser callable. Must accept list[str] and return a
            tensor of shape (1, context_length); element [0] is returned per sample.
        num_samples: Total number of samples used to set the epoch length so
            PyTorch Lightning can compute steps-per-epoch correctly.
            For CC12M train use CC12M_TRAIN_SAMPLES (12_423_374).
        resampled: If True, shards are sampled with replacement (ResampledShards).
        shuffle_buffer: In-pipeline shuffle buffer size (number of samples).
        seed: Random seed passed to WebDataset for shard shuffling.

    Returns:
        SizedWebDataset (IterableDataset with __len__) yielding
        (image_tensor, text_tensor) tuples.
    """
    dataset = (
        wds.WebDataset(
            shard_pattern,
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
