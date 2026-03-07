"""CC3M WebDataset loader for the pixparse/cc3m-wds HuggingFace format.

Shard layout (https://huggingface.co/datasets/pixparse/cc3m-wds):
  Training:   cc3m-train-{0000..0575}.tar   — 576 shards, ~2,905,954 samples
  Validation: cc3m-validation-{0000..0015}.tar — 16 shards, ~13,443 samples

Each sample inside a shard contains:
  jpg  — raw JPEG bytes decoded to a PIL Image
  txt  — caption string

Returns an IterableDataset yielding (image_tensor, token_tensor) pairs.
DDP-awareness is handled via webdataset's split_by_node/split_by_worker.
"""
from __future__ import annotations

from typing import Callable, Union

import webdataset as wds

# CC3M shard counts / sample counts (pixparse/cc3m-wds)
CC3M_TRAIN_SHARDS = 576
CC3M_TRAIN_SAMPLES = 2_905_954
CC3M_VAL_SHARDS = 16
CC3M_VAL_SAMPLES = 13_443


def build_cc3m_wds(
    shard_pattern: Union[str, list[str]],
    transforms: Callable,
    tokenizer: Callable,
    num_samples: int,
    resampled: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> wds.WebDataset:
    """Build a WebDataset pipeline compatible with the pixparse/cc3m-wds format.

    Args:
        shard_pattern: Brace-expansion pattern or list of shard URLs / local paths.
            Local example:  "/data/cc3m-wds/cc3m-train-{0000..0575}.tar"
            HF Hub stream:  "pipe:curl -s -L https://huggingface.co/datasets/
                             pixparse/cc3m-wds/resolve/main/cc3m-train-{0000..0575}.tar"
        transforms: Image preprocessing callable (open_clip train or eval transforms).
        tokenizer: Text tokeniser callable. Must accept list[str] and return a
            tensor of shape (1, context_length); element [0] is returned per sample.
        num_samples: Total number of samples used to set the epoch length so
            PyTorch Lightning can compute steps-per-epoch correctly.
            For CC3M train use CC3M_TRAIN_SAMPLES (2_905_954).
        resampled: If True, shards are sampled with replacement (ResampledShards).
            Suitable for DDP with an effectively infinite training stream.
            If False (default), shards are split deterministically across nodes
            via wds.split_by_node, giving each node a disjoint subset.
        shuffle_buffer: In-pipeline shuffle buffer size (number of samples).
        seed: Random seed passed to WebDataset for shard shuffling.

    Returns:
        wds.WebDataset (IterableDataset) yielding (image_tensor, text_tensor) tuples.
        Pass this directly to torch.utils.data.DataLoader — do NOT add a sampler.
    """
    dataset = (
        wds.WebDataset(
            shard_pattern,
            resampled=resampled,
            nodesplitter=wds.split_by_node,
            seed=seed,
        )
        .shuffle(shuffle_buffer)
        .decode("pil")                                   # jpg → PIL.Image
        .to_tuple("jpg", "txt")
        .map_tuple(transforms, lambda t: tokenizer([t])[0])
        .with_epoch(num_samples)
    )
    return dataset
