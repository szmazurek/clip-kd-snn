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

import sys
import warnings

import torch.utils.data as tud
import webdataset as wds

# Promote PIL's "Truncated File Read" UserWarning to an exception so that
# _decode_warn_and_continue can catch it (with shard URL + sample key context)
# and skip the corrupt sample rather than passing partial image data downstream.
warnings.filterwarnings("error", message="Truncated File Read", category=UserWarning)
warnings.filterwarnings("error", message="Corrupt EXIF data",   category=UserWarning)


def _shard_warn_and_continue(exn: Exception) -> bool:
    """Shard-level error handler: log the missing/corrupt shard path and skip it.

    str(FileNotFoundError) includes the filename, e.g.:
      [Errno 2] No such file or directory: '/path/to/shard.tar'
    """
    print(
        f"[wds] Skipping missing/corrupt shard: {type(exn).__name__}: {exn}",
        file=sys.stderr,
        flush=True,
    )
    return True


def _decode_warn_and_continue(exn: Exception) -> bool:
    """WebDataset decode-stage error handler: log corrupt sample and skip it.

    Uses print(stderr, flush=True) rather than logging so the message is
    guaranteed to appear in SLURM err logs regardless of how Lightning
    configures the root Python logger.
    """
    url = getattr(exn, "url", "?")
    key = getattr(exn, "key", "?")
    cause = exn.__cause__ if exn.__cause__ is not None else exn
    print(
        f"[wds] Skipping corrupt sample [{url} / {key}]: {type(cause).__name__}: {cause}",
        file=sys.stderr,
        flush=True,
    )
    return True

# CC3M shard counts / sample counts (pixparse/cc3m-wds)
CC3M_TRAIN_SHARDS = 576
CC3M_TRAIN_SAMPLES = 2_905_954
CC3M_VAL_SHARDS = 16
CC3M_VAL_SAMPLES = 13_443


class SizedWebDataset(tud.IterableDataset):
    """IterableDataset wrapper that exposes __len__ for Lightning compatibility.

    WebDataset pipelines do not implement __len__, so Lightning cannot compute
    estimated_stepping_batches and the progress bar shows '?'. The pipeline
    already stores nsamples from .with_epoch(), so we read it directly.
    """

    def __init__(self, dataset: wds.WebDataset) -> None:
        self._dataset = dataset

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self) -> int:
        return self._dataset.nsamples  # set by .with_epoch(num_samples)


def build_cc3m_wds(
    shard_pattern: Union[str, list[str]],
    transforms: Callable,
    tokenizer: Callable,
    num_samples: int,
    resampled: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> SizedWebDataset:
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
