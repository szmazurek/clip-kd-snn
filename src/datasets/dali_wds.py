"""DALI-accelerated WebDataset loaders for CLIP training.

Full GPU pipeline per batch:
  nvJPEG decode  →  GPU random-resized crop  →  GPU normalize

The only data that crosses the PCIe bus is the raw compressed JPEG stream
(≈1-5 KB/image vs ≈590 KB/image for a decoded float32 tensor), eliminating
the large HtoD memcpy that otherwise fragments the CUDA execution timeline.

Text tokenisation runs inside the DALI pipeline via fn.python_function
(batch_processing=True): the tokenizer is called once per batch instead of
once per sample, reducing GIL acquisition overhead from N→1 per batch.

Usage (via CLIPDataModule — set dataset.type to one of the DALI variants):
    build_dali_train_loader(...)  →  DALILoader
    build_dali_val_loader(...)    →  DALILoader

DDP wiring
----------
Each Lightning DDP process builds its own DALI pipeline with its own
  device_id  = local_rank   (CUDA device index on this node)
  shard_id   = global_rank  (data-partition index, 0..world_size-1)
DALI's webdataset reader distributes whole tar shards across the shard_id
range, mirroring wds.split_by_node semantics.

Interplay with Lightning
------------------------
DALILoader implements __iter__ / __next__ / __len__ so Lightning treats it
exactly like a regular DataLoader.  Images arrive already on the correct CUDA
device as float32 tensors; Lightning's transfer_batch_to_device is a no-op
for them.  Tokens are CPU int64 (~153 KB for B=256) and are moved to GPU by
Lightning's transfer_batch_to_device in the normal way.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import torch
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, Resize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_paths(pattern: Union[str, Sequence[str]]) -> list[str]:
    """Expand a brace-expansion shard pattern into a sorted list of real paths.

    Uses braceexpand (installed as a dependency of webdataset).  Falls back to
    glob if braceexpand is unavailable.  Accepts a pre-expanded list unchanged.
    """
    if isinstance(pattern, (list, tuple)):
        return sorted(str(p) for p in pattern)
    import glob

    try:
        import braceexpand

        expanded = list(braceexpand.braceexpand(pattern))
        # braceexpand silently returns the input unchanged when it can't expand
        # zero-padded ranges like {0000..0575} — fall through to glob in that case
        if len(expanded) > 1 or (len(expanded) == 1 and expanded[0] != pattern):
            return sorted(expanded)
    except ImportError:
        pass

    return sorted(glob.glob(pattern))


def _find_index_paths(paths: list[str]) -> list[str] | None:
    """Return a list of .idx paths if ALL shards have a pre-built index, else None.

    DALI's fn.readers.webdataset accepts an index_path list that must be the
    same length as paths.  If even one index is missing we skip the fast path
    so training still starts (just slower on first run).
    """
    import os

    idx_paths = [p.replace(".tar", ".idx") for p in paths]
    if all(os.path.exists(p) for p in idx_paths):
        return idx_paths
    missing = sum(1 for p in idx_paths if not os.path.exists(p))
    print(
        f"[dali_wds] WARNING: {missing}/{len(paths)} index files missing — "
        "startup will be slow.  Run scripts/create_dali_indices.py to pre-build them."
    )
    return None


def _extract_train_params(preprocess_train: Compose) -> dict:
    """Extract preprocessing parameters from an openclip train Compose.

    Parses RandomResizedCrop (image_size, scale, ratio) and Normalize
    (mean, std).  Mean and std are converted to [0, 255] scale so they can
    be passed directly to DALI's crop_mirror_normalize which operates on
    uint8 inputs.
    """
    rrc = next(
        t for t in preprocess_train.transforms if isinstance(t, RandomResizedCrop)
    )
    norm = next(t for t in preprocess_train.transforms if isinstance(t, Normalize))
    size = rrc.size
    return {
        "image_size": size[0] if isinstance(size, (tuple, list)) else int(size),
        "scale": (float(rrc.scale[0]), float(rrc.scale[1])),
        "ratio": (float(rrc.ratio[0]), float(rrc.ratio[1])),
        "mean_255": [float(m) * 255.0 for m in norm.mean],
        "std_255": [float(s) * 255.0 for s in norm.std],
    }


def _extract_val_params(preprocess_val: Compose) -> dict:
    """Extract preprocessing parameters from an openclip val Compose.

    Parses Resize (image_size) and Normalize (mean, std).
    """
    resize = next(t for t in preprocess_val.transforms if isinstance(t, Resize))
    norm = next(t for t in preprocess_val.transforms if isinstance(t, Normalize))
    size = resize.size
    return {
        "image_size": int(size) if isinstance(size, int) else int(size[0]),
        "mean_255": [float(m) * 255.0 for m in norm.mean],
        "std_255": [float(s) * 255.0 for s in norm.std],
    }


# ---------------------------------------------------------------------------
# DALI pipeline definitions
# ---------------------------------------------------------------------------

# Standard context length for all open_clip CLIP models (SimpleTokenizer).
_CLIP_CONTEXT_LENGTH = 77


@pipeline_def
def _clip_train_pipeline_pretok(
    paths: list[str],
    shard_id: int,
    num_shards: int,
    image_size: int,
    scale: tuple,
    ratio: tuple,
    mean_255: list,
    std_255: list,
    shuffle_buffer: int = 1000,
    reader_seed: int = 42,
    index_paths: list[str] | None = None,
    context_length: int = _CLIP_CONTEXT_LENGTH,
):
    """DALI pipeline for CLIP training with pre-tokenized shards.

    Reads .jpg + .bin pairs produced by scripts/pretokenize_wds.py.
    The .bin file contains context_length × int32 token IDs stored as raw
    little-endian bytes (no header).  fn.reinterpret performs a zero-copy
    cast from uint8 to int32 entirely in C++ — no Python, no GIL.

    Execution order:
      1. fn.readers.webdataset     — CPU I/O: compressed JPEG + raw int32 bytes
      2. fn.decoders.image         — nvJPEG (mixed): JPEG bytes → GPU RGB uint8
      3. fn.random_resized_crop    — GPU: bicubic crop + resize to image_size²
      4. fn.crop_mirror_normalize  — GPU: uint8 → float32, (x − mean) / std, CHW
      5. fn.reinterpret            — CPU: uint8 view → int32[context_length]  (zero-copy)
    """
    reader_kwargs = {}
    if index_paths is not None:
        reader_kwargs["index_paths"] = index_paths

    jpegs, raw_tokens = fn.readers.webdataset(
        paths=paths,
        ext=["jpg", "bin"],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        initial_fill=shuffle_buffer,
        seed=reader_seed,
        pad_last_batch=False,
        name="train_reader",
        **reader_kwargs,
    )

    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
    )

    images = fn.random_resized_crop(
        images,
        device="gpu",
        size=[image_size, image_size],
        random_area=[scale[0], scale[1]],
        random_aspect_ratio=[ratio[0], ratio[1]],
        interp_type=types.INTERP_CUBIC,
        antialias=True,
    )

    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        mean=mean_255,
        std=std_255,
        output_layout="CHW",
    )

    # Zero-copy reinterpret: uint8[context_length*4] → int32[context_length].
    # The .bin file is raw little-endian int32 bytes written by pretokenize_wds.py.
    # CLIP vocab size ≤ 49408 fits in int32.  DALILoader.__next__ casts to int64
    # so downstream encode_text (which requires LongTensor) works unchanged.
    tokens = fn.reinterpret(
        raw_tokens,
        dtype=types.INT32,  # type: ignore[attr-defined]  — stubs incomplete
        shape=[context_length],
    )

    return images, tokens


@pipeline_def
def _clip_train_pipeline(
    paths: list[str],
    tokenizer: Callable,
    shard_id: int,
    num_shards: int,
    image_size: int,
    scale: tuple,
    ratio: tuple,
    mean_255: list,
    std_255: list,
    shuffle_buffer: int = 1000,
    reader_seed: int = 42,
    index_paths: list[str] | None = None,
):
    """DALI pipeline for CLIP training.

    Execution order:
      1. fn.readers.webdataset     — CPU I/O: compressed JPEG bytes + raw text
      2. fn.decoders.image         — nvJPEG (mixed): tiny HtoD of JPEG → GPU RGB
      3. fn.random_resized_crop    — GPU: bicubic crop + resize to image_size²
      4. fn.crop_mirror_normalize  — GPU: uint8 → float32, (x − mean) / std, CHW
      5. fn.python_function        — CPU thread: text bytes → fixed-length tokens

    Both outputs are dense uniform tensors so DALIGenericIterator can wrap
    them in standard PyTorch tensors without hitting the ragged-batch error.
    """
    reader_kwargs = {}
    if index_paths is not None:
        reader_kwargs["index_paths"] = index_paths

    jpegs, texts = fn.readers.webdataset(
        paths=paths,
        ext=["jpg", "txt"],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        initial_fill=shuffle_buffer,
        seed=reader_seed,
        pad_last_batch=False,
        name="train_reader",
        **reader_kwargs,
    )

    # nvJPEG decode: compressed JPEG bytes (CPU) → RGB uint8 (GPU, HWC).
    # Only the compressed stream crosses PCIe — O(KB) vs O(MB) for float32.
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
    )

    # Random resized crop: GPU bicubic, log-uniform aspect ratio sampling.
    # Matches torchvision RandomResizedCrop(scale=scale, ratio=ratio, BICUBIC).
    images = fn.random_resized_crop(
        images,
        device="gpu",
        size=[image_size, image_size],
        random_area=[scale[0], scale[1]],
        random_aspect_ratio=[ratio[0], ratio[1]],
        interp_type=types.INTERP_CUBIC,
        antialias=True,
    )

    # Fused cast + normalize: uint8 HWC → float32 CHW, (x − mean_255) / std_255
    # Mathematically identical to: (x/255 − mean_01) / std_01
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        mean=mean_255,
        std=std_255,
        output_layout="CHW",
    )

    # Tokenise captions on CPU inside DALI's operator thread pool.
    # batch_processing=True: DALI passes the whole batch of byte tensors at
    # once, so the tokenizer is called once per batch instead of once per
    # sample.  This reduces GIL acquisition overhead from N to 1 per batch.
    # The function receives a list of N uint8 tensors and must return a list
    # of num_outputs lists, each containing N numpy arrays.
    def _tokenize_batch(texts_bytes_list):
        texts = [
            bytes(t).rstrip(b"\x00").decode("utf-8", errors="replace").strip()
            for t in texts_bytes_list
        ]
        token_array = tokenizer(texts).numpy()  # [N, context_length] int32/int64
        return list(token_array)  # [arr_0, arr_1, …, arr_{N-1}]

    tokens = fn.python_function(
        texts,
        function=_tokenize_batch,
        num_outputs=1,
        batch_processing=True,
    )

    return images, tokens


@pipeline_def
def _clip_val_pipeline(
    paths: list[str],
    tokenizer: Callable,
    shard_id: int,
    num_shards: int,
    image_size: int,
    mean_255: list,
    std_255: list,
    index_paths: list[str] | None = None,
):
    """DALI pipeline for CLIP evaluation.

    Mirrors openclip's preprocess_val (resize_mode='shortest'):
      Resize shortest-edge → image_size  →  centre crop  →  cast + normalize
    """
    reader_kwargs = {}
    if index_paths is not None:
        reader_kwargs["index_paths"] = index_paths

    jpegs, texts = fn.readers.webdataset(
        paths=paths,
        ext=["jpg", "txt"],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        pad_last_batch=True,
        name="val_reader",
        **reader_kwargs,
    )

    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
    )

    # Resize shortest edge → image_size  (mirrors torchvision Resize(image_size))
    images = fn.resize(
        images,
        device="gpu",
        resize_shorter=image_size,
        interp_type=types.INTERP_CUBIC,
        antialias=True,
    )

    # Centre crop + cast + normalize fused into one op
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        crop=[image_size, image_size],
        crop_pos_x=0.5,  # 0 = left/top, 1 = right/bottom, 0.5 = centre
        crop_pos_y=0.5,
        mean=mean_255,
        std=std_255,
        output_layout="CHW",
    )

    def _tokenize_batch(texts_bytes_list):
        texts = [
            bytes(t).rstrip(b"\x00").decode("utf-8", errors="replace").strip()
            for t in texts_bytes_list
        ]
        token_array = tokenizer(texts).numpy()
        return list(token_array)

    tokens = fn.python_function(
        texts,
        function=_tokenize_batch,
        num_outputs=1,
        batch_processing=True,
    )

    return images, tokens


# ---------------------------------------------------------------------------
# DALILoader: DataLoader-compatible wrapper
# ---------------------------------------------------------------------------


class DALILoader:
    """Makes a DALI pipeline iterator look like a DataLoader to Lightning.

    Both pipeline outputs (images, tokens) are dense uniform tensors produced
    entirely inside DALI, so no per-batch Python processing is needed here.

    Batch format returned by __next__:
      images : float32 CUDA tensor  [B, 3, H, W]  already normalised — zero HtoD
      tokens : int64  CPU  tensor   [B, ctx_len]  moved to GPU by Lightning's
               transfer_batch_to_device  (≈153 KB for B=256, ctx_len=77)

    Epoch reset
    -----------
    auto_reset=False: we manage resets explicitly in __next__ so that the
    reset() call always lands on a fully-exhausted iterator.  Using
    auto_reset=True would reset the iterator immediately on StopIteration
    (starting epoch N+1), and then our explicit reset() call below would
    land mid-epoch → DALI warning "epoch not finished" + corrupted prefetch
    queue → zero-filled GPU tensors → NaN loss.

    DDP early-finisher protection: in DDP, shard subsets are distributed by
    whole tars, so different ranks may have slightly different sample counts.
    When a rank exhausts its data before __len__ steps, __next__ catches
    StopIteration, performs a clean reset, and immediately yields the first
    batch of the next data cycle.  This keeps all ranks advancing together
    and prevents NCCL all-gather hangs.
    """

    def __init__(
        self,
        pipeline,
        num_samples: int,
        batch_size: int,
        reader_name: str = "train_reader",
    ) -> None:
        pipeline.build()
        self._iter = DALIGenericIterator(
            [pipeline],
            output_map=["images", "tokens"],
            reader_name=reader_name,  # enables accurate epoch-level iteration
            auto_reset=False,  # we call reset() explicitly (see __next__)
            last_batch_policy=LastBatchPolicy.DROP,
        )
        self._len = max(1, num_samples // batch_size)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            # Rank exhausted its shard subset (early finisher in DDP) or
            # Lightning is iterating past the natural epoch boundary.
            # DALI's epoch is now fully done, so reset() is safe here.
            self._iter.reset()
            batch = next(self._iter)

        images = batch[0]["images"].detach().contiguous()
        # .long(): no-op for int64 (python_function path), int32→int64 for
        # pretokenized path — encode_text requires LongTensor for embedding lookup.
        tokens = batch[0]["tokens"].detach().contiguous().long()
        return images, tokens


# ---------------------------------------------------------------------------
# Public factory functions (called from CLIPDataModule.setup())
# ---------------------------------------------------------------------------


def build_dali_train_loader(
    shard_pattern: Union[str, Sequence[str]],
    tokenizer: Callable,
    preprocess_train: Compose,
    num_samples: int,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    shuffle_buffer: int = 1000,
    seed: int = 42,
    prefetch_queue_depth: int = 2,
) -> DALILoader:
    """Build a DALI training loader from a WDS shard pattern.

    Args:
        shard_pattern: Brace-expansion pattern or explicit list of .tar paths.
        tokenizer:     open_clip tokenizer (batch-callable, returns [B, ctx] tensor).
        preprocess_train: openclip preprocess_train Compose.  Used only to
            extract image_size, RRC scale/ratio, and normalise mean/std — the
            actual Compose is never applied; DALI replaces its function entirely.
        num_samples:   Per-rank sample count for one epoch (total // world_size).
        shard_id:      Global rank of this process  (0 .. world_size − 1).
        num_shards:    Total DDP world size.
        batch_size:    Samples per batch.
        num_threads:   CPU threads for DALI's reader/prefetch stage.  Much lower
            than PyTorch DataLoader workers because heavy work is on GPU.
        device_id:     CUDA device index on this node  (= local_rank).
        shuffle_buffer: Depth of DALI's sample-level shuffle buffer.
        seed:          Random seed for shard and sample shuffling.
    """
    params = _extract_train_params(preprocess_train)
    paths = _expand_paths(shard_pattern)
    index_paths = _find_index_paths(paths)

    pipeline = _clip_train_pipeline(
        paths=paths,
        tokenizer=tokenizer,
        shard_id=shard_id,
        num_shards=num_shards,
        image_size=params["image_size"],
        scale=params["scale"],
        ratio=params["ratio"],
        mean_255=params["mean_255"],
        std_255=params["std_255"],
        shuffle_buffer=shuffle_buffer,
        reader_seed=seed,
        index_paths=index_paths,
        # pipeline_def standard kwargs:
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )

    return DALILoader(pipeline, num_samples, batch_size, reader_name="train_reader")


def build_dali_val_loader(
    shard_pattern: Union[str, Sequence[str]],
    tokenizer: Callable,
    preprocess_val: Compose,
    num_samples: int,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
) -> DALILoader:
    """Build a DALI validation loader from a WDS shard pattern."""
    params = _extract_val_params(preprocess_val)
    paths = _expand_paths(shard_pattern)
    index_paths = _find_index_paths(paths)

    pipeline = _clip_val_pipeline(
        paths=paths,
        tokenizer=tokenizer,
        shard_id=shard_id,
        num_shards=num_shards,
        image_size=params["image_size"],
        mean_255=params["mean_255"],
        std_255=params["std_255"],
        index_paths=index_paths,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )

    return DALILoader(pipeline, num_samples, batch_size, reader_name="val_reader")


def build_dali_train_loader_pretok(
    shard_pattern: Union[str, Sequence[str]],
    preprocess_train: Compose,
    num_samples: int,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    shuffle_buffer: int = 1000,
    seed: int = 42,
    context_length: int = _CLIP_CONTEXT_LENGTH,
    prefetch_queue_depth: int = 2,
) -> DALILoader:
    """Build a DALI training loader from pre-tokenized WDS shards (.jpg + .bin).

    Shards must be produced by scripts/pretokenize_wds.py.
    No Python function in the pipeline — tokens are read and reinterpreted
    entirely in C++ via fn.reinterpret (zero GIL, zero Python overhead).

    Args:
        shard_pattern:  Brace-expansion pattern or explicit list of .tar paths.
        preprocess_train: openclip preprocess_train Compose (image params only).
        num_samples:    Per-rank sample count for one epoch (total // world_size).
        shard_id:       Global rank of this process  (0 .. world_size − 1).
        num_shards:     Total DDP world size.
        batch_size:     Samples per batch.
        num_threads:    CPU threads for DALI's reader/prefetch stage.
        device_id:      CUDA device index on this node  (= local_rank).
        shuffle_buffer: Depth of DALI's sample-level shuffle buffer.
        seed:           Random seed for shard and sample shuffling.
        context_length: Token sequence length (must match pretokenize_wds.py --context-length).
    """
    params = _extract_train_params(preprocess_train)
    paths = _expand_paths(shard_pattern)
    index_paths = _find_index_paths(paths)

    pipeline = _clip_train_pipeline_pretok(
        paths=paths,
        shard_id=shard_id,
        num_shards=num_shards,
        image_size=params["image_size"],
        scale=params["scale"],
        ratio=params["ratio"],
        mean_255=params["mean_255"],
        std_255=params["std_255"],
        shuffle_buffer=shuffle_buffer,
        reader_seed=seed,
        index_paths=index_paths,
        context_length=context_length,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )

    return DALILoader(pipeline, num_samples, batch_size, reader_name="train_reader")
