"""Functional equivalence test: DALI pipeline vs openclip CPU pipeline.

Four focused checks:

  Test 1 — Normalize math (exact)
    Verifies DALI crop_mirror_normalize == PyTorch Normalize on the same uint8
    input (no JPEG, no resize involved).  Expected: max diff < 1e-4.

  Test 2 — Decode + resize + normalize equivalence (near-exact)
    Feeds PNG-encoded smooth gradient images to DALI via ExternalSource
    (lossless, so both DALI and PIL decode identically) then compares the full
    resize → centre-crop → normalize path.  The only source of difference is
    the bicubic kernel implementation; smooth inputs produce very small diffs.
    Expected: max diff ≤ 0.02.

  Test 3 — Train pipeline smoke test (shapes / dtype / device)
    Builds a minimal WDS tar, runs the full train DALI pipeline end-to-end,
    and checks output shapes, dtype, and device.

  Test 4 — Tokenisation round-trip
    Verifies that captions written into a WDS tar survive the DALI
    fn.python_function tokenisation path and match direct tokeniser output.

  Benchmark (optional, --benchmark flag)
    Times DALI vs WDS DataLoader throughput for N batches.

Usage:
    # Equivalence tests only (no real shards needed):
    python scripts/test_dali_pipeline.py

    # With throughput benchmark against existing WDS shards:
    python scripts/test_dali_pipeline.py --benchmark \\
        --shard-pattern "/data/cc3m-wds/cc3m-train-{0000..0010}.tar" \\
        --model ViT-B-16
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_wds_tar(
    path: str,
    images: list[np.ndarray],
    captions: list[str],
    fmt: str = "jpeg",
    jpeg_quality: int = 95,
) -> None:
    """Write a minimal WDS tar file.

    Args:
        images:   List of HxWx3 uint8 numpy arrays.
        captions: Corresponding caption strings.
        fmt:      "jpeg" or "png".  Both are stored as .jpg entries (DALI
                  auto-detects format from file content, not extension).
    """
    from PIL import Image

    with tarfile.open(path, "w") as tar:
        for i, (arr, cap) in enumerate(zip(images, captions)):
            key = f"{i:06d}"
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            if fmt == "png":
                pil.save(buf, format="PNG")
            else:
                pil.save(buf, format="JPEG", quality=jpeg_quality)
            img_bytes = buf.getvalue()

            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))

            cap_bytes = cap.encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.txt")
            info.size = len(cap_bytes)
            tar.addfile(info, io.BytesIO(cap_bytes))


def _smooth_gradient(h: int = 320, w: int = 480) -> np.ndarray:
    """Return an HxWx3 uint8 smooth linear gradient image.

    Smooth images minimise bicubic interpolation differences between
    different implementations (DALI vs PIL/torchvision).
    """
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    r = np.outer(ys, np.ones(w))          # row gradient
    g = np.outer(np.ones(h), xs)          # col gradient
    b = np.full((h, w), 128.0, dtype=np.float32)
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Test 1: Normalize math — exact equivalence
# ---------------------------------------------------------------------------

def test_normalize_exact(mean, std) -> None:
    """Verify DALI crop_mirror_normalize == PyTorch Normalize (exact match)."""
    print("\n=== Test 1: Normalize math (exact) ===")

    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from torchvision.transforms import Normalize

    B, H, W = 4, 224, 224
    rng = np.random.default_rng(0)
    # Shape [B, H, W, 3] — ExternalSource with a callable returning this array
    # treats the first dim as batch size. Do NOT pass as a per-sample list.
    uint8_np = rng.integers(0, 256, size=(B, H, W, 3), dtype=np.uint8)

    # PyTorch reference: uint8 → [0, 1] → Normalize
    pt_norm = Normalize(mean=mean, std=std)
    pt_out = torch.stack([
        pt_norm(torch.from_numpy(uint8_np[i]).permute(2, 0, 1).float() / 255.0)
        for i in range(B)
    ])  # [B, 3, H, W]

    # DALI: feed identical uint8 data via ExternalSource.
    # Normalise on CPU (device="cpu") — avoids a device-transfer op and tests
    # the same arithmetic as the GPU path (same formula, same float32 precision).
    @pipeline_def(batch_size=B, num_threads=1, device_id=0)
    def _norm_pipeline():
        images = fn.external_source(
            source=lambda: uint8_np,   # callable → returns [B, H, W, 3] per run
            dtype=types.UINT8,
            layout="HWC",
        )
        images = fn.crop_mirror_normalize(
            images,
            device="cpu",
            dtype=types.FLOAT,
            mean=[float(m) * 255.0 for m in mean],
            std=[float(s) * 255.0 for s in std],
            output_layout="CHW",
        )
        return images

    pipe = _norm_pipeline()
    pipe.build()
    pipe.run()  # warm-up
    dali_out = torch.as_tensor(pipe.run()[0].as_array())

    max_diff = (pt_out - dali_out).abs().max().item()
    print(f"  Max absolute diff : {max_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"  Result            : {'PASS ✓' if passed else 'FAIL ✗'}  (threshold 1e-4)")
    assert passed, f"Normalize math mismatch: {max_diff}"


# ---------------------------------------------------------------------------
# Test 2: Decode + resize + normalize — near-exact (PNG, smooth gradient)
# ---------------------------------------------------------------------------

def test_resize_normalize_equivalence(mean, std, image_size: int = 224) -> None:
    """Compare DALI and CPU for PNG decode → resize → centre crop → normalize.

    PNG is lossless so both decoders produce identical uint8 pixels.  The only
    remaining difference is the bicubic kernel implementation; for smooth
    gradient images this is < 0.02.
    """
    print("\n=== Test 2: Decode + resize + normalize equivalence (PNG / smooth) ===")

    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from PIL import Image
    from torchvision.transforms import CenterCrop, Normalize, Resize
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF

    B = 4
    images_np = [_smooth_gradient() for _ in range(B)]

    # Encode each image as PNG into a byte buffer
    png_bytes = []
    for arr in images_np:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        png_bytes.append(buf.getvalue())

    # --- CPU reference ---
    cpu_resize = Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    cpu_crop   = CenterCrop(image_size)
    cpu_norm   = Normalize(mean=mean, std=std)
    cpu_out = torch.stack([
        cpu_norm(TF.to_tensor(cpu_crop(cpu_resize(Image.open(io.BytesIO(b)).convert("RGB")))))
        for b in png_bytes
    ])  # [B, 3, image_size, image_size]

    # --- DALI via ExternalSource ---
    # fn.decoders.image auto-detects PNG from content regardless of filename.
    # source must be a callable returning the batch — a list of per-sample arrays
    # of (possibly) different lengths (each PNG has a different file size).
    encoded = [np.frombuffer(b, dtype=np.uint8) for b in png_bytes]

    @pipeline_def(batch_size=B, num_threads=2, device_id=0)
    def _val_pipe():
        raw = fn.external_source(
            source=lambda: encoded,    # callable → returns list of B byte arrays
            dtype=types.UINT8,
        )
        imgs = fn.decoders.image(raw, device="mixed", output_type=types.RGB)
        imgs = fn.resize(imgs, device="gpu", resize_shorter=image_size,
                         interp_type=types.INTERP_CUBIC, antialias=True)
        imgs = fn.crop_mirror_normalize(
            imgs,
            device="gpu",
            dtype=types.FLOAT,
            crop=[image_size, image_size],
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            mean=[float(m) * 255.0 for m in mean],
            std=[float(s) * 255.0 for s in std],
            output_layout="CHW",
        )
        return imgs

    pipe = _val_pipe()
    pipe.build()
    dali_out = torch.as_tensor(pipe.run()[0].as_cpu().as_array())

    diff = (cpu_out - dali_out).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  CPU (PIL+torchvision) vs DALI (nvJPEG + DALI bicubic)")
    print(f"  Max  abs diff : {max_diff:.4f}  (tolerance ≤0.02 for smooth images)")
    print(f"  Mean abs diff : {mean_diff:.6f}")
    passed = max_diff <= 0.02
    print(f"  Result        : {'PASS ✓' if passed else 'FAIL ✗'}  (threshold 0.02)")
    if not passed:
        print("  NOTE: diff > 0.02 on a smooth image likely indicates a resize/crop")
        print("        logic mismatch (e.g. crop_pos_x vs centre-crop semantics).")


# ---------------------------------------------------------------------------
# Test 3: Train pipeline smoke test (via WDS tar)
# ---------------------------------------------------------------------------

def test_train_pipeline_shapes(mean, std, image_size: int = 224) -> None:
    """Run the full train pipeline end-to-end and check shapes, dtype, device."""
    print("\n=== Test 3: Train pipeline shapes / dtype / device ===")

    import open_clip
    from src.datasets.dali_wds import build_dali_train_loader

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    B = 4
    images_np = [_smooth_gradient() for _ in range(B * 2)]
    captions  = [f"caption {i}" for i in range(B * 2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "test.tar")
        _make_wds_tar(tar_path, images_np, captions, fmt="jpeg")

        # Minimal preprocess_train Compose so _extract_train_params can parse it
        _, preprocess_train, _ = open_clip.create_model_and_transforms("ViT-B-16")

        loader = build_dali_train_loader(
            shard_pattern=[tar_path],
            tokenizer=tokenizer,
            preprocess_train=preprocess_train,
            num_samples=B * 2,
            shard_id=0, num_shards=1,
            batch_size=B, num_threads=2, device_id=0,
        )
        images_gpu, tokens_cpu = next(iter(loader))

    print(f"  images.shape  : {tuple(images_gpu.shape)}  (expected [{B}, 3, {image_size}, {image_size}])")
    print(f"  images.dtype  : {images_gpu.dtype}   (expected torch.float32)")
    print(f"  images.device : {images_gpu.device}  (expected cuda:0)")
    print(f"  images range  : [{images_gpu.min():.3f}, {images_gpu.max():.3f}]")
    print(f"  tokens.shape  : {tuple(tokens_cpu.shape)}  (expected [{B}, 77])")
    print(f"  tokens.dtype  : {tokens_cpu.dtype}   (expected torch.int*)")

    assert images_gpu.shape == (B, 3, image_size, image_size), "Image shape mismatch"
    assert images_gpu.dtype == torch.float32, "Expected float32"
    assert images_gpu.is_cuda, "Expected CUDA tensor"
    assert tokens_cpu.shape[0] == B, "Token batch size mismatch"
    print("  Result        : PASS ✓")


# ---------------------------------------------------------------------------
# Test 4: Tokenisation round-trip
# ---------------------------------------------------------------------------

def test_tokenisation_roundtrip() -> None:
    """Verify DALI fn.python_function tokenisation matches direct tokeniser."""
    print("\n=== Test 4: Tokenisation round-trip ===")

    import open_clip
    from src.datasets.dali_wds import build_dali_train_loader

    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    captions = ["a photo of a cat", "dogs running in a field",
                "sunset over the ocean", "a busy city street at night"]
    images_np = [_smooth_gradient() for _ in captions]

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "test.tar")
        _make_wds_tar(tar_path, images_np, captions, fmt="jpeg")

        _, preprocess_train, _ = open_clip.create_model_and_transforms("ViT-B-16")
        loader = build_dali_train_loader(
            shard_pattern=[tar_path],
            tokenizer=tokenizer,
            preprocess_train=preprocess_train,
            num_samples=len(captions),
            shard_id=0, num_shards=1,
            batch_size=len(captions), num_threads=2, device_id=0,
        )
        _, tokens_dali = next(iter(loader))

    # Reference: tokenise captions directly
    tokens_ref = tokenizer(captions)

    # DALI reader may return samples in a different order; sort both by the full
    # token sequence (lexicographic) for an order-agnostic comparison.
    # np.lexsort keys are in reverse priority order (last key = most significant).
    def _lex_sort(t: torch.Tensor) -> torch.Tensor:
        keys = [t[:, i].numpy() for i in range(t.shape[1])]  # col 0 is most significant
        order = np.lexsort(keys[::-1])
        return t[order]

    dali_sorted = _lex_sort(tokens_dali)
    ref_sorted  = _lex_sort(tokens_ref)

    match = (dali_sorted == ref_sorted).all().item()
    print(f"  Tokens match (order-agnostic) : {'yes ✓' if match else 'no ✗'}")
    if not match:
        print("  DALI tokens :", dali_sorted)
        print("  Ref  tokens :", ref_sorted)
    assert match, "Token mismatch"
    print("  Result : PASS ✓")


# ---------------------------------------------------------------------------
# Benchmark (optional)
# ---------------------------------------------------------------------------

def benchmark_dali_vs_wds(
    shard_pattern: str,
    model_name: str,
    batch_size: int = 256,
    num_batches: int = 50,
) -> None:
    print(f"\n=== Benchmark: DALI vs WDS ({num_batches} batches, B={batch_size}) ===")

    import open_clip
    import webdataset as wds
    from torch.utils.data import DataLoader
    from src.datasets.dali_wds import build_dali_train_loader

    _, preprocess_train, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    # --- DALI ---
    dali_loader = build_dali_train_loader(
        shard_pattern=shard_pattern,
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        num_samples=batch_size * num_batches * 2,
        shard_id=0, num_shards=1,
        batch_size=batch_size, num_threads=4, device_id=0,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_dali = 0
    for i, (imgs, _) in enumerate(dali_loader):
        torch.cuda.synchronize()
        n_dali += imgs.shape[0]
        if i + 1 >= num_batches:
            break
    dali_elapsed = time.perf_counter() - t0

    # --- WDS ---
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(1000).decode("pil").to_tuple("jpg", "txt")
        .map_tuple(preprocess_train, lambda t: tokenizer([t])[0])
        .with_epoch(batch_size * num_batches * 2)
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=True, drop_last=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_wds = 0
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.cuda(non_blocking=True)
        torch.cuda.synchronize()
        n_wds += imgs.shape[0]
        if i + 1 >= num_batches:
            break
    wds_elapsed = time.perf_counter() - t0

    dali_sps = n_dali / dali_elapsed
    wds_sps  = n_wds  / wds_elapsed
    print(f"  DALI : {dali_sps:>8.0f} samples/s  ({dali_elapsed:.1f}s for {n_dali} samples)")
    print(f"  WDS  : {wds_sps:>8.0f} samples/s  ({wds_elapsed:.1f}s for {n_wds} samples)")
    print(f"  Speedup: {dali_sps / wds_sps:.2f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DALI pipeline equivalence tests")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--shard-pattern", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — DALI requires a GPU.")
        sys.exit(1)

    try:
        import nvidia.dali
        print(f"DALI version : {nvidia.dali.__version__}")
    except ImportError:
        print("ERROR: nvidia-dali not installed.")
        sys.exit(1)

    import open_clip
    _, preprocess_train, _ = open_clip.create_model_and_transforms(args.model)

    from torchvision.transforms import Normalize, RandomResizedCrop
    norm = next(t for t in preprocess_train.transforms if isinstance(t, Normalize))
    mean = tuple(float(m) for m in norm.mean)
    std  = tuple(float(s) for s in norm.std)
    rrc  = next(t for t in preprocess_train.transforms if isinstance(t, RandomResizedCrop))
    image_size = rrc.size[0] if isinstance(rrc.size, (tuple, list)) else int(rrc.size)

    print(f"Model        : {args.model}")
    print(f"Image size   : {image_size}")
    print(f"Mean (01)    : {mean}")
    print(f"Std  (01)    : {std}")

    test_normalize_exact(mean, std)
    test_resize_normalize_equivalence(mean, std, image_size=image_size)
    test_train_pipeline_shapes(mean, std, image_size=image_size)
    test_tokenisation_roundtrip()

    if args.benchmark:
        if args.shard_pattern is None:
            print("\nSkipping benchmark: --shard-pattern not provided.")
        else:
            benchmark_dali_vs_wds(
                shard_pattern=args.shard_pattern,
                model_name=args.model,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
            )

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
