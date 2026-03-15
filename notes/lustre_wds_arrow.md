# WebDataset vs Arrow on Lustre FS — Observations & Fix

## Problem: Truncated File Reads on Lustre

During training on this machine (Lustre filesystem), the WebDataset-based loaders
(`cc3m_wds`, `cc12m_wds`, `combined_wds`) produced frequent warnings:

```
WARNING: Truncated file read in <worker N>
```

This caused samples to be silently skipped, degraded throughput, and introduced
noise in the training loss curve. The **same code and data did not produce these
warnings on a non-Lustre machine** (local ext4 / NFS).

## Root Cause: Lustre MDS Saturation from Concurrent Tar Opens

The HuggingFace hub cache stores dataset files as blobs with a symlink layer:

```
$SCRATCH/.cache/hub/datasets--pixparse--cc3m-wds/
  snapshots/<commit-hash>/
    cc3m-train-0000.tar -> ../../blobs/<sha256>
    cc3m-train-0001.tar -> ../../blobs/<sha256>
    ...
```

During training with 8 GPUs × 12 DataLoader workers = **96 concurrent worker
processes**, each worker opens a different tar shard at the start of each epoch.
Every `open()` on a symlink triggers **2 Lustre MDS (Metadata Server) operations**:
one to resolve the symlink, one to open the blob file. 96 workers doing this
simultaneously saturates the MDS, causing:

- MDS request timeouts → partial reads → "Truncated file read" in tarfile module
- Non-deterministic: depends on MDS load, network congestion, number of workers

**Why it works elsewhere:** On local ext4 or NFS, symlink resolution is
sub-microsecond (local inode lookup). On Lustre, every MDS op is a network RPC
to a centralised metadata server — latency is 0.1–1 ms, and throughput is
limited. With 96 concurrent workers, the MDS becomes the bottleneck.

## Solution: Convert to Arrow Format

One-time conversion using `scripts/convert_wds_to_hf.py` writes the tars into
HuggingFace Arrow shards:

```bash
python scripts/convert_wds_to_hf.py \
    --dataset cc3m \
    --hub-cache $SCRATCH/.cache/hub \
    --output-dir $SCRATCH/cc3m-hf \
    --num-shards 128
```

Training then loads with `datasets.load_from_disk()` which uses `mmap()`:

- **No tar opens during training** — Arrow files are mmap'd once at DataLoader
  init, not per-sample
- **128 files vs 576 tar shards** — far fewer simultaneous file descriptors
- **No symlinks** — Arrow dir contains real files, eliminating MDS symlink ops
- **Large sequential reads** — mmap + OS page cache handles read-ahead; Lustre
  is optimised for large sequential I/O, not random small metadata ops

## Observed Results (as of 2026-03-13)

| Metric | WDS (tar) | Arrow (mmap) |
|--------|-----------|--------------|
| Truncated read warnings | frequent | **none** |
| Per-GPU throughput | baseline | ~2× slower |
| Scaling (2× GPUs) | sublinear | **~2× (linear)** |
| OOM at 30 GB RAM | no | yes* |

*OOM was caused by a secondary bug (see below), now fixed.

The ~2× per-GPU slowdown comes from map-style Dataset overhead vs streaming WDS:
DistributedSampler + random index access into Arrow is less cache-friendly than
WDS sequential streaming. However, the **linear scaling** means the total
throughput at scale matches or exceeds WDS on this machine.

## OOM Regression: HF Image Auto-Decode

After the Arrow conversion, training with 2 GPUs + 30 GB host RAM was
OOM-killed after ~4K batches. Root cause:

`Dataset.from_generator()` infers column feature types from the first samples.
If it infers `jpg` as HuggingFace's `Image` feature type, then every `ds[idx]`
call **automatically decodes the JPEG into a full PIL image** inside the Arrow
reader, before `__getitem__` runs. With 24 worker processes (12 × 2 GPUs) each
holding decoded images (~1–4 MB each) plus mmap pages faulted in by random
DistributedSampler access across all 128 shards, RSS grows until OOM.

**Fix applied** in `build_cc3m_hfd()` and `build_cc12m_hfd()`:

```python
ds = hf_datasets.load_from_disk(arrow_dir)
if isinstance(ds.features.get("jpg"), hf_datasets.Image):
    ds = ds.cast_column("jpg", hf_datasets.Value("binary"))
```

This ensures `row["jpg"]` is always raw bytes — decoding happens in
`__getitem__` only for the current sample, same behaviour as WDS.

**Recommendation:** Use `num_workers ≤ 6` per GPU on this machine as an
additional safeguard. Higher worker counts increase the mmap working set from
random Arrow page faults.

## File Locations

| Purpose | Path |
|---------|------|
| Conversion script | `scripts/convert_wds_to_hf.py` |
| CC3M Arrow loader | `src/datasets/cc3m_hfd.py` |
| CC12M Arrow loader | `src/datasets/cc12m_hfd.py` |
| Combined loader | `src/datasets/combined_hfd.py` |
| CC3M Arrow data | `$SCRATCH/cc3m-hf/` (128 shards) |
| CC12M Arrow data | `$SCRATCH/cc12m-hf/` (512 shards) |
| Hydra configs | `configs/dataset/cc3m_hfd.yaml`, `cc12m_hfd.yaml`, `cc3m_12m_hfd.yaml` |
