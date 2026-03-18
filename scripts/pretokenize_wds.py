"""Pre-tokenize WebDataset shards: replace .txt captions with .bin token files.

Each .bin file stores 77 × int32 token IDs as raw little-endian bytes
(77 × 4 = 308 bytes/sample, no numpy header).  DALI reads these with
fn.readers.webdataset(ext=["jpg", "bin"]) and reinterprets them zero-copy via
fn.reinterpret(dtype=INT32, shape=[77]) — eliminating all Python/GIL overhead
from the text branch of the pipeline.

CLIP vocab size is 49408, which fits comfortably in int32.
DALILoader.__next__ casts the output to int64 (torch.long) for encode_text.

Storage overhead vs .txt captions:
  CC3M  (~2.9M samples) : +0.9 GB
  CC12M (~11M samples)  : +3.4 GB
  Combined              : +4.3 GB

Preprocessing time (estimate): ~5 min on 32 CPU cores.

Usage
-----
  # CC3M
  python scripts/pretokenize_wds.py \\
      --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc3m-wds/snapshots/<hash>/cc3m-train-{0000..0575}.tar" \\
      --output-dir "${SCRATCH}/.cache/hub/pretokenized/cc3m/"

  # CC12M
  python scripts/pretokenize_wds.py \\
      --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc12m-wds/snapshots/<hash>/cc12m-train-{0000..1101}.tar" \\
      --output-dir "${SCRATCH}/.cache/hub/pretokenized/cc12m/"

  # Combined (both at once, different output dirs via separate invocations)
  # Control parallelism
  python scripts/pretokenize_wds.py --pattern "..." --output-dir "..." --workers 32

  # Verify a shard without writing
  python scripts/pretokenize_wds.py --pattern "..." --output-dir "..." --dry-run
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path


# ---------------------------------------------------------------------------
# Shard processing worker
# ---------------------------------------------------------------------------


def _process_shard(args: tuple) -> tuple[str, bool, str]:
    """Worker function: tokenize one tar shard and write a new tar with .bin.

    Returns (tar_path, ok, message).
    """
    tar_path, output_dir, tokenizer_name, context_length, dry_run, force = args

    out_path = os.path.join(output_dir, os.path.basename(tar_path))
    if not force and os.path.exists(out_path):
        return tar_path, True, "skipped (already exists)"

    try:
        import numpy as np
        import open_clip

        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        # Collect all samples: key → {ext: bytes}
        samples: dict[str, dict[str, bytes]] = {}

        with tarfile.open(tar_path, "r") as src_tar:
            for member in src_tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                dot = name.rfind(".")
                if dot == -1:
                    continue
                key, ext = name[:dot], name[dot + 1 :]
                if key not in samples:
                    samples[key] = {}
                file_obj = src_tar.extractfile(member)
                if file_obj is not None:
                    samples[key][ext] = file_obj.read()

        if dry_run:
            # Just count and return without writing
            n_valid = sum(1 for s in samples.values() if "jpg" in s and "txt" in s)
            return (
                tar_path,
                True,
                f"dry-run: {len(samples)} entries, {n_valid} valid jpg+txt pairs",
            )

        # Batch-tokenize all captions for this shard
        keys_with_both = [k for k, s in samples.items() if "jpg" in s and "txt" in s]
        captions = [
            samples[k]["txt"].rstrip(b"\x00").decode("utf-8", errors="replace").strip()
            for k in keys_with_both
        ]

        # Tokenize in one batch: [N, context_length] int64 tensor
        if captions:
            token_tensor = tokenizer(captions)  # returns torch.Tensor [N, 77]
            token_array = token_tensor.numpy().astype(np.int64)  # ensure int64
        else:
            token_array = np.empty((0, context_length), dtype=np.int64)

        # Map key → token row (int32 LE: 77 × 4 = 308 bytes, fits CLIP vocab ≤ 49408)
        key_to_tokens: dict[str, bytes] = {
            k: token_array[i].astype("<i4").tobytes()  # little-endian int32
            for i, k in enumerate(keys_with_both)
        }

        # Write new tar: .jpg unchanged, .txt replaced with .bin
        buf = io.BytesIO()
        n_written = 0
        with tarfile.open(fileobj=buf, mode="w") as dst_tar:
            for key in sorted(samples.keys()):
                sample = samples[key]
                if "jpg" not in sample or key not in key_to_tokens:
                    continue  # skip incomplete samples

                # Write JPEG
                jpg_data = sample["jpg"]
                info = tarfile.TarInfo(name=f"{key}.jpg")
                info.size = len(jpg_data)
                dst_tar.addfile(info, io.BytesIO(jpg_data))

                # Write binary tokens
                bin_data = key_to_tokens[key]
                info = tarfile.TarInfo(name=f"{key}.bin")
                info.size = len(bin_data)
                dst_tar.addfile(info, io.BytesIO(bin_data))

                n_written += 1

        # Atomic write: write to .tmp then rename
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())
        os.replace(tmp_path, out_path)

        return tar_path, True, f"created ({n_written} samples)"

    except Exception as e:
        # Clean up partial output
        tmp_path = out_path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return tar_path, False, str(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_pattern(pattern: str) -> list[str]:
    try:
        import braceexpand

        return sorted(braceexpand.braceexpand(pattern))
    except ImportError:
        import glob

        return sorted(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize WebDataset .txt captions to .bin int64 token files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        required=True,
        metavar="BRACE_PATTERN",
        help="Brace-expansion shard pattern (repeatable for multiple datasets)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write pre-tokenized shards (created if missing)",
    )
    parser.add_argument(
        "--tokenizer",
        default="ViT-B-16",
        help="open_clip model name to derive tokenizer from (default: ViT-B-32). "
        "All standard CLIP models use the same SimpleTokenizer, so the model "
        "name only affects SigLIP variants.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=77,
        help="Token sequence length (default: 77, standard for all CLIP models)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel worker processes (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output shards",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse shards and count samples without writing output",
    )
    args = parser.parse_args()

    # Expand shard patterns
    all_tars: list[str] = []
    for pattern in args.patterns:
        tars = _expand_pattern(pattern)
        if not tars:
            print(f"WARNING: no files matched: {pattern}", file=sys.stderr)
        all_tars.extend(tars)

    if not all_tars:
        print("No tar files found — check your patterns.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Shards to process : {len(all_tars)}")
    print(f"Output directory  : {args.output_dir}")
    print(f"Tokenizer         : {args.tokenizer}  context_length={args.context_length}")
    print(f"Workers           : {args.workers}")
    if args.dry_run:
        print("DRY RUN — no files will be written")
    print()

    work = [
        (
            tar,
            args.output_dir,
            args.tokenizer,
            args.context_length,
            args.dry_run,
            args.force,
        )
        for tar in all_tars
    ]

    n_ok = n_skip = n_fail = 0
    with Pool(processes=args.workers) as pool:
        for i, (tar, ok, msg) in enumerate(
            pool.imap_unordered(_process_shard, work), 1
        ):
            name = os.path.basename(tar)
            if not ok:
                n_fail += 1
                print(f"  [{i:>{len(str(len(work)))}}/{len(work)}] FAIL  {name}: {msg}")
            elif msg.startswith("skipped"):
                n_skip += 1
            else:
                n_ok += 1
                if i % max(1, len(work) // 20) == 0 or i == len(work):
                    print(
                        f"  [{i:>{len(str(len(work)))}}/{len(work)}] done  {name}  — {msg}"
                    )

    print(f"\nDone — created: {n_ok}  skipped: {n_skip}  failed: {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
