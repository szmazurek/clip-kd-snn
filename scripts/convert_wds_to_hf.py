"""One-time conversion: WebDataset tars → HuggingFace Arrow format.

Reads tar shards from the HF hub cache (datasets--pixparse--{cc3m,cc12m}-wds),
resolves snapshot symlinks to actual blob paths to avoid double Lustre MDS ops,
and writes Arrow-format shards via datasets.save_to_disk().

After conversion, use dataset type cc3m_hfd / cc12m_hfd / combined_hfd in
training configs which call datasets.load_from_disk() — true mmap Arrow
reads, no tar opens during training, Lustre-friendly.

Usage (after cc3m download is complete):
  python scripts/convert_wds_to_hf.py \\
      --dataset cc3m \\
      --hub-cache $SCRATCH/.cache/hub \\
      --output-dir $SCRATCH/cc3m-hf \\
      --num-shards 128 \\
      --num-proc 16

For cc12m (after download):
  python scripts/convert_wds_to_hf.py \\
      --dataset cc12m \\
      --hub-cache $SCRATCH/.cache/hub \\
      --output-dir $SCRATCH/cc12m-hf \\
      --num-shards 512 \\
      --num-proc 32

Shard count guidance:
  CC3M  128 shards ≈ 22 K samples/shard, ~1.9 GB/shard
  CC12M 512 shards ≈ 21 K samples/shard, ~2.0 GB/shard
  Fewer shards = fewer Lustre file opens during training (128 vs 576 for CC3M).
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import datasets as hf_datasets


HF_REPOS = {
    "cc3m": "pixparse/cc3m-wds",
    "cc12m": "pixparse/cc12m-wds",
}

_FEATURES = hf_datasets.Features(
    {
        "jpg": hf_datasets.Value("binary"),
        "txt": hf_datasets.Value("string"),
        "__key__": hf_datasets.Value("string"),
        "__url__": hf_datasets.Value("string"),
    }
)


def find_snapshot_dir(hub_cache_dir: str, repo_id: str) -> str:
    slug = "datasets--" + repo_id.replace("/", "--")
    refs_main = os.path.join(hub_cache_dir, slug, "refs", "main")
    if not os.path.exists(refs_main):
        raise FileNotFoundError(
            f"HF hub refs not found at {refs_main}. "
            "Has the dataset been downloaded with huggingface-cli?"
        )
    with open(refs_main) as f:
        commit = f.read().strip()
    snap = os.path.join(hub_cache_dir, slug, "snapshots", commit)
    if not os.path.isdir(snap):
        raise FileNotFoundError(f"Snapshot dir not found: {snap}")
    return snap


def resolve_blob_paths(snapshot_dir: str, split: str) -> list:
    """Resolve snapshot symlinks to absolute blob paths for a dataset split."""
    info_path = os.path.join(snapshot_dir, "_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"_info.json not found in {snapshot_dir}")
    with open(info_path) as f:
        info = json.load(f)
    if split not in info.get("splits", {}):
        available = list(info.get("splits", {}).keys())
        raise ValueError(f"Split '{split}' not in _info.json. Available: {available}")
    filenames = info["splits"][split]["filenames"]
    resolved = []
    missing = []
    for name in sorted(filenames):
        link = os.path.join(snapshot_dir, name)
        target = Path(link).resolve()
        if not target.exists():
            missing.append(name)
        else:
            resolved.append(str(target))
    if missing:
        print(
            f"[convert] WARNING: {len(missing)}/{len(filenames)} shards missing "
            f"(download still in progress?). First missing: {missing[0]}",
            file=sys.stderr,
        )
    if not resolved:
        raise RuntimeError(f"No shards found for split '{split}' in {snapshot_dir}")
    return resolved


def _iter_wds_tars(shard_paths: list, chunk_idx: int):
    """Iterate samples from a list of WDS tar shards using Python tarfile.

    Yields dicts with keys: jpg (bytes), txt (str), __key__, __url__.
    Skips corrupt tars and corrupt/incomplete samples; returns skip count.
    """
    skipped = 0
    for shard_path in shard_paths:
        try:
            tf = tarfile.open(shard_path, "r")
        except Exception as e:
            skipped += 1
            print(
                f"[chunk {chunk_idx:03d}] WARNING: cannot open shard {shard_path} — {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        # Collect all members, then group by stem key.
        # WDS guarantees members for the same key are contiguous, but
        # grouping is safer and handles any ordering edge case.
        groups: dict[str, dict[str, tarfile.TarInfo]] = {}
        try:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                stem, ext = os.path.splitext(member.name)
                ext = ext.lstrip(".")
                groups.setdefault(stem, {})[ext] = member
        except Exception as e:
            skipped += 1
            print(
                f"[chunk {chunk_idx:03d}] WARNING: truncated shard {shard_path} — {e}",
                file=sys.stderr,
                flush=True,
            )
            tf.close()
            continue

        for key, files in groups.items():
            if "jpg" not in files or "txt" not in files:
                continue
            try:
                jpg_fobj = tf.extractfile(files["jpg"])
                txt_fobj = tf.extractfile(files["txt"])
                if jpg_fobj is None or txt_fobj is None:
                    skipped += 1
                    continue
                jpg_bytes = jpg_fobj.read()
                txt_bytes = txt_fobj.read()
                txt = txt_bytes.decode("utf-8", errors="replace").strip()
                yield {
                    "jpg": jpg_bytes,
                    "txt": txt,
                    "__key__": key,
                    "__url__": shard_path,
                }
            except Exception as e:
                skipped += 1
                if skipped <= 5 or skipped % 1000 == 0:
                    print(
                        f"[chunk {chunk_idx:03d}] WARNING: skipped sample {key}"
                        f" — {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

        tf.close()

    return skipped


def _process_chunk(args: tuple) -> tuple[int, int, str]:
    """Worker: read one chunk of tar shards → Arrow, write to tmp_path.

    Returns (n_skipped, n_written, tmp_path).
    """
    chunk_idx, shard_paths, tmp_path = args

    # _iter_wds_tars is a generator with a return value (skipped count).
    # Wrap it so we can capture that return value while still yielding rows.
    skipped_ref: list[int] = [0]

    def gen_with_count():
        skipped_ref[0] = yield from _iter_wds_tars(shard_paths, chunk_idx)

    chunk_ds = hf_datasets.Dataset.from_generator(  # type: ignore[assignment]
        gen_with_count,
        features=_FEATURES,
    )
    chunk_ds.save_to_disk(tmp_path)
    return skipped_ref[0], len(chunk_ds), tmp_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WebDataset tar shards to HuggingFace Arrow format."
    )
    parser.add_argument(
        "--dataset",
        choices=list(HF_REPOS.keys()),
        required=True,
    )
    parser.add_argument("--hub-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    repo_id = HF_REPOS[args.dataset]
    print(f"[convert] Dataset  : {args.dataset} ({repo_id})")
    print(f"[convert] Hub cache: {args.hub_cache}")
    print(f"[convert] Output   : {args.output_dir}")
    print(f"[convert] Split    : {args.split}")
    print(f"[convert] Shards   : {args.num_shards}")
    print(f"[convert] Workers  : {args.num_proc}")

    snapshot_dir = find_snapshot_dir(args.hub_cache, repo_id)
    print(f"[convert] Snapshot : {snapshot_dir}")

    blob_paths = resolve_blob_paths(snapshot_dir, args.split)
    print(f"[convert] Found {len(blob_paths)} tar shards to read.")

    num_proc = min(args.num_proc, len(blob_paths))
    # Round-robin so each worker gets an interleaved mix of shards.
    chunks = [blob_paths[i::num_proc] for i in range(num_proc)]
    chunks = [c for c in chunks if c]

    tmp_base = args.output_dir + "_tmp_chunks"
    os.makedirs(tmp_base, exist_ok=True)
    work = [
        (i, chunk, os.path.join(tmp_base, f"chunk_{i:04d}"))
        for i, chunk in enumerate(chunks)
    ]

    print(f"[convert] Starting {len(chunks)} parallel workers …")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_proc) as pool:
        results = pool.map(_process_chunk, work)

    total_skipped = sum(r[0] for r in results)
    total_written = sum(r[1] for r in results)
    chunk_paths = [r[2] for r in results]

    print(f"[convert] All workers done in {(time.time()-t0)/60:.1f} min.")
    print(f"[convert] Written: {total_written:,}  Skipped: {total_skipped:,}")

    print("[convert] Concatenating chunks and saving final Arrow dataset …")
    t1 = time.time()
    datasets_list = [hf_datasets.load_from_disk(p) for p in chunk_paths]
    combined = hf_datasets.concatenate_datasets(datasets_list)
    combined.save_to_disk(args.output_dir, num_shards=args.num_shards)
    print(f"[convert] Concatenation + save: {(time.time()-t1)/60:.1f} min.")

    shutil.rmtree(tmp_base, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"[convert] Total time: {elapsed/60:.1f} min. Output: {args.output_dir}")
    print(f"[convert] Skipped samples (corrupt/truncated): {total_skipped:,}")
    print(
        f"[convert] Smoke-test: python -c \"import datasets; ds = datasets.load_from_disk('{args.output_dir}'); print(len(ds), list(ds[0].keys()))\""
    )


if __name__ == "__main__":
    main()
