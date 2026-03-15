"""Pre-generate DALI WebDataset index files for all tar shards.

DALI's fn.readers.webdataset can use pre-built .idx files so it doesn't scan
every tar on pipeline startup (which hangs for 30-60 s per process on a
4-GPU run because each rank scans the same shards independently).

Index files are placed next to the tars with the same name, .tar → .idx:
  cc3m-train-0001.tar  →  cc3m-train-0001.idx

Usage:
    # CC3M
    python scripts/create_dali_indices.py \\
        --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc3m-wds/snapshots/<hash>/cc3m-train-{0000..0575}.tar"

    # CC12M
    python scripts/create_dali_indices.py \\
        --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc12m-wds/snapshots/<hash>/cc12m-train-{0000..1101}.tar"

    # Combined (both patterns at once)
    python scripts/create_dali_indices.py \\
        --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc3m-wds/snapshots/<hash>/cc3m-train-{0000..0575}.tar" \\
        --pattern "${SCRATCH}/.cache/hub/datasets--pixparse--cc12m-wds/snapshots/<hash>/cc12m-train-{0000..1101}.tar"

    # Control parallelism (default: all CPUs)
    python scripts/create_dali_indices.py --pattern "..." --workers 16

wds2idx is a command-line tool bundled with DALI.  If it is not on PATH,
set DALI_WDS2IDX to its full path, e.g.:
    export DALI_WDS2IDX=$(python -c "import nvidia.dali, os; print(os.path.join(os.path.dirname(nvidia.dali.__file__), 'wds2idx'))")
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path


def _expand_pattern(pattern: str) -> list[str]:
    try:
        import braceexpand
        return sorted(braceexpand.braceexpand(pattern))
    except ImportError:
        import glob
        return sorted(glob.glob(pattern))


def _find_wds2idx() -> str:
    """Return the path to the wds2idx binary."""
    # 1. Explicit override via env var
    if "DALI_WDS2IDX" in os.environ:
        return os.environ["DALI_WDS2IDX"]

    # 2. On PATH
    if shutil.which("wds2idx"):
        return "wds2idx"

    # 3. Next to nvidia.dali.__file__
    try:
        import nvidia.dali
        candidate = Path(nvidia.dali.__file__).parent / "wds2idx"
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass

    print(
        "ERROR: wds2idx not found.\n"
        "  It ships with DALI; try: pip install nvidia-dali-cudaXXX\n"
        "  Or set DALI_WDS2IDX=/path/to/wds2idx",
        file=sys.stderr,
    )
    sys.exit(1)


def _create_one(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    """Worker: run `wds2idx <tar> <idx>`.  Returns (tar, ok, message)."""
    wds2idx, tar_path, idx_path = args
    if os.path.exists(idx_path):
        return tar_path, True, "skipped (already exists)"
    try:
        result = subprocess.run(
            [wds2idx, tar_path, idx_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        return tar_path, True, "created"
    except subprocess.CalledProcessError as e:
        return tar_path, False, e.stderr.decode().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate DALI WebDataset index files")
    parser.add_argument(
        "--pattern", action="append", dest="patterns", required=True,
        metavar="BRACE_PATTERN",
        help="Brace-expansion shard pattern (repeatable for combined datasets)",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Parallel worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recreate index files even if they already exist",
    )
    args = parser.parse_args()

    wds2idx = _find_wds2idx()
    print(f"Using wds2idx: {wds2idx}")

    all_tars: list[str] = []
    for pattern in args.patterns:
        tars = _expand_pattern(pattern)
        if not tars:
            print(f"WARNING: no files matched pattern: {pattern}", file=sys.stderr)
        all_tars.extend(tars)

    if not all_tars:
        print("No tar files found — check your patterns.", file=sys.stderr)
        sys.exit(1)

    # Build work list: (wds2idx, tar, idx) triples
    work = []
    for tar in all_tars:
        idx = tar.replace(".tar", ".idx")
        if args.force and os.path.exists(idx):
            os.remove(idx)
        work.append((wds2idx, tar, idx))

    print(f"Processing {len(work)} shards with {args.workers} workers...")

    n_ok = n_skip = n_fail = 0
    with Pool(processes=args.workers) as pool:
        for i, (tar, ok, msg) in enumerate(pool.imap_unordered(_create_one, work), 1):
            name = os.path.basename(tar)
            if not ok:
                n_fail += 1
                print(f"  [{i}/{len(work)}] FAIL  {name}: {msg}")
            elif msg.startswith("skipped"):
                n_skip += 1
                if args.workers == 1:   # verbose only in serial mode
                    print(f"  [{i}/{len(work)}] skip  {name}")
            else:
                n_ok += 1
                if i % max(1, len(work) // 20) == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] done  {name}")

    print(f"\nDone — created: {n_ok}  skipped: {n_skip}  failed: {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
