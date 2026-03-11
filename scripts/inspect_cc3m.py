"""Quick sanity-check: instantiate CC3M WebDataset and inspect a few samples.

Usage:
    python scripts/inspect_cc3m.py \
        --shard_pattern "$SCRATCH/cc3m/cc3m-train-{0000..0575}.tar"
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.models.factory  # registers custom model configs (timm-vit_*) as side effect
import open_clip
from src.datasets.cc3m_wds import build_cc3m_wds, CC3M_TRAIN_SAMPLES
from src.datasets.tokenizer import get_tokenizer

MODEL_NAME = "timm-vit_tiny_patch16_224"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shard_pattern",
    required=True,
    help="Brace-expansion shard pattern, e.g. /data/cc3m-train-{0000..0575}.tar",
)
parser.add_argument(
    "--n_samples", type=int, default=5, help="Number of samples to inspect"
)
args = parser.parse_args()

_, preprocess_train, _ = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = get_tokenizer(MODEL_NAME)

ds = build_cc3m_wds(
    shard_pattern=args.shard_pattern,
    transforms=preprocess_train,
    tokenizer=tokenizer,
    num_samples=CC3M_TRAIN_SAMPLES,
)
print(f"WebDataset built (IterableDataset, num_samples={CC3M_TRAIN_SAMPLES})")

for i, (img, txt) in enumerate(ds):
    print(f"  [{i}] OK — image shape={tuple(img.shape)}, text shape={tuple(txt.shape)}")
    if i + 1 >= args.n_samples:
        break
