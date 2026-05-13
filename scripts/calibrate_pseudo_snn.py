"""Post-training calibration and evaluation for PseudoSNN SEW-ResNet.

After training with train_imagenet_pseudo_snn.py --no-val, load the Lightning
checkpoint and run the calibration procedure from the original PseudoSNN paper:

  1. BN layers stay in train() mode  →  running stats are recomputed from
     actual spikes rather than from the continuous ReLU proxy used in training.
  2. PseudoNeurons switch to eval() mode  →  real IF spiking neurons fire.
  3. Only the per-neuron `scale` and `bias` parameters are updated  →  weights
     frozen, no T-logit change.
  4. After calibration, evaluate on the full ImageNet validation set.

The calibrated model is saved as a plain state-dict checkpoint so it can be
loaded directly for inference without reconstructing the Lightning wrapper.

Usage:
    python scripts/calibrate_pseudo_snn.py \\
        --checkpoint runs/imagenet_pseudo_snn/last.ckpt \\
        --data-dir /storage/imagenet-full-hf/data \\
        --model sew_resnet18 \\
        --calib-epochs 10 \\
        --calib-lr 1e-3 \\
        --output-dir ./runs/imagenet_pseudo_snn
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.visual_encoders.sew_resnet_pseudo import (
    sew_resnet18,
    sew_resnet34,
    sew_resnet50,
)
from src.models.visual_encoders.pseudo_neuron import PseudoNeuron
from src.lightning.pseudo_snn_imagenet_module import PseudoSNNImageNetModule

torch.set_float32_matmul_precision("high")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_VAL_SAMPLES = 50_000

_MODEL_REGISTRY = {
    "sew_resnet18": sew_resnet18,
    "sew_resnet34": sew_resnet34,
    "sew_resnet50": sew_resnet50,
}


# ---------------------------------------------------------------------------
# Data helpers  (mirrors train_imagenet_pseudo_snn.py)
# ---------------------------------------------------------------------------

def _detect_format(data_dir: str) -> str:
    if glob.glob(os.path.join(data_dir, "train-*.parquet")):
        return "parquet"
    if glob.glob(os.path.join(data_dir, "*.tar")):
        return "wds"
    return "imagefolder"


def build_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class ParquetImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, parquet_files, transform=None, shuffle=False, seed=42):
        import pyarrow.parquet as pq
        self.transform = transform
        self._files = sorted(parquet_files)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._total = sum(pq.read_metadata(f).num_rows for f in self._files)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        import pyarrow.parquet as pq
        import random
        worker_info = torch.utils.data.get_worker_info()
        files = list(self._files)
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(files)
        for fpath in files:
            table = pq.read_table(fpath, columns=["image", "label"])
            rows = table.to_pydict()
            images, labels = rows["image"], rows["label"]
            indices = list(range(len(labels)))
            if self.shuffle:
                rng = random.Random(self.seed + self._epoch + hash(fpath))
                rng.shuffle(indices)
            for i in indices:
                img = Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                yield img, int(labels[i])


def build_loaders(data_dir: str, batch_size: int, workers: int):
    """Return (train_loader, val_loader) for the detected dataset format."""
    fmt = _detect_format(data_dir)
    print(f"[data] Detected format: {fmt}  ({data_dir})")

    train_tf = build_train_transforms()
    val_tf = build_val_transforms()

    if fmt == "parquet":
        train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
        val_files   = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
        train_ds = ParquetImageDataset(train_files, train_tf, shuffle=True)
        val_ds   = ParquetImageDataset(val_files,   val_tf)

    elif fmt == "wds":
        import webdataset as wds

        def decode_cls(cls_raw):
            return int(cls_raw.decode() if isinstance(cls_raw, bytes) else cls_raw)

        train_shards = sorted(glob.glob(os.path.join(data_dir, "train", "*.tar"))) \
                    or sorted(glob.glob(os.path.join(data_dir, "*.tar")))
        val_shards   = sorted(glob.glob(os.path.join(data_dir, "val",   "*.tar")))

        train_ds = (
            wds.WebDataset(train_shards, shardshuffle=True)
            .shuffle(5000).decode("pil").to_tuple("jpg;webp", "cls")
            .map_tuple(train_tf, decode_cls)
        )
        val_ds = (
            wds.WebDataset(val_shards, shardshuffle=False, empty_check=False)
            .decode("pil").to_tuple("jpg;webp", "cls")
            .map_tuple(val_tf, decode_cls)
            .with_epoch(IMAGENET_VAL_SAMPLES)
        )

    else:
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        val_ds   = ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    from torch.utils.data import IterableDataset
    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(not is_iterable),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(workers > 0 and not is_iterable),
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, backbone: torch.nn.Module) -> torch.nn.Module:
    """Load a Lightning checkpoint into the backbone nn.Module.

    PseudoSNNImageNetModule uses save_hyperparameters(ignore=['model']), so
    the backbone must be supplied explicitly at load time.
    """
    lit = PseudoSNNImageNetModule.load_from_checkpoint(
        ckpt_path,
        model=backbone,
        map_location="cpu",
    )
    return lit.model


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    calib_epochs: int,
    calib_lr: float,
) -> None:
    """Fine-tune the entire network (except T-logits) with real spikes.

    Matches the original PseudoSNN calibrate() procedure exactly:
      - model.train()  →  BN layers recompute running stats from spike data
      - PseudoNeurons set to eval()  →  real IF neurons fire
      - logit (T) parameters are frozen  →  timestep counts don't change
      - ALL other parameters (conv weights, BN, scale, bias, fc) get lr=calib_lr

    The original reuses the training optimizer with accumulated momentum.
    Here we build a fresh Adam, which is equivalent but starts without momentum.
    """
    logit_ids = {id(m.logit) for m in model.modules() if isinstance(m, PseudoNeuron)}
    calib_params = [p for p in model.parameters() if id(p) not in logit_ids]

    if not calib_params:
        print("[calib] No trainable parameters found — skipping calibration.")
        return

    optimizer = torch.optim.Adam(calib_params, lr=calib_lr)
    model.to(device)

    for epoch in range(calib_epochs):
        # Hybrid mode: BN in train (updates running stats), PseudoNeurons in eval
        model.train()
        for m in model.modules():
            if isinstance(m, PseudoNeuron):
                m.eval()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"calib {epoch + 1}/{calib_epochs}", leave=True)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(pbar), 1)

        t_vals = [m.get_timesteps().item() for m in model.modules() if isinstance(m, PseudoNeuron)]
        avg_t = sum(t_vals) / len(t_vals) if t_vals else float("nan")
        print(f"  → avg_loss={avg_loss:.4f}  avg_T={avg_t:.1f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    model.to(device)

    correct1 = correct5 = total = 0

    pbar = tqdm(val_loader, desc="eval", leave=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        top5 = logits.topk(5, dim=1).indices
        correct1 += (logits.argmax(dim=1) == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total    += labels.size(0)

        pbar.set_postfix(acc1=f"{100.0 * correct1 / total:.2f}%")

    acc1 = 100.0 * correct1 / total
    acc5 = 100.0 * correct5 / total
    print(f"[eval]  Top-1: {acc1:.2f}%  Top-5: {acc5:.2f}%  ({total} samples)")
    return {"acc1": acc1, "acc5": acc5, "total": total}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate a PseudoSNN Lightning checkpoint on ImageNet"
    )

    # Required
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument("--data-dir",   required=True, help="ImageNet data dir (Parquet/WDS/ImageFolder).")

    # Must match training config
    parser.add_argument("--model",     default="sew_resnet18", choices=list(_MODEL_REGISTRY.keys()))
    parser.add_argument("--connect-f", default="ADD", choices=["ADD", "AND", "IAND"])
    parser.add_argument("--noise-type", default="gaussian", choices=["uniform", "gaussian"])
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--init-T",    type=float, default=8.0)
    parser.add_argument("--min-T",     type=int,   default=1)
    parser.add_argument("--max-T",     type=int,   default=16)

    # Calibration
    parser.add_argument(
        "--calib-epochs", type=int, default=10,
        help="Calibration epochs (full passes over train data with spikes + scale/bias updates).",
    )
    parser.add_argument(
        "--calib-lr", type=float, default=1e-3,
        help="Adam learning rate for scale and bias during calibration.",
    )

    # Data / hardware
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for calibration and evaluation (e.g. 'cuda', 'cuda:1', 'cpu').",
    )

    # Output
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save the calibrated model state dict.  "
             "Defaults to the checkpoint's parent directory.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)

    # ---- Build backbone (architecture must match training) ----
    neuron_kwargs = dict(
        noise_type=args.noise_type,
        noise_prob=args.noise_prob,
        init_T=args.init_T,
        min_T=args.min_T,
        max_T=args.max_T,
    )
    backbone = _MODEL_REGISTRY[args.model](
        num_classes=1000, connect_f=args.connect_f, **neuron_kwargs
    )
    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"[model] {args.model}  ({n_params:.1f} M params)")

    # ---- Load Lightning checkpoint ----
    print(f"[ckpt]  Loading {args.checkpoint}")
    backbone = load_model_from_checkpoint(args.checkpoint, backbone)

    # Log T values before calibration
    t_vals = [m.get_timesteps().item() for m in backbone.modules() if isinstance(m, PseudoNeuron)]
    if t_vals:
        print(f"[ckpt]  Loaded  avg_T={sum(t_vals)/len(t_vals):.1f}  "
              f"min_T={min(t_vals):.1f}  max_T={max(t_vals):.1f}")

    # ---- Data ----
    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size, args.workers)

    # ---- Calibration ----
    if args.calib_epochs > 0:
        print(f"\n[calib] Running {args.calib_epochs} calibration epoch(s) "
              f"(lr={args.calib_lr})...")
        calibrate(backbone, train_loader, device, args.calib_epochs, args.calib_lr)
    else:
        print("[calib] calib-epochs=0, skipping calibration.")
        backbone.to(device)

    # ---- Evaluate ----
    print("\n[eval]  Evaluating on ImageNet validation set...")
    results = evaluate(backbone, val_loader, device)

    # ---- Save calibrated model ----
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(output_dir, f"{ckpt_stem}_calibrated.pt")
    torch.save(
        {
            "model_state_dict": backbone.state_dict(),
            "acc1": results["acc1"],
            "acc5": results["acc5"],
            "args": vars(args),
        },
        out_path,
    )
    print(f"[save]  Calibrated model saved to {out_path}")


if __name__ == "__main__":
    main()
