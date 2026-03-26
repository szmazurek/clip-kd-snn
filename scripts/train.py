"""Main training entry point for CLIP-KD.

Usage:
    # Baseline CLIP training
    python scripts/train.py experiment=baseline_vit_t16

    # KD training (ViT-B/16 → ViT-T/16)
    python scripts/train.py experiment=kd_vit_b16_to_t16 \
        model.teacher_checkpoint=/path/to/teacher.pt \
        dataset.train_root=/path/to/cc3m/images \
        dataset.train_csv=/path/to/cc3m/train.tsv

    # Multi-GPU DDP
    python scripts/train.py trainer.devices=4 trainer.strategy=ddp \
        experiment=kd_vit_b16_to_t16

    # Override individual loss weights
    python scripts/train.py +loss.alpha_gd=1e8 experiment=kd_vit_b16_to_t16
"""

from __future__ import annotations

import sys
import os

# Allow importing from clip_kd/src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.datasets.factory import CLIPDataModule
from src.datasets.tokenizer import get_tokenizer
from src.lightning.callbacks import LogitScaleMonitor
from src.lightning.clip_kd_module import CLIPKDModule
from src.lightning.clip_module import CLIPModule
from src.models.factory import build_student_model


def _print_model_summary(module: L.LightningModule, model_name: str) -> None:
    model = module.student.model
    if hasattr(model, "_orig_mod"):  # unwrap torch.compile
        model = model._orig_mod

    def _M(params) -> str:
        return f"{sum(p.numel() for p in params) / 1e6:.1f} M"

    visual_params = list(model.visual.parameters())
    total_params = list(model.parameters())
    logit_params = [model.logit_scale]
    text_params = [
        p for p in total_params
        if not any(p is q for q in visual_params + logit_params)
    ]

    sep = "=" * 48
    print(sep)
    print(f"  Model: {model_name}")
    print(f"  Visual encoder : {_M(visual_params)}")
    print(f"  Text encoder   : {_M(text_params)}")
    print(f"  Total          : {_M(total_params)}")
    print(sep)


class _Tee:
    """Duplicates writes to both the original stream and a log file."""

    def __init__(self, stream, log_path: str):
        self._stream = stream
        self._fh = open(log_path, "a", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return (
            False  # tqdm uses static (newline-per-update) mode — cleaner in log files
        )

    def fileno(self):
        return self._stream.fileno()


def _setup_stdout_capture(output_dir: str) -> None:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    log_name = "stdout.log" if rank == 0 else f"stdout_rank{rank}.log"
    log_path = os.path.join(output_dir, log_name)
    sys.stdout = _Tee(sys.__stdout__, log_path)
    sys.stderr = _Tee(sys.__stderr__, log_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.training.seed, workers=True)

    # Determine whether this is a KD or baseline run
    is_kd = (
        cfg.model.get("teacher_name") is not None
        and cfg.model.get("teacher_name") != "null"
    )

    # Build model + transforms
    student, preprocess_train, preprocess_val = build_student_model(cfg)
    tokenizer = get_tokenizer(cfg.model.name)

    # Lightning module
    if is_kd:
        module = CLIPKDModule(cfg=cfg, tokenizer=tokenizer)
    else:
        module = CLIPModule(cfg=cfg, tokenizer=tokenizer)

    # DataModule
    datamodule = CLIPDataModule(
        cfg=cfg,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
    )

    # Resolve paths inside this run's Hydra output dir so each run is self-contained
    output_dir = HydraConfig.get().runtime.output_dir
    _setup_stdout_capture(output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")

    # Callbacks
    callbacks = [
        LogitScaleMonitor(),
        LearningRateMonitor(logging_interval="step"),
        # Best-model checkpoint: tracks val_imagenet_top1, keeps top-3 + last
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-epoch={epoch:03d}-top1={val_imagenet_top1:.4f}",
            monitor="val_imagenet_top1",
            mode="max",
            save_top_k=3,
            save_last=True,  # always writes checkpoints/last.ckpt
            auto_insert_metric_name=False,
        ),
        # Periodic checkpoint: saves every N epochs unconditionally
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="periodic-epoch={epoch:03d}",
            every_n_epochs=cfg.training.get("save_every_n_epochs", 5),
            save_top_k=-1,  # keep all periodic checkpoints
            save_last=False,
            auto_insert_metric_name=False,
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    # Logger
    logger = CSVLogger(save_dir=log_dir)

    # Trainer
    trainer_kwargs = dict(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        devices="auto",
        accelerator="auto",
        strategy="auto",
        # Run validation every N epochs (reuses zeroshot_frequency config key)
        check_val_every_n_epoch=cfg.training.get("zeroshot_frequency", 1),
        num_sanity_val_steps=0,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        fast_dev_run=cfg.training.get("fast_dev_run", False),
    )
    if cfg.training.get("grad_clip_norm"):
        trainer_kwargs["gradient_clip_val"] = cfg.training.grad_clip_norm

    # Allow trainer overrides from CLI: python train.py trainer.devices=4
    trainer_cfg = cfg.get("trainer", OmegaConf.create({}))
    trainer_kwargs.update(OmegaConf.to_container(trainer_cfg, resolve=True))

    trainer = L.Trainer(**trainer_kwargs)

    # Log full config
    if trainer.is_global_zero:
        _print_model_summary(module, cfg.model.name)
        print(OmegaConf.to_yaml(cfg))

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
