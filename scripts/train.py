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

    # Callbacks
    callbacks = [
        LogitScaleMonitor(),
        LearningRateMonitor(logging_interval="step"),
        # Best-model checkpoint: tracks val/imagenet/top1, keeps top-3 + last
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="best-epoch={epoch:03d}-top1={val/imagenet/top1:.4f}",
            monitor="val/imagenet/top1",
            mode="max",
            save_top_k=3,
            save_last=True,  # always writes checkpoints/last.ckpt
            auto_insert_metric_name=False,
        ),
        # Periodic checkpoint: saves every N epochs unconditionally
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="periodic-epoch={epoch:03d}",
            every_n_epochs=cfg.training.get("save_every_n_epochs", 5),
            save_top_k=-1,  # keep all periodic checkpoints
            save_last=False,
            auto_insert_metric_name=False,
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    # Logger
    logger = CSVLogger(save_dir="logs/", name="clip_kd")

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
    )
    if cfg.training.get("grad_clip_norm"):
        trainer_kwargs["gradient_clip_val"] = cfg.training.grad_clip_norm

    # Allow trainer overrides from CLI: python train.py trainer.devices=4
    trainer_cfg = cfg.get("trainer", OmegaConf.create({}))
    trainer_kwargs.update(OmegaConf.to_container(trainer_cfg, resolve=True))

    trainer = L.Trainer(**trainer_kwargs)

    # Log full config
    if trainer.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
