"""Training entry point for HiVG-style visual grounding on a LoopViT backbone.

Usage:
    # Depth=1/steps=12 (global) backbone, RefCOCO (unc)
    python scripts/train_grounding.py downstream=grounding \
        downstream.model.backbone_checkpoint=/path/to/loopvit.ckpt

    # Depth=3 per_block backbone, RefCOCO+ (unc+)
    python scripts/train_grounding.py downstream=grounding \
        downstream/model=bvit_d3 downstream/dataset=unc_plus \
        downstream.model.backbone_checkpoint=/path/to/loopvit_d3.ckpt

    # depth=12/steps=1 control backbone, RefCOCOg
    python scripts/train_grounding.py downstream=grounding \
        downstream/model=vit_b16_control downstream/dataset=gref_umd \
        downstream.model.backbone_checkpoint=/path/to/control.ckpt
"""

from __future__ import annotations

import os
import sys

# Allow importing from clip_kd/src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf

from src.datasets.tokenizer import get_tokenizer
from src.downstream.datasets.data_module import GroundingDataModule
from src.downstream.lightning.grounding_module import GroundingModule

import torch

torch._dynamo.config.optimize_ddp = False


class _TrustedCheckpointIO(TorchCheckpointIO):
    """TorchCheckpointIO with weights_only=False — see scripts/train.py for rationale."""

    def load_checkpoint(self, path, map_location=None, **kwargs):
        return torch.load(path, map_location=map_location, weights_only=False)


def _print_model_summary(module: GroundingModule) -> None:
    model = module.model

    def _M(params) -> str:
        return f"{sum(p.numel() for p in params) / 1e6:.1f} M"

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]

    sep = "=" * 48
    print(sep)
    print("  Model: HiVGLoopViT")
    print(f"  Trainable : {_M(trainable)}")
    print(f"  Frozen    : {_M(frozen)}")
    print(f"  Total     : {_M(list(model.parameters()))}")
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
        return False

    def fileno(self):
        return self._stream.fileno()


def _setup_stdout_capture(output_dir: str) -> None:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    log_name = "stdout.log" if rank == 0 else f"stdout_rank{rank}.log"
    log_path = os.path.join(output_dir, log_name)
    sys.stdout = _Tee(sys.__stdout__, log_path)
    sys.stderr = _Tee(sys.__stderr__, log_path)


@hydra.main(version_base=None, config_path="../configs/downstream", config_name="grounding")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.training.seed, workers=True)

    tokenizer = get_tokenizer(cfg.model.get("text_encoder_name", "ViT-B-16"))

    module = GroundingModule(cfg=cfg, tokenizer=tokenizer)
    datamodule = GroundingDataModule(cfg=cfg, tokenizer=tokenizer)

    output_dir = HydraConfig.get().runtime.output_dir
    _setup_stdout_capture(output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-epoch={epoch:03d}-acc={val_acc@0.5:.4f}",
            monitor="val_acc@0.5",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="periodic-epoch={epoch:03d}",
            every_n_epochs=cfg.training.get("save_every_n_epochs", 5),
            save_top_k=-1,
            save_last=False,
            auto_insert_metric_name=False,
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    logger = CSVLogger(save_dir=log_dir)

    trainer_kwargs = dict(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        plugins=[_TrustedCheckpointIO()],
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        check_val_every_n_epoch=cfg.training.get("zeroshot_frequency", 1),
        num_sanity_val_steps=0,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        fast_dev_run=cfg.training.get("fast_dev_run", False),
    )
    if cfg.training.get("grad_clip_norm"):
        trainer_kwargs["gradient_clip_val"] = cfg.training.grad_clip_norm

    trainer_cfg = cfg.get("trainer", OmegaConf.create({}))
    trainer_kwargs.update(OmegaConf.to_container(trainer_cfg, resolve=True))

    trainer = L.Trainer(**trainer_kwargs)

    if trainer.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    resume_ckpt = cfg.training.get("resume_ckpt") or None
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)

    if trainer.is_global_zero:
        _print_model_summary(module)


if __name__ == "__main__":
    main()
