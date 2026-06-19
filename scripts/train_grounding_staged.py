"""Runs HiVG's literal multi-phase HiLoRA curriculum as a sequence of
separate Lightning Trainer.fit calls: warmup (backbone fully frozen, only the
new grounding heads train) -> stage1 -> stage2 -> stage3 (cumulative vision
HiLoRA + flat text HiLoRA). Each phase is initialized from the previous
phase's best checkpoint — weights only, with a fresh optimizer/LR schedule —
matching --hi_lora_retrain in HiVG/train_and_eval_script/*.sh, not a
Lightning training-state resume (architecture grows new LoRA modules between
phases, so a strict resume isn't applicable).

Usage:
    python scripts/train_grounding_staged.py \
        model=vit_b16_hf_clip_baseline dataset=unc \
        training=hivg_paper_base curriculum=hivg_unc_base

Each phase's hi_lora_stage/epochs/lr/batch_size come from curriculum.phases
(configs/downstream/curriculum/*.yaml); everything else (loss weights, model
architecture, dataset paths, optimizer betas/eps/weight_decay, precision,
workers, ...) is shared across all phases from the selected model/dataset/
loss/training configs.

Eval-only mode: rerun test-set evaluation (e.g. testA/testB for unc/unc+)
against an already-trained checkpoint, skipping training entirely:

    python scripts/train_grounding_staged.py --eval-only \
        --checkpoint /path/to/best-epoch=000-acc=0.7216.ckpt \
        model=vit_b16_hf_clip_baseline dataset=unc training=hivg_paper_base

model.hi_lora_stage (from the model config, default 3) must match the stage
the checkpoint was trained at, since it determines which LoRA modules get
patched onto the architecture before the checkpoint's weights are loaded —
override with model.hi_lora_stage=1/2 for an earlier-stage checkpoint.
--eval-only/--checkpoint are stripped out of argv before Hydra parses it
(Hydra itself only understands key=value overrides), so they can appear
anywhere on the command line alongside the usual key=value overrides.
"""

from __future__ import annotations

import os
import sys

# Allow importing from clip_kd/src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf

from src.datasets.tokenizer import get_tokenizer
from src.downstream.datasets.data_module import GroundingDataModule
from src.downstream.lightning.grounding_module import GroundingModule

torch._dynamo.config.optimize_ddp = False

# --eval-only / --checkpoint aren't key=value Hydra overrides, so they're
# pulled out of sys.argv (see _consume_eval_flags(), called in __main__
# below, before hydra.main's decorator gets to parse the rest) into these
# module-level flags instead.
_EVAL_ONLY = False
_EVAL_CHECKPOINT: str | None = None


def _consume_eval_flags() -> None:
    global _EVAL_ONLY, _EVAL_CHECKPOINT
    remaining = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--eval-only":
            _EVAL_ONLY = True
            i += 1
        elif arg == "--checkpoint":
            if i + 1 >= len(argv):
                raise ValueError("--checkpoint requires a path argument")
            _EVAL_CHECKPOINT = argv[i + 1]
            i += 2
        elif arg.startswith("--checkpoint="):
            _EVAL_CHECKPOINT = arg.split("=", 1)[1]
            i += 1
        else:
            remaining.append(arg)
            i += 1
    sys.argv = [sys.argv[0]] + remaining


class _TrustedCheckpointIO(TorchCheckpointIO):
    """TorchCheckpointIO with weights_only=False — see scripts/train.py for rationale."""

    def load_checkpoint(self, path, map_location=None, **kwargs):
        return torch.load(path, map_location=map_location, weights_only=False)


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


def _load_weights_only(module: GroundingModule, ckpt_path: str) -> None:
    """Transplant a previous phase's weights into this (freshly built,
    possibly larger-LoRA) module. strict=False since later stages add LoRA
    keys the earlier checkpoint doesn't have — those just keep their fresh
    init, matching the paper's "low-stage parameters included in high-stage."
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"[Curriculum] loaded weights from {ckpt_path}")
    if missing:
        print(f"[Curriculum]   {len(missing)} missing key(s) (new LoRA for this stage, kept at fresh init)")
    if unexpected:
        print(f"[Curriculum]   {len(unexpected)} unexpected key(s) (ignored)")


def _run_phase(
    base_cfg: DictConfig,
    phase: DictConfig,
    phase_idx: int,
    tokenizer,
    prev_ckpt: str | None,
    staged_output_dir: str,
    is_last_phase: bool = False,
) -> str:
    cfg = copy.deepcopy(base_cfg)
    cfg.training.lr = phase.lr
    cfg.training.epochs = phase.epochs
    cfg.training.batch_size = phase.batch_size
    cfg.model.hi_lora_stage = phase.hi_lora_stage

    module = GroundingModule(cfg=cfg, tokenizer=tokenizer)
    datamodule = GroundingDataModule(cfg=cfg, tokenizer=tokenizer)

    # Force backbone load + LoRA patching now (idempotent — Trainer.fit calls
    # setup() again internally, see GroundingModule.setup's _setup_done
    # guard), so we can transplant the previous phase's full weights before
    # this phase's Trainer/optimizer is constructed.
    module.setup(stage="fit")
    if prev_ckpt is not None:
        _load_weights_only(module, prev_ckpt)

    output_dir = os.path.join(staged_output_dir, phase.name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

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
        print(f"\n{'=' * 60}\n[Curriculum] phase {phase_idx}: {phase.name} "
              f"(hi_lora_stage={phase.hi_lora_stage}, epochs={phase.epochs}, "
              f"lr={phase.lr}, batch_size={phase.batch_size})\n{'=' * 60}")

    trainer.fit(module, datamodule=datamodule)

    if trainer.is_global_zero:
        _print_model_summary(module)

    best = trainer.checkpoint_callback.best_model_path
    last = os.path.join(ckpt_dir, "last.ckpt")
    result_ckpt = best if (best and os.path.exists(best)) else last
    print(f"[Curriculum] phase {phase.name} done. checkpoint for next phase: {result_ckpt}")

    if is_last_phase:
        # Matches HiVG's hivg_eval.py: test metrics are reported for the
        # best (not last-epoch) checkpoint of the final stage, on every real
        # test split (e.g. testA + testB for unc/unc+) — see
        # cfg.dataset.test_splits / GroundingModule.test_step.
        _load_weights_only(module, result_ckpt)
        trainer.test(module, datamodule=datamodule)

    return result_ckpt


def _run_eval_only(cfg: DictConfig, tokenizer, checkpoint_path: str, output_dir: str) -> None:
    """Loads checkpoint_path's weights into a freshly-built module (whose
    architecture comes from cfg.model.hi_lora_stage — the caller is
    responsible for matching it to the stage the checkpoint was trained at)
    and runs trainer.test() only, no training.
    """
    module = GroundingModule(cfg=cfg, tokenizer=tokenizer)
    datamodule = GroundingDataModule(cfg=cfg, tokenizer=tokenizer)

    module.setup(stage="test")
    _load_weights_only(module, checkpoint_path)

    log_dir = os.path.join(output_dir, "logs")
    logger = CSVLogger(save_dir=log_dir)
    trainer_kwargs = dict(
        precision=cfg.training.precision,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        plugins=[_TrustedCheckpointIO()],
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    )
    trainer_cfg = cfg.get("trainer", OmegaConf.create({}))
    trainer_kwargs.update(OmegaConf.to_container(trainer_cfg, resolve=True))
    trainer = L.Trainer(**trainer_kwargs)

    if trainer.is_global_zero:
        print(f"\n{'=' * 60}\n[Eval-only] hi_lora_stage={cfg.model.hi_lora_stage}, "
              f"checkpoint={checkpoint_path}\n{'=' * 60}")

    trainer.test(module, datamodule=datamodule)


@hydra.main(version_base=None, config_path="../configs/downstream", config_name="grounding_staged")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.training.seed, workers=True)
    tokenizer = get_tokenizer(cfg.model.get("text_encoder_name", "ViT-B-16"))

    output_dir = HydraConfig.get().runtime.output_dir
    _setup_stdout_capture(output_dir)
    print(OmegaConf.to_yaml(cfg))

    if _EVAL_ONLY:
        if not _EVAL_CHECKPOINT:
            raise ValueError("--eval-only requires --checkpoint <path>")
        _run_eval_only(cfg, tokenizer, _EVAL_CHECKPOINT, output_dir)
        return

    phases = cfg.curriculum.phases
    prev_ckpt = cfg.training.get("resume_ckpt") or None
    for i, phase in enumerate(phases):
        prev_ckpt = _run_phase(
            cfg, phase, i, tokenizer, prev_ckpt, output_dir,
            is_last_phase=(i == len(phases) - 1),
        )

    print(f"\n[Curriculum] all phases complete. final checkpoint: {prev_ckpt}")


if __name__ == "__main__":
    _consume_eval_flags()
    main()
