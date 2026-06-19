# clip-kd-snn

CLIP-style contrastive vision-language training with knowledge distillation,
adapted from the [CLIP-KD paper](original_clip_kd_paper.pdf), used as a
testbed for three lines of research:

1. **Knowledge distillation** from a ViT-B/16 CLIP teacher to smaller/cheaper
   student image+text encoders, using the loss suite from the paper plus a
   couple of additions (SigReg, activation matching).
2. **Architectural alternatives to the student backbone** — a recurrent
   "loop" transformer (LoopViT/LoopText), and several spiking-neural-network
   (SNN) image encoders (QKFormer, MSViT/MSFormer, PseudoSNN).
3. **Downstream transfer to visual grounding** (RefCOCO/+/g), reusing the
   pretrained backbones above as frozen-ish feature extractors under a
   HiVG-style grounding head with HiLoRA fine-tuning.

Everything is wired together with Hydra configs and PyTorch Lightning
modules, with one model/dataset/loss factory per concern so new
architectures and datasets plug into the same training scripts.

## Contents

- [Pipeline overview](#pipeline-overview)
- [What's implemented](#whats-implemented)
- [Conventions](#conventions)
- [Data pipelines](#data-pipelines)
- [Hydra configuration](#hydra-configuration)
- [Running training](#running-training)
- [Evaluation](#evaluation)
- [Visual grounding (downstream)](#visual-grounding-downstream)
- [Spiking neural network backbones](#spiking-neural-network-backbones)
- [Tests](#tests)

## Pipeline overview

```
                 ┌────────────┐        ┌──────────────────┐
   image/text →  │ DataModule │  →     │ student / teacher │  → CompositeLoss → CLIPModule /
   (CC3M/CC12M)   │ (Lightning)│        │   CLIP wrappers   │     (task + KD terms)  CLIPKDModule
                 └────────────┘        └──────────────────┘                         (Lightning)
```

- **Data**: `src/datasets/factory.py` (`CLIPDataModule`) builds the train/val
  loaders for pretraining/KD; `src/downstream/datasets/data_module.py`
  (`GroundingDataModule`) does the same for grounding.
- **Models**: `src/models/factory.py` builds student/teacher CLIP models from
  a Hydra `model` config — either a standard `open_clip` model, or one of the
  custom backbones registered by name prefix (`LoopViT-`, `QKFormer-`,
  `MSViT-`, `PseudoSNN-`). `src/downstream/models/` builds the grounding head
  on top of a pretrained backbone.
- **Losses**: `src/losses/factory.py` (`build_loss`) assembles a
  `CompositeLoss` from whichever `alpha_*` weights are non-zero in the Hydra
  `loss` config — task CLIP loss is always on, every KD term is opt-in.
- **Training loop**: `src/lightning/clip_module.py` (no-KD baseline) and
  `src/lightning/clip_kd_module.py` (KD: holds both student and frozen
  teacher, runs the composite loss, handles projections for dimension
  mismatches) are the two `LightningModule`s used for pretraining. Spiking
  backbones and grounding each have their own `LightningModule` (see below).

## What's implemented

**CLIP-KD core**
- Standard CLIP InfoNCE (task loss) and the paper's distillation losses:
  CRD/CKD (`losses/crd.py`), FD / Masked-FD (`losses/fd.py`, `losses/mfd.py`),
  GD — gradient distillation (`losses/gd.py`), ICL — interactive contrastive
  learning (`losses/icl.py`), AFD — attention/feature-fusion distillation
  (`losses/afd.py`). All combinable via `configs/loss/unified.yaml`.
- Two extra losses outside the paper: activation matching between student
  loop-iterations and teacher blocks (`losses/activation_matching.py`), and
  SigReg — sketched isotropic Gaussian covariance regularization, ported from
  the vendored `sigreg/` reference repo (`losses/sigreg.py`).
- Student backbones: `open_clip` ViT/ResNet/Swin/MobileViT (anything
  `open_clip` or our `model_configs/` timm configs support), plus the
  LoopViT/LoopText and SNN backbones below.
- Teacher: any `open_clip` model, loaded from a Lightning checkpoint or
  `open_clip` pretrained weights.

**LoopViT / LoopText**
- `src/models/visual_encoders/loopvit.py` / `src/models/text_encoders/looptext.py`:
  a shared transformer "core" (1+ blocks) applied recurrently for
  `max_loop_steps` iterations instead of stacking unique blocks per depth —
  same parameter budget as a 1-block transformer, ViT-B/16-equivalent compute
  when `max_loop_steps=12`. Supports a global (whole-encoder) or per-block
  loop mode, optional per-step embeddings, and an optional learned
  early-exit gate (`use_exit_gate`) with an entropy/step-count regularizer
  computed in `_compute_gate_regularizers`.
- Registered in `models/factory.py` under the `LoopViT-*` / model configs
  `configs/model/loopvit_vitb*.yaml` (plain, wide, no-loop control,
  per-block variants) and `loopvit_looptext_vitb.yaml` (both encoders
  recurrent).
- Used both as a standalone baseline (`configs/experiment/baseline_loopvit_looptext_vitb.yaml`)
  and as the backbone under KD and under the grounding head.

**Spiking neural network backbones** — see
[Spiking neural network backbones](#spiking-neural-network-backbones) for detail.
- QKFormer (`models/qkformer_clip.py`, vendored reference in `QKFormer/`).
- MSViT / MSFormer (`models/msvit_clip.py`, vendored reference in `MSViT/`).
- PseudoSNN — SEWResNet with a custom differentiable "pseudo-spiking" neuron
  that avoids the `torch.compile` graph breaks spikingjelly's stateful
  neurons cause (`models/sew_resnet_clip.py`,
  `models/visual_encoders/{pseudo_neuron,psn_node,lif_node,sew_resnet_pseudo}.py`,
  vendored reference in `PseudoSNN/`). See
  `notes/lif_node_compiler_journey.md` for why the custom LIF node exists.

**Visual grounding (HiVG-style)**
- `src/downstream/`: a HiVG-style grounding head (visual-language transformer
  + MLP box regression + HiLoRA-adapted backbone) on top of a pretrained
  LoopViT or `open_clip` ViT-B/16 backbone, trained/evaluated on
  RefCOCO/RefCOCO+/RefCOCOg. Ported from the vendored `HiVG/` reference repo
  (architecture, HiLoRA staging, augmentations) onto our model/data/training
  stack. See [Visual grounding](#visual-grounding-downstream).

## Conventions

- **Config-driven, not flag-driven.** Every script is a thin `@hydra.main`
  wrapper; almost no argparse flags exist outside one-off data-prep scripts.
  New architectures/datasets/losses get a new Hydra config group entry, not a
  new CLI flag.
- **One factory per concern.** `models/factory.py`, `losses/factory.py`,
  `datasets/factory.py` each take a Hydra (sub)config and return a built
  object; this is the only place that switches on `cfg.model.name` /
  `cfg.dataset.type` / `alpha_*` keys. Add a new backbone/dataset/loss by
  extending the relevant factory + adding a config, not by changing the
  training scripts.
- **Name-prefix model registry.** Custom (non-`open_clip`) visual backbones
  are dispatched by string prefix on `model.name`: `LoopViT-`, `QKFormer-`,
  `MSViT-`, `PseudoSNN-`. Each prefix maps to a fixed CLIP embedding dim in a
  `_*_EMBED_DIMS` dict in `models/factory.py`.
  `src/downstream/models/hivg_loopvit.py` follows the same idea for the
  grounding backbone choice (LoopViT vs. plain HF/open_clip ViT).
  `alpha > 0` switches a loss on in `losses/factory.py`; `alpha == 0`
  (the default for every KD term) means "not part of this run" — there is no
  separate boolean enable flag.
- **Hydra config layering**: `configs/experiment/*.yaml` is the unit you
  actually pass on the CLI (`experiment=...`); it overrides `model` /
  `dataset` / `loss` group defaults from `configs/config.yaml` and adds
  experiment-specific paths/hyperparameters. Don't edit the base
  `configs/model/*.yaml` etc. for a one-off run — override on the CLI or add
  an experiment file.
- **Checkpoints carry architecture-defining fields, not just weights** — e.g.
  a grounding model config's `hi_lora_stage` must match the stage a
  checkpoint was trained at, because it determines which LoRA modules get
  patched onto the architecture *before* weights are loaded. Read the
  comments at the top of `scripts/train_grounding_staged.py` before resuming
  from a checkpoint at a different stage.
- **`torch.compile` is on by default** for most backbones (`compile: true` in
  model configs) with `torch._dynamo.config.optimize_ddp = False` set in
  every training script — DDP + compile + custom backbones is fragile;
  see `notes/lif_node_compiler_journey.md` for the specific spikingjelly
  incompatibilities this works around.
- **Vendored reference repos are read-only references**, not dependencies we
  import from at runtime — we port the relevant logic into `src/` with
  comments pointing back at the original file/line, rather than importing
  `HiVG`/`PseudoSNN`/`sigreg` directly. Keep it that way: don't add
  `sys.path` hacks into the vendored trees from `src/`.

## Data pipelines

### Pretraining/KD: CC3M / CC12M

Six interchangeable loader backends for the same logical dataset, selected
via `dataset.type` in a `configs/dataset/*.yaml` (see `datasets/factory.py`
`CLIPDataModule.setup` for the full dispatch):

| `dataset.type` | Backing format | Notes |
|---|---|---|
| `cc3m` / `cc12m` / `combined` | Local CSV/TSV + image files on disk | Simplest, slowest path; needs `train_root` + `train_csv`. |
| `cc3m_wds` / `cc12m_wds` / `combined_wds` | `pixparse/*-wds` HuggingFace WebDataset tar shards | Streamed via `webdataset`, sharded across ranks with `split_by_node`. |
| `cc3m_hfd` / `cc12m_hfd` / `combined_hfd` | Arrow shards via `datasets.load_from_disk()` (mmap) | Built to fix Lustre MDS saturation from many concurrent tar `open()`s — see `notes/lustre_wds_arrow.md`. Requires a one-time conversion: `scripts/convert_wds_to_hf.py --dataset {cc3m,cc12m} --hub-cache $SCRATCH/.cache/hub --output-dir <out> --num-shards 128`. ~2x slower per-GPU than WDS but scales linearly with GPU count (WDS is sublinear on Lustre). |
| `cc3m_wds_dali` / `cc12m_wds_dali` / `combined_wds_dali` | Same WDS tar shards, decoded by **DALI** | GPU-side pipeline: `src/datasets/dali_wds.py`. Only the compressed JPEG bytes cross PCIe; nvJPEG decode + RandomResizedCrop + normalize all run on GPU. Captions still tokenize on CPU via `fn.python_function`. |
| `combined_wds_dali_pretok` | WDS tar shards with `.bin` token files instead of `.txt` captions, decoded by DALI | Zero-Python text path: tokens are pre-baked into 308-byte little-endian int32 blobs and read with `fn.reinterpret` (no GIL, no per-batch tokenizer call). Requires `scripts/pretokenize_wds.py` first. |

**DALI loader setup, in order:**
1. Get the WDS tar shards (from `pixparse/cc3m-wds` / `pixparse/cc12m-wds` on
   the HF Hub, or your own).
2. Build `.idx` index files so DALI doesn't rescan every tar on every rank's
   startup (30–60s/process otherwise):
   `python scripts/create_dali_indices.py --pattern "<brace-expansion pattern>"`.
3. *(Optional, for the pretok path)* Pre-tokenize captions:
   `python scripts/pretokenize_wds.py --pattern "..." --output-dir "..."`.
4. Point `configs/dataset/{cc3m,cc12m,combined}_wds_dali{,_pretok}.yaml` at
   the shard pattern (env-var interpolated, e.g. `${oc.env:SCRATCH}/...`).
5. DDP wiring is automatic — each Lightning process builds its own DALI
   pipeline with `device_id=local_rank`, `shard_id=global_rank`,
   `num_shards=world_size`; `CLIPDataModule.train_dataloader()` detects a
   `DALILoader` and returns it directly instead of wrapping it in a
   `torch.utils.data.DataLoader` (DALI's iterator already implements
   `__iter__`/`__next__`/`__len__`).
6. Sanity-check a pipeline without a full training run:
   `python scripts/test_dali_pipeline.py`, or inspect raw samples with
   `python scripts/inspect_cc3m.py --shard_pattern "..."`.

Evaluation datasets (ImageNet + ImageNet-V2/R/Sketch, MS-COCO, Flickr30K) are
configured the same way via dataset-config keys (`imagenet_val_root`,
`imagenet_wds_dir`, `mscoco_root`, ...) and are added to `val_dataloader()`
opportunistically — only datasets with a path set in the active dataset
config are evaluated.

### Visual grounding: RefCOCO / RefCOCO+ / RefCOCOg

`src/downstream/datasets/refcoco.py` reads HiVG's pre-built `.pth` annotation
files (image path + referring expression + bbox per example). Data layout:

```
$SCRATCH/grounding_data/
├── data/                                   # split_root — annotations
│   ├── unc/unc_{train,val,testA,testB}.pth
│   ├── unc+/unc+_{train,val,testA,testB}.pth
│   └── gref_umd/gref_umd_{train,val,test}.pth
└── other/images/mscoco/images/train2014/   # data_root — MSCOCO train2014 JPEGs
```

Setup (annotations are gated behind a manual Google Drive download upstream
— HiVG doesn't let this be scripted end-to-end):
1. Download `ref_data_shuffled.zip` (HiVG's "Text-Box Annotations") by hand
   from the link in `HiVG/README.md`.
2. `python scripts/prepare_grounding_data.py --data_root $SCRATCH/grounding_data --annotations /path/to/ref_data_shuffled.zip`
   — this downloads MSCOCO train2014 itself (direct cocodataset.org URLs, via
   `HiVG/download_mscoco2014.sh`) and extracts/validates the annotation
   archive into the layout above. Pass `--skip-images` if you already have
   train2014 locally.
3. Point `configs/downstream/dataset/{unc,unc_plus,gref_umd}.yaml`'s
   `data_root`/`split_root` at `$SCRATCH/grounding_data` (defaults already
   assume this via `${oc.env:SCRATCH}`).

`GroundingDataModule` (`src/downstream/datasets/data_module.py`) returns a
**list** of test DataLoaders (one per `dataset.test_splits`, e.g.
`["testA", "testB"]` for unc/unc+ vs. `["test"]` for gref_umd) since RefCOCO
test sets are disjoint by design — `GroundingModule.test_step` uses
`dataloader_idx` to attribute metrics back to the right split name.

## Hydra configuration

Layered like this (`configs/config.yaml` defaults, override per-experiment):

```
config.yaml
├── model:    configs/model/*.yaml            (architecture + backbone hyperparams)
├── dataset:  configs/dataset/*.yaml           (which loader, paths, shard patterns)
├── loss:     configs/loss/*.yaml              (alpha_* weights → which KD terms are active)
├── training: configs/training/*.yaml          (optimizer, schedule, precision, batch size)
└── experiment: configs/experiment/*.yaml      (a single CLI override bundle: which model/
                                                 dataset/loss + experiment-specific paths/hparams)
```

Run dir / output base is also config: `exp_name` and `run_base_dir` control
where Hydra writes outputs
(`${run_base_dir}/${exp_name}/${now:%Y-%m-%d_%H-%M-%S}`).

The downstream/grounding tree mirrors this exact pattern one level deeper
under `configs/downstream/` (`configs/downstream/{model,dataset,loss,training,curriculum}/*.yaml`,
selected from `configs/downstream/{grounding,grounding_staged}.yaml`), plus a
`curriculum` group unique to grounding — see below.

Typical invocation pattern (works for both trees): pick an `experiment=` (or
`downstream=`) bundle, then override anything else by dotted key on the CLI:

```bash
python scripts/train.py experiment=kd_vit_b16_to_t16 \
    model.teacher_checkpoint=/path/to/teacher.ckpt \
    dataset.train_root=/data/cc3m/images dataset.train_csv=/data/cc3m/train.tsv \
    +loss.alpha_gd=1e8 \
    trainer.devices=4 trainer.strategy=ddp
```

## Running training

All entry points live in `scripts/` and are plain `hydra.main` wrappers — run
them with `python scripts/<name>.py <hydra overrides>` from the repo root.

| Script | Purpose |
|---|---|
| `scripts/train.py` | Main CLIP / CLIP-KD training loop (`CLIPModule` if no teacher, `CLIPKDModule` if `model.teacher_checkpoint` is set / loss has KD terms active). |
| `scripts/train_imagenet_loopvit.py`, `train_imagenet_qkformer.py`, `train_imagenet_pseudo_snn.py`, `train_imagenet_cls.py` | Standalone ImageNet classification pretraining for each custom backbone (used to sanity-check an architecture before plugging it into CLIP training). |
| `scripts/train_cifar100_pseudo_snn.py` | Small-scale PseudoSNN sanity check on CIFAR-100 (the `data/cifar-100-python` dataset on disk is for this). |
| `scripts/train_grounding.py` | Single-phase grounding fine-tuning (`downstream=grounding`) — see [Visual grounding](#visual-grounding-downstream). |
| `scripts/train_grounding_staged.py` | HiVG's literal multi-phase HiLoRA curriculum (`downstream=grounding_staged`) — see below. |

`scripts_slurm/*.sh` are the SLURM wrappers actually used to launch the
table above on this cluster — read one as a worked example of a full
command line (module loads, env vars, `srun`/`torchrun` invocation) rather
than relying on this README for cluster-specific flags.

Baseline (no KD) example:
```bash
python scripts/train.py experiment=baseline_vit_t16
```

KD example (teacher → student, unified loss):
```bash
python scripts/train.py experiment=kd_vit_b16_to_t16 \
    model.teacher_checkpoint=/path/to/teacher_pretrained_vit_b_16/last.ckpt \
    dataset.train_root=/data/cc3m/images dataset.train_csv=/data/cc3m/train.tsv
```

LoopViT/LoopText baseline:
```bash
python scripts/train.py experiment=baseline_loopvit_looptext_vitb
```

SNN baselines (QKFormer / MSViT / PseudoSNN), all CLIP-only (no KD) by default:
```bash
python scripts/train.py experiment=baseline_qkformer_vitb
python scripts/train.py experiment=baseline_msformer_vitb
python scripts/train.py experiment=baseline_pseudo_snn_vitb
```

Multi-GPU DDP: append `trainer.devices=N trainer.strategy=ddp` to any of the
above (or use `torchrun`/`srun` per the SLURM scripts).

## Evaluation

| Script | Use |
|---|---|
| `scripts/eval.py` | Standalone zero-shot ImageNet + retrieval eval from a checkpoint, against local-file datasets. |
| `scripts/eval_wds.py` | Same, but against the WDS ImageNet variants (v1/v2/R/Sketch) configured in a `dataset=*_wds*` config. |
| `scripts/eval_open_clip.py` | Eval against an off-the-shelf `open_clip` pretrained checkpoint (no Lightning checkpoint needed) — useful as a sanity baseline. |
| `scripts/eval_all.sh` | Batch-runs eval across a set of checkpoints/configs. |

In-training eval (zero-shot ImageNet + retrieval) runs automatically every
`training.zeroshot_frequency` epochs via `src/lightning/eval_mixin.py`,
shared by `CLIPModule`/`CLIPKDModule`.

## Visual grounding (downstream)

Two training entry points share the same model/dataset/loss building blocks
but differ in how HiLoRA staging is driven:

- **`scripts/train_grounding.py`** — one `Trainer.fit()` call at whatever
  `model.hi_lora_stage` is set in the model config (default stage 3 = full
  cumulative HiLoRA). Use this for a single non-curriculum run, e.g. to
  compare against other backbones already trained in this repo at a fixed
  budget.
- **`scripts/train_grounding_staged.py`** — runs HiVG's literal published
  curriculum as 4 sequential `Trainer.fit()` calls (`warmup → stage1 → stage2
  → stage3`), each phase initialized from the previous phase's *best
  checkpoint, weights only* (fresh optimizer/LR schedule per phase — this is
  `--hi_lora_retrain` in HiVG's original scripts, not a Lightning
  training-state resume, because the architecture grows new LoRA modules
  between phases). Per-phase `hi_lora_stage`/`epochs`/`lr`/`batch_size` come
  from `configs/downstream/curriculum/hivg_unc_base.yaml`; everything else
  (loss weights, architecture, dataset, optimizer betas/eps/wd, precision)
  is shared across phases from the selected `model`/`dataset`/`loss`/`training`
  configs. Supports `--eval-only --checkpoint <path>` to just re-run the
  test-set evaluation (testA/testB/...) against an already-trained
  checkpoint — match `model.hi_lora_stage` to the stage the checkpoint was
  trained at.

Backbone choice for grounding (`configs/downstream/model/*.yaml`):
- `bvit_d1` / `bvit_d3` — LoopViT-ViT-B-16 backbones (depth=1 global loop,
  depth=3 per-block loop respectively), loaded from a `backbone_checkpoint`
  produced by the matching `configs/model/loopvit_vitb*.yaml` pretraining run.
- `bvit_d1_looptext` — same, with LoopText instead of `open_clip`'s text
  transformer.
- `vit_b16_control` — plain ViT-B/16 (depth=12, no loop), as an architecture
  control.
- `vit_b16_hf_clip_baseline` — HuggingFace CLIP ViT-B/16, matching HiVG's
  original published backbone, for comparing our grounding head against
  HiVG's own numbers.

`src/downstream/models/hivg_loopvit.py` documents the architecture wiring in
its module docstring: 4 MACB (multi-modal adapter/cross-attention block)
modules injected at block-execution steps `{1, 4, 8, 12}` (matching HiVG's
0-indexed `adapt_layer=[0,3,7,11]`), a 6-layer `VisionLanguageEncoder`, an MLP
box-regression head, and an auxiliary soft-segmentation head
(`use_mask_loss`, always on, matching HiVG's released scripts).

## Spiking neural network backbones

Three independent image-encoder lines, each with a vendored reference
implementation we ported logic from:

- **QKFormer** (`models/qkformer_clip.py`, `models/visual_encoders/qkformer.py`)
  — spiking transformer with selectable neuron model
  (`lif | sj_lif | plif | nlif | glif | psn | masked_psn | sliding_psn`) and
  backend (`torch | triton | cupy`), configured per-model in
  `configs/model/qkformer_vitb.yaml`'s `snn:` block.
- **MSViT/MSFormer** (`models/msvit_clip.py`, `models/visual_encoders/msformer.py`)
  — similar spiking transformer, narrower neuron-type selection
  (`lif | plif | nlif | glif`).
- **PseudoSNN** (`models/sew_resnet_clip.py`,
  `models/visual_encoders/{pseudo_neuron,psn_node,sew_resnet_pseudo}.py`) —
  SEWResNet (18/34/50) with a custom **PseudoNeuron**: a differentiable
  proxy that replaces actual binary spiking with a learnable, noise-injected
  continuous approximation (`init_T`/`min_T`/`max_T` learned timestep count,
  `noise_type ∈ {uniform, gaussian}`) specifically so the model stays
  `torch.compile`-safe end to end (`compile_snn: false` is fine because the
  ReLU-proxy path doesn't need the SNN-specific compile path that QKFormer/MSViT
  use). `scripts/calibrate_pseudo_snn.py` calibrates the proxy against a real
  spiking forward pass.

Why a custom LIF node exists at all instead of using `spikingjelly` directly:
`notes/lif_node_compiler_journey.md` is a full writeup of the specific
`torch.compile`/Dynamo graph breaks spikingjelly's stateful `MemoryModule`
causes (surrogate-gradient `autograd.Function`, `_memories` dict dispatch,
`isinstance` state-type guards, backend dispatch branching, per-batch
`reset_net` CPU/GPU sync) and how `models/visual_encoders/lif_node.py` avoids
each one. `scripts/benchmark_neurons.py` and `scripts/profile_linear_spike.py`
are the supporting benchmarks; `scripts/debug_lif.py` is a minimal repro
harness.

All three SNN lines plug into the same `CLIPDataModule`/`CompositeLoss`/
`CLIPModule`/`CLIPKDModule` stack as any `open_clip` backbone — there is no
SNN-specific training script for CLIP pretraining (only the standalone
ImageNet pretraining scripts above are SNN-specific, used for architecture
sanity-checks before CLIP training).

## Tests

```bash
pytest tests/
```

`tests/test_models.py` / `test_losses.py` / `test_datasets.py` cover the
factories and individual components in isolation; `test_integration.py`
exercises a couple of small end-to-end forward+loss passes. There is no test
coverage yet for `src/downstream/` (grounding) or the vendored-repo
integrations — treat manual checkpoint-driven eval runs as the verification
path there for now.
