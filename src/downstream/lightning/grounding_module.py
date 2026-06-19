"""Lightning module for HiVG-style visual grounding with a LoopViT backbone.

Loads a frozen, CLIP-distilled LoopViT (+ text encoder) backbone from a
checkpoint produced by src/lightning/{clip_module,clip_kd_module}.py, patches
it with HiLoRA adapters (staged curriculum, see hilora.py), and trains the
new HiVGLoopViT grounding heads (MACB, multi-level projection, the
vision-language encoder, and the box head) from scratch.
"""
from __future__ import annotations

from typing import Callable, Optional

import lightning as L
import open_clip
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from transformers import CLIPModel

from ...models.text_encoders.looptext import LoopText
from ...models.visual_encoders.loopvit import LoopViT
from ...utils.misc import cosine_lr_lambda, exclude_weight_decay
from ..losses.grounding_loss import grounding_loss
from ..models.hf_clip_visual import HFCLIPVisionBackbone
from ..models.hilora import blocks_for_stage, patch_block_with_lora, set_lora_trainable
from ..models.hivg_loopvit import HiVGLoopViT
from ..models.text_wrappers import HFCLIPTextWrapper, LoopTextWrapper, OpenCLIPTextWrapper, build_text_wrapper
from ..utils.box_utils import acc_at_iou
from ..utils.checkpoint_io import load_submodule_from_lightning_ckpt


def _build_visual(cfg: DictConfig) -> LoopViT:
    m = cfg.model
    loop_schedule = list(m.loop_schedule) if m.get("loop_schedule") is not None else None
    return LoopViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=int(m.get("visual_embed_dim", 768)),
        num_heads=int(m.get("num_heads", 12)),
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        dropout=0.0,
        loop_core_depth=int(m.get("loop_core_depth", 1)),
        max_loop_steps=int(m.get("max_loop_steps", 12)),
        min_loop_steps=int(m.get("min_loop_steps", 1)),
        add_step_embeddings=bool(m.get("add_step_embeddings", False)),
        use_exit_gate=False,
        loop_mode=str(m.get("loop_mode", "global")),
        loop_schedule=loop_schedule,
    )


def _build_text(cfg: DictConfig) -> tuple[OpenCLIPTextWrapper | LoopTextWrapper, torch.nn.Module]:
    """Returns (text_wrapper, raw_text_module) — the latter is what the checkpoint loads into."""
    m = cfg.model
    embed_dim = int(m.get("embed_dim", 512))
    text_encoder_type = str(m.get("text_encoder", "open_clip"))

    if text_encoder_type == "open_clip":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            m.get("text_encoder_name", "ViT-B-16"), pretrained=None
        )
        del clip_model.visual
        return build_text_wrapper("open_clip", clip_model=clip_model, embed_dim=embed_dim), clip_model
    elif text_encoder_type == "looptext":
        looptext = LoopText(
            vocab_size=int(m.get("text_vocab_size", 49408)),
            context_length=int(m.get("text_seq_len", 77)),
            embed_dim=embed_dim,
            width=int(m.get("text_width", 512)),
            num_heads=int(m.get("text_num_heads", 8)),
            mlp_ratio=float(m.get("text_mlp_ratio", 4.0)),
            loop_core_depth=int(m.get("text_loop_core_depth", 1)),
            max_loop_steps=int(m.get("text_max_loop_steps", 12)),
            add_step_embeddings=bool(m.get("text_add_step_embeddings", False)),
        )
        return build_text_wrapper("looptext", looptext=looptext, embed_dim=embed_dim), looptext
    raise ValueError(f"Unknown text_encoder type: {text_encoder_type!r}")


def _build_backbone(
    cfg: DictConfig,
) -> tuple[nn.Module, OpenCLIPTextWrapper | LoopTextWrapper | HFCLIPTextWrapper, nn.Module]:
    """Returns (visual, text_wrapper, raw_text_module).

    cfg.model.backbone_source selects:
      "loopvit" (default)   — our own LoopViT, weights from
                               cfg.model.backbone_checkpoint (a Lightning
                               .ckpt produced by this repo's CLIP-KD training).
      "hf_clip_pretrained"  — a genuine pretrained HuggingFace CLIP model
                               (cfg.model.hf_clip_pretrained_id, e.g.
                               "openai/clip-vit-base-patch16"), for
                               reproducing HiVG's published baseline. Used
                               instead of open_clip specifically because
                               HuggingFace's CLIPAttention exposes separate
                               q_proj/k_proj/v_proj/out_proj nn.Linear
                               modules (matching HiVG's own released code),
                               where open_clip's fused nn.MultiheadAttention
                               can only ever be LoRA-patched on its MLP (see
                               hilora.py). No backbone_checkpoint loading
                               needed — weights come pre-loaded from the Hub.
    """
    m = cfg.model
    backbone_source = str(m.get("backbone_source", "loopvit"))
    if backbone_source == "hf_clip_pretrained":
        if str(m.get("text_encoder", "hf_clip")) != "hf_clip":
            raise ValueError("backbone_source='hf_clip_pretrained' requires text_encoder='hf_clip'")
        embed_dim = int(m.get("embed_dim", 512))
        pretrained_id = str(m.get("hf_clip_pretrained_id", "openai/clip-vit-base-patch16"))
        clip_model = CLIPModel.from_pretrained(pretrained_id)
        clip_model.train()  # from_pretrained() defaults to eval mode, unlike open_clip
        visual = HFCLIPVisionBackbone(clip_model.vision_model)
        del clip_model.vision_model
        text_wrapper = build_text_wrapper("hf_clip", hf_clip_model=clip_model, embed_dim=embed_dim)
        return visual, text_wrapper, clip_model
    elif backbone_source == "loopvit":
        visual = _build_visual(cfg)
        text_wrapper, raw_text_module = _build_text(cfg)
        return visual, text_wrapper, raw_text_module
    raise ValueError(f"Unknown backbone_source: {backbone_source!r}")


def _text_lora_blocks(raw_text_module: nn.Module, text_encoder_type: str) -> list:
    """Every distinct transformer block in the text encoder, for a flat
    (non-staged) LoRA adapter. Per the HiVG paper: "It is imperative to
    employ HiLoRA for the text encoder with only one layer group as well, in
    order to mitigate the risk of catastrophic forgetting" — one flat group
    covering every layer (unlike vision's 3 cumulative stages), gated on
    hi_lora_stage >= 1 (off during the no-LoRA warmup phase, on for the rest).
    """
    if text_encoder_type == "open_clip":
        return list(raw_text_module.transformer.resblocks)
    elif text_encoder_type == "looptext":
        return list(raw_text_module.blocks)
    elif text_encoder_type == "hf_clip":
        return list(raw_text_module.text_model.encoder.layers)
    raise ValueError(f"Unknown text_encoder type: {text_encoder_type!r}")


class GroundingModule(L.LightningModule):
    """Trains HiVGLoopViT grounding heads on top of a frozen LoopViT/LoopText backbone.

    Args:
        cfg: Full Hydra config. Reads cfg.model, cfg.training, cfg.loss.
        tokenizer: Stored for reference; the data module owns tokenization.
    """

    def __init__(self, cfg: DictConfig, tokenizer: Callable) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "cfg"])
        self.cfg = cfg
        self.tokenizer = tokenizer

        visual, text_wrapper, self._raw_text_module = _build_backbone(cfg)
        self._backbone_source = str(cfg.model.get("backbone_source", "loopvit"))

        m = cfg.model
        self.model = HiVGLoopViT(
            visual=visual,
            text_wrapper=text_wrapper,
            visual_dim=int(m.get("visual_embed_dim", 768)),
            embed_dim=int(m.get("embed_dim", 512)),
            text_seq_len=int(m.get("text_seq_len", 77)),
            vl_hidden_dim=int(m.get("vl_hidden_dim", 512)),
            vl_nheads=int(m.get("vl_nheads", 8)),
            vl_enc_layers=int(m.get("vl_enc_layers", 6)),
            vl_dropout=float(m.get("vl_dropout", 0.1)),
            num_patches=int(m.get("num_patches", 196)),
        )

        self._hilora_rank = int(m.get("hilora_rank", 32))
        self._hilora_alpha = float(m.get("hilora_alpha", 16.0))
        # Fixed for the whole run (one Trainer.fit call = one HiLoRA stage),
        # matching HiVG's --hi_lora_stage: 0 = warmup (backbone fully frozen,
        # no LoRA anywhere, only new heads train), 1/2/3 = cumulative vision
        # HiLoRA + flat text HiLoRA. See scripts/train_grounding_staged.py
        # for running the full 0->1->2->3 curriculum as separate phases.
        self._hi_lora_stage = int(m.get("hi_lora_stage", 3))
        self._stage_blocks: dict[int, list] = {}
        self._text_encoder_type = str(m.get("text_encoder", "open_clip"))
        self._text_lora_enabled = bool(m.get("text_lora", True))
        self._text_blocks: list = []
        self._setup_done = False

    # ------------------------------------------------------------------
    # Setup: load frozen backbone, patch HiLoRA
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        # Idempotent: scripts/train_grounding_staged.py calls setup() manually
        # (to patch LoRA + transplant the previous phase's weights) before
        # handing the module to a fresh Trainer, whose .fit() calls setup()
        # again internally — the second call must be a harmless no-op so it
        # doesn't clobber the manually-loaded weights with backbone_checkpoint.
        if self._setup_done:
            return

        if self._backbone_source == "hf_clip_pretrained":
            pretrained_id = self.cfg.model.get("hf_clip_pretrained_id", "openai/clip-vit-base-patch16")
            print(f"[Backbone] using pretrained HuggingFace CLIP weights ({pretrained_id!r}) — "
                  "already loaded at construction time")
        else:
            backbone_ckpt = self.cfg.model.get("backbone_checkpoint")
            if backbone_ckpt:
                print(f"[Backbone] loading visual + text weights from: {backbone_ckpt}")
                load_submodule_from_lightning_ckpt(backbone_ckpt, self.model.visual, "visual")
                load_submodule_from_lightning_ckpt(backbone_ckpt, self._raw_text_module, "text_model")
            else:
                print("[Backbone] WARNING: no backbone_checkpoint set — random weights")

        # Freeze the entire backbone; only HiLoRA adapters + new grounding heads train.
        for p in self.model.visual.parameters():
            p.requires_grad_(False)
        for p in self.model.text.parameters():
            p.requires_grad_(False)

        if self._hi_lora_stage == 0:
            print("[HiLoRA] stage 0 (warmup): backbone fully frozen, no LoRA — training new heads only")
        else:
            for s in range(1, self._hi_lora_stage + 1):
                self._stage_blocks[s] = blocks_for_stage(self.model.visual, s)
            active_blocks = self._stage_blocks[self._hi_lora_stage]
            for block in active_blocks:
                patch_block_with_lora(block, rank=self._hilora_rank, alpha=self._hilora_alpha)
            set_lora_trainable(active_blocks, True)
            print(f"[HiLoRA] vision stage {self._hi_lora_stage}: "
                  f"{len(active_blocks)} block(s) LoRA-adapted & trainable")

            # Text encoder: a single, flat (non-staged) LoRA across every
            # layer, active for every stage >= 1 (see _text_lora_blocks).
            if self._text_lora_enabled:
                self._text_blocks = _text_lora_blocks(self._raw_text_module, self._text_encoder_type)
                for block in self._text_blocks:
                    patch_block_with_lora(block, rank=self._hilora_rank, alpha=self._hilora_alpha)
                set_lora_trainable(self._text_blocks, True)
                print(f"[HiLoRA] text encoder: {len(self._text_blocks)} block(s) LoRA-adapted & trainable")

        self._setup_done = True

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch):
        images, token_ids, gt_box, obj_mask, _img_files, _phrases = batch
        out = self.model(images, token_ids)
        losses = grounding_loss(
            pred_box=out.pred_box,
            gt_box=gt_box,
            logits_per_text=out.logits_per_text,
            visu_sim=out.visu_sim,
            obj_mask=obj_mask,
            seg_mask=out.seg_mask,
            lambda_l1=self.cfg.loss.get("lambda_l1", 2.0),
            lambda_giou=self.cfg.loss.get("lambda_giou", 2.0),
            lambda_focal=self.cfg.loss.get("lambda_focal", 20.0),
            lambda_dice=self.cfg.loss.get("lambda_dice", 2.0),
            use_contrastive=self.cfg.loss.get("use_contrastive", True),
            use_rtcc=self.cfg.loss.get("use_rtcc", True),
            use_mask_loss=self.cfg.loss.get("use_mask_loss", True),
        )
        total = sum(losses.values())
        return total, losses, out.pred_box, gt_box

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        total, losses, _, _ = self._shared_step(batch)
        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in losses.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=False)
        self.log("hilora_stage", float(self._hi_lora_stage), on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        total, losses, pred_box, gt_box = self._shared_step(batch)
        acc = acc_at_iou(pred_box, gt_box, threshold=0.5)
        self.log("val_loss", total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc@0.5", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in losses.items():
            self.log(f"val_{name}", value, on_epoch=True, sync_dist=True)
        return total

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """dataloader_idx indexes cfg.dataset.test_splits (see
        GroundingDataModule.test_dataloader) — e.g. 0="testA", 1="testB" for
        unc/unc+, 0="test" for gref_umd. Metrics are suffixed with the split
        name (add_dataloader_idx=False) instead of Lightning's default
        "/dataloader_idx_N" so e.g. "test_acc@0.5_testA" is directly readable.
        """
        total, losses, pred_box, gt_box = self._shared_step(batch)
        acc = acc_at_iou(pred_box, gt_box, threshold=0.5)
        test_splits = list(self.cfg.dataset.get("test_splits", ["test"]))
        split_name = test_splits[dataloader_idx] if dataloader_idx < len(test_splits) else f"split{dataloader_idx}"
        self.log(f"test_loss_{split_name}", total, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        self.log(f"test_acc@0.5_{split_name}", acc, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        for name, value in losses.items():
            self.log(f"test_{name}_{split_name}", value, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        return total

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        no_wd, wd_params = exclude_weight_decay(trainable)

        optimizer = torch.optim.AdamW(
            [
                {"params": no_wd, "weight_decay": 0.0},
                {"params": wd_params, "weight_decay": self.cfg.training.weight_decay},
            ],
            lr=self.cfg.training.lr,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
            eps=self.cfg.training.eps,
        )

        total_steps = self.trainer.estimated_stepping_batches
        lr_lambda = cosine_lr_lambda(
            warmup_steps=self.cfg.training.warmup_steps,
            total_steps=total_steps,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
