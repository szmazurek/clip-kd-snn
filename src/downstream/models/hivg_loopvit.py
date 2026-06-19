"""HiVG model with LoopViT visual backbone.

Wires together:
  - LoopViT visual encoder (block-level execution via iter_blocks())
  - Text encoder wrapper (OpenCLIP or LoopText)
  - 4 MACB modules at block-execution slots {1, 4, 8, 12} (matches HiVG's
    adapt_layer=[0,3,7,11], 0-indexed — same four points as extraction)
  - Multi-level visual projection (stash at execs {1, 4, 8, 12} → 512)
  - VisionLanguageEncoder (6-layer grounding transformer)
  - MLP box regression head
  - visu_token_norm/mlp: LayerNorm + QuickGELU FFN applied to the grounding
    encoder's patch outputs before the RTCC dot-product (HiVG's
    clip_last_layer_features path)
  - seg_conv1/2/3: 3 ConvTranspose2d layers (14x14 -> 112x112) producing a
    second, higher-resolution soft-segmentation map from the same raw patch
    outputs, for the use_mask_loss auxiliary loss (always on in HiVG's
    released training scripts, no dataset exception)

Forward returns a GroundingOutput dataclass for clean loss computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import QuickGELUActivation

from .macb import MACB
from .text_wrappers import OpenCLIPTextWrapper, LoopTextWrapper
from .vl_transformer import VisionLanguageEncoder, MLP

# Block-execution indices where MACB injects and features are extracted
_INJECT_EXECS = frozenset({1, 4, 8, 12})
_EXTRACT_EXECS = frozenset({1, 4, 8, 12})
# Mapping injection exec → MACB module index
_INJECT_SLOT = {1: 0, 4: 1, 8: 2, 12: 3}


class _TokenMLP(nn.Module):
    """Matches HiVG's TOKEN_MLP: Linear -> QuickGELU -> Linear, no residual."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


@dataclass
class GroundingOutput:
    pred_box: Tensor            # [B, 4]  sigmoid (cx,cy,w,h) normalized
    logits_per_text: Tensor     # [B, B]  for CLC loss
    visu_sim: Tensor            # [B, N_patches]  for RTCC loss
    seg_mask: Tensor            # [B, 1, 112, 112]  for use_mask_loss
    img_cls_embed: Tensor       # [B, embed_dim]  L2-normalised
    text_eos_embed: Tensor      # [B, embed_dim]  L2-normalised


class HiVGLoopViT(nn.Module):
    """HiVG grounding model using LoopViT as the visual backbone.

    Args:
        visual: LoopViT instance (any depth/step config).
        text_wrapper: OpenCLIPTextWrapper or LoopTextWrapper.
        visual_dim: Dimensionality of LoopViT token outputs (e.g. 768).
        embed_dim: CLIP projection dimension (e.g. 512).
        text_seq_len: Text sequence length (77 for CLIP tokenizer).
        vl_hidden_dim: Grounding encoder hidden dim (512).
        vl_nheads: Grounding encoder attention heads (8).
        vl_enc_layers: Grounding encoder depth (6).
        vl_dropout: Grounding encoder dropout (0.1).
        num_patches: Number of visual patch tokens (196 for 224×224, patch=16).
        logit_scale_init: Initial value for temperature parameter.
    """

    def __init__(
        self,
        visual: nn.Module,
        text_wrapper: OpenCLIPTextWrapper | LoopTextWrapper,
        visual_dim: int = 768,
        embed_dim: int = 512,
        text_seq_len: int = 77,
        vl_hidden_dim: int = 512,
        vl_nheads: int = 8,
        vl_enc_layers: int = 6,
        vl_dropout: float = 0.1,
        num_patches: int = 196,
        logit_scale_init: float = math.log(1 / 0.07),
    ) -> None:
        super().__init__()

        self.visual = visual
        self.text = text_wrapper
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.text_seq_len = text_seq_len

        n_text_layers = text_wrapper.n_layers
        text_hidden_dim = text_wrapper.hidden_dim

        # Four MACB modules, one per injection slot (matches extraction points)
        self.macb = nn.ModuleList([
            MACB(
                visual_dim=visual_dim,
                text_dim=text_hidden_dim,
                n_text_layers=n_text_layers,
                text_seq_len=text_seq_len,
                num_heads=vl_nheads,
            )
            for _ in range(4)
        ])

        # Multi-level visual projection: 4 extractions × visual_dim → embed_dim
        self.ml_visual_projection = nn.Linear(4 * visual_dim, embed_dim)

        # Token-level projections into the grounding encoder space
        self.visu_proj = nn.Linear(embed_dim, vl_hidden_dim)
        self.text_proj = nn.Linear(text_hidden_dim, vl_hidden_dim)

        # CLS token projections (for CLC loss)
        self.visual_cls_proj = nn.Linear(visual_dim, embed_dim)
        self.text_eos_proj = nn.Linear(text_hidden_dim, embed_dim)

        # Learnable [REG] token read by the box head
        self.reg_token = nn.Embedding(1, vl_hidden_dim)

        # Positional embedding for the VL encoder input sequence:
        # [REG](1) + [CLS_vis](1) + patches(num_patches) + text(text_seq_len)
        num_vl_tokens = 1 + 1 + num_patches + text_seq_len
        self.vl_pos_embed = nn.Embedding(num_vl_tokens, vl_hidden_dim)

        # Grounding encoder
        self.vl_encoder = VisionLanguageEncoder(
            d_model=vl_hidden_dim,
            nhead=vl_nheads,
            num_layers=vl_enc_layers,
            dim_feedforward=vl_hidden_dim * 4,
            dropout=vl_dropout,
            normalize_before=True,
        )

        # Pre-RTCC transform on the grounding encoder's patch outputs
        # (HiVG's visu_token_norm / visu_token_mlp, applied before the
        # text-eos dot-product — we previously dotted the raw patch outputs).
        self.visu_token_norm = nn.LayerNorm(vl_hidden_dim, eps=1e-5)
        self.visu_token_mlp = _TokenMLP(vl_hidden_dim, vl_hidden_dim * 6)

        # use_mask_loss auxiliary head: 14x14 -> 28x28 -> 56x56 -> 112x112,
        # a second, higher-resolution soft-segmentation map from the same
        # raw patch outputs, always active in HiVG's released training runs.
        self.seg_conv1 = nn.ConvTranspose2d(vl_hidden_dim, vl_hidden_dim, kernel_size=2, stride=2, bias=False)
        self.seg_conv2 = nn.ConvTranspose2d(vl_hidden_dim, vl_hidden_dim, kernel_size=2, stride=2, bias=False)
        self.seg_conv3 = nn.ConvTranspose2d(vl_hidden_dim, vl_hidden_dim, kernel_size=2, stride=2, bias=False)

        # Box regression head: [REG] token → (cx,cy,w,h)
        self.bbox_embed = MLP(vl_hidden_dim, vl_hidden_dim, 4, num_layers=3)

        # Shared logit scale (used for CLC loss, initialised from CLIP convention)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ml_visual_projection.weight)
        nn.init.zeros_(self.ml_visual_projection.bias)
        nn.init.xavier_uniform_(self.visu_proj.weight)
        nn.init.zeros_(self.visu_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.visual_cls_proj.weight)
        nn.init.zeros_(self.visual_cls_proj.bias)
        nn.init.xavier_uniform_(self.text_eos_proj.weight)
        nn.init.zeros_(self.text_eos_proj.bias)
        nn.init.normal_(self.reg_token.weight, std=0.02)
        nn.init.normal_(self.vl_pos_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Visual forward (block-level, with MACB injection)
    # ------------------------------------------------------------------

    def _visual_grounding_forward(
        self,
        images: Tensor,
        text_hidden_states: list[Tensor],
    ) -> tuple[dict[int, Tensor], Tensor]:
        """Run LoopViT block-by-block, inject MACB residuals, stash intermediates.

        Args:
            images: [B, 3, H, W]
            text_hidden_states: K × [B, text_seq_len, text_hidden_dim]

        Returns:
            stash: dict mapping exec_idx ∈ {1,4,8,12} → [B, N_vis, visual_dim]
            cls_token: [B, visual_dim] after head_norm
        """
        x = self.visual.image_tokens(images)   # [B, 1+num_patches, visual_dim]
        stash: dict[int, Tensor] = {}
        exec_idx = 0

        for block in self.visual.iter_blocks():
            x = block(x)
            exec_idx += 1

            if exec_idx in _EXTRACT_EXECS:
                stash[exec_idx] = x

            if exec_idx in _INJECT_EXECS:
                slot = _INJECT_SLOT[exec_idx]
                delta = self.macb[slot](x, text_hidden_states)
                x = x + delta

        x = self.visual.head_norm(x)
        return stash, x[:, 0, :]   # cls token [B, visual_dim]

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, images: Tensor, token_ids: Tensor) -> GroundingOutput:
        """
        Args:
            images:    [B, 3, H, W] preprocessed images.
            token_ids: [B, 77] CLIP token ids.

        Returns:
            GroundingOutput with all tensors needed for loss computation.
        """
        B = images.shape[0]

        # --- Text encoding ---
        text_eos_raw, text_hidden_states = self.text.forward_with_hidden(token_ids)
        # text_eos_raw: [B, embed_dim] from CLIP projection
        # text_hidden_states: K × [B, 77, text_hidden_dim]

        # --- Visual encoding with block-level MACB injection ---
        stash, vis_cls_raw = self._visual_grounding_forward(images, text_hidden_states)

        # --- Multi-level visual feature aggregation ---
        # Stack stash at execs {1,4,8,12} → [B, N_vis, 4*visual_dim]
        ml_vis = torch.cat([stash[1], stash[4], stash[8], stash[12]], dim=-1)
        # [B, N_vis, embed_dim]  (N_vis = 1 + num_patches)
        ml_vis = self.ml_visual_projection(ml_vis)

        # Separate CLS and patch tokens after projection
        vis_cls_embed = ml_vis[:, 0, :]           # [B, embed_dim]
        patch_tokens = ml_vis[:, 1:, :]           # [B, num_patches, embed_dim]

        # --- Project text to embed_dim for VL encoder ---
        # Use last-layer text hidden states projected to embed_dim
        # text_hidden_states[-1]: [B, 77, text_hidden_dim]
        text_tokens_proj = self.text_proj(text_hidden_states[-1])   # [B, 77, vl_hidden_dim]

        # --- CLC loss embeddings (L2-normalised) ---
        img_cls_norm = F.normalize(self.visual_cls_proj(vis_cls_raw), dim=-1)
        # text_eos_raw is already in embed_dim space from CLIP projection
        text_eos_norm = F.normalize(text_eos_raw, dim=-1)

        # Logit-scaled similarity [B, B]
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_eos_norm, img_cls_norm.t()) * logit_scale

        # --- VL encoder input assembly ---
        # Project visual patch tokens to vl_hidden_dim
        visu_src = self.visu_proj(patch_tokens)   # [B, num_patches, vl_hidden_dim]

        # [REG] token: [B, 1, vl_hidden_dim]
        reg = self.reg_token.weight.unsqueeze(0).expand(B, -1, -1)

        # CLS as a single token, projected to vl_hidden_dim like the patches
        cls_src = self.visu_proj(vis_cls_embed.unsqueeze(1))   # [B, 1, vl_hidden_dim]

        # Concatenate: [REG, CLS, patches, text_tokens] → [B, L_total, D]
        vl_src = torch.cat([reg, cls_src, visu_src, text_tokens_proj], dim=1)

        # Convert to [L, B, D] for nn.MultiheadAttention (seq-first)
        vl_src = vl_src.permute(1, 0, 2)

        # Positional embedding [L, B, D]
        pos = self.vl_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        # Padding mask: True where tokens should be ignored.
        # text padding: positions with token_id == 0 are padding
        text_pad_mask = (token_ids == 0)   # [B, 77]
        # visual and [REG]/[CLS] tokens are never masked
        vis_pad = torch.zeros(B, 1 + 1 + self.num_patches,
                              dtype=torch.bool, device=images.device)
        key_padding_mask = torch.cat([vis_pad, text_pad_mask], dim=1)  # [B, L]

        # Run grounding encoder
        vg_hs = self.vl_encoder(vl_src, key_padding_mask=key_padding_mask, pos=pos)
        # vg_hs: [L, B, D]

        # --- Box regression ---
        pred_box = self.bbox_embed(vg_hs[0]).sigmoid()   # [B, 4]

        # --- RTCC: patch-level text-visual similarity ---
        # Visual patch outputs from grounding encoder [B, num_patches, D], raw
        # (both the RTCC and seg-mask heads below branch off this same tensor)
        patch_hs = vg_hs[2: 2 + self.num_patches].permute(1, 0, 2)
        # Text EOS output from grounding encoder [B, D]
        text_hs = vg_hs[2 + self.num_patches:].permute(1, 0, 2)  # [B, 77, D]
        eos_pos = token_ids.argmax(dim=-1)  # position of EOS token
        text_eos_hs = text_hs[torch.arange(B, device=images.device), eos_pos]   # [B, D]
        text_eos_hs = F.normalize(text_eos_hs, dim=-1)

        # visu_token_norm/mlp transform before the dot-product (HiVG's
        # clip_last_layer_features) — RTCC similarity per patch [B, num_patches]
        rtcc_patch_hs = self.visu_token_mlp(self.visu_token_norm(patch_hs))
        visu_sim = (rtcc_patch_hs * text_eos_hs.unsqueeze(1)).sum(dim=-1)

        # --- use_mask_loss: second, higher-res soft-segmentation map ---
        patch_num = int(self.num_patches ** 0.5)
        seg_features = patch_hs.permute(0, 2, 1).reshape(B, -1, patch_num, patch_num)
        seg_features = self.seg_conv3(self.seg_conv2(self.seg_conv1(seg_features)))
        seg_features = seg_features.permute(0, 2, 3, 1)   # [B, 112, 112, D]
        seg_mask = (text_eos_hs.reshape(B, 1, 1, -1) * seg_features).sum(dim=-1).unsqueeze(1)  # [B,1,112,112]

        return GroundingOutput(
            pred_box=pred_box,
            logits_per_text=logits_per_text,
            visu_sim=visu_sim,
            seg_mask=seg_mask,
            img_cls_embed=img_cls_norm,
            text_eos_embed=text_eos_norm,
        )
