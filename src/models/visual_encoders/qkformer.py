"""QKFormer: Hierarchical Spiking Transformer with Q-K attention for image encoding.

Adapted from QKFormer/imagenet/qkformer.py with the following changes:
- Replaced spikingjelly.clock_driven (old API) with the project's compile-friendly
  LIFNode (src/models/visual_encoders/lif_node.py) — same strategy as msformer.py
- Removed module-level globals; SNN hyperparameters are passed via SNNParams dataclass
- Removed noisy compute_non_zero_rate debug prints
- Added num_classes=0 support to return raw feature embeddings (no classification head)
- QKFormer_10_512 is pre-configured for 224x224 ImageNet-scale CLIP image encoding

Architecture overview (depths=10):
  Stage 1 (1 block):  Token_QK_Attention — Q-K only (no V), embed_dims//4
  Stage 2 (2 blocks): Token_QK_Attention — Q-K only (no V), embed_dims//2
  Stage 3 (7 blocks): Spiking_Self_Attention — full Q-K-V,  embed_dims
Each stage is preceded by a patch-embedding/downsampling module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, to_2tuple

from src.models.visual_encoders.lif_node import LIFNode as _LIFNode

__all__ = ["QKFormer_10_512", "SNNParams"]


# ---------------------------------------------------------------------------
# SNN configuration  (mirrors msformer.SNNParams exactly)
# ---------------------------------------------------------------------------

@dataclass
class SNNParams:
    """Hyperparameters for spiking neurons throughout QKFormer.

    Attributes:
        neuron_type: Neuron model. One of 'lif', 'sj_lif', 'plif', 'nlif', 'glif'.
        v_threshold: Spike threshold voltage (default 1.0).
        tau: Membrane time constant (default 2.0).
        backend: Computation backend. 'torch' is always safe; 'triton' for
                 newer GPUs. 'cupy' requires CuPy to be installed.
    """

    neuron_type: str = "lif"
    v_threshold: float = 1.0
    tau: float = 2.0
    backend: str = "torch"


def _build_lif_node(snn: SNNParams, v_threshold: Optional[float] = None) -> nn.Module:
    """Instantiate a spiking neuron according to SNNParams.

    Args:
        snn: SNN configuration.
        v_threshold: Override spike threshold (e.g. 0.5 for attention layers).
                     If None, uses snn.v_threshold.

    Returns:
        A neuron module in multi-step mode.
    """
    vth = v_threshold if v_threshold is not None else snn.v_threshold

    if snn.neuron_type in ("lif", "nlif"):
        return _LIFNode(
            tau=snn.tau,
            v_threshold=vth,
            v_reset=0.0,
            detach_reset=True,
            surrogate="sigmoid",
            surrogate_alpha=4.0,
        )
    # Fallback to spikingjelly for neuron types not yet ported.
    from spikingjelly.activation_based import neuron as _sj_neuron

    if snn.neuron_type == "sj_lif":
        return _sj_neuron.LIFNode(
            step_mode="m",
            tau=snn.tau,
            v_threshold=vth,
            v_reset=0.0,
            detach_reset=True,
            backend=snn.backend,
        )
    if snn.neuron_type == "plif":
        return _sj_neuron.ParametricLIFNode(
            step_mode="m",
            init_tau=snn.tau,
            detach_reset=True,
            v_threshold=vth,
            backend=snn.backend,
        )
    if snn.neuron_type == "glif":
        return _sj_neuron.GatedLIFNode(step_mode="m", backend=snn.backend)
    raise ValueError(
        f"Unknown neuron_type '{snn.neuron_type}'. "
        "Choose from: 'lif', 'sj_lif', 'plif', 'nlif', 'glif'."
    )


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer spiking MLP using 1x1 convolutions."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = _build_lif_node(snn)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = _build_lif_node(snn)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, W, H = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, W, H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, W, H).contiguous()
        x = self.fc2_lif(x)
        return x


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class Token_QK_Attention(nn.Module):
    """Token-based Q-K spiking attention (no value projection).

    Used in stages 1 and 2.  Computes attention as a sparse binary mask derived
    from the sum of Q across heads, then modulates K with it — linear in sequence
    length and more energy-efficient than full QKV attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = _build_lif_node(snn)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = _build_lif_node(snn)

        # Lower threshold for the attention gate — same as original (0.5)
        self.attn_lif = _build_lif_node(snn, v_threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(3)                          # [T, B, C, N]
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)              # [T*B, C, N]

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        # Attention gate: sum Q over the head-dim dimension, spike it, multiply K
        q = torch.sum(q, dim=3, keepdim=True)    # [T, B, heads, 1, N]
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)                   # [T, B, heads, C//heads, N]

        x = x.flatten(2, 3)                      # [T, B, C, N]
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        return x


class Spiking_Self_Attention(nn.Module):
    """Full Q-K-V spiking self-attention.

    Used in stage 3.  Computes attention as (Q @ (K^T @ V)) * scale, all in
    the spiking domain — no softmax.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = _build_lif_node(snn)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = _build_lif_node(snn)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = _build_lif_node(snn)

        self.attn_lif = _build_lif_node(snn, v_threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _build_lif_node(snn)

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(3)                          # [T, B, C, N]
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)              # [T*B, C, N]

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = (q_conv_out.transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())   # [T, B, heads, N, C//heads]

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = (k_conv_out.transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = (v_conv_out.transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        # Spiking attention: Q @ (K^T @ V) * scale
        x = k.transpose(-2, -1) @ v             # [T, B, heads, C//heads, C//heads]
        x = (q @ x) * self.scale                # [T, B, heads, N, C//heads]

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)                     # [T*B, C, N]
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TokenSpikingTransformer(nn.Module):
    """Transformer block using Token_QK_Attention (stages 1 & 2)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        self.tssa = Token_QK_Attention(dim, num_heads=num_heads, snn=snn)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, snn=snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class SpikingTransformer(nn.Module):
    """Transformer block using Spiking_Self_Attention (stage 3)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        self.attn = Spiking_Self_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, snn=snn,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, snn=snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Patch embedding stages
# ---------------------------------------------------------------------------

class PatchEmbedInit(nn.Module):
    """Initial patch embedding: 3 conv layers with BN and residual, 4x spatial reduction."""

    def __init__(
        self,
        img_size_h: int = 128,
        img_size_w: int = 128,
        patch_size: int = 4,
        in_channels: int = 2,
        embed_dims: int = 256,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H = self.image_size[0] // patch_size[0]
        self.W = self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # Downsampling path
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = _build_lif_node(snn)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = _build_lif_node(snn)

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = _build_lif_node(snn)

        # Residual shortcut (embed_dims//2 → embed_dims, stride 2)
        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x


class PatchEmbeddingStage(nn.Module):
    """Subsequent patch embedding stage: 2 conv layers with BN, residual, 2x spatial reduction."""

    def __init__(
        self,
        img_size_h: int = 128,
        img_size_w: int = 128,
        patch_size: int = 4,
        in_channels: int = 2,
        embed_dims: int = 256,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H = self.image_size[0] // patch_size[0]
        self.W = self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = _build_lif_node(snn)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = _build_lif_node(snn)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------

class hierarchical_spiking_transformer(nn.Module):
    """Hierarchical QKFormer backbone.

    Three-stage transformer:
      patch_embed1 → stage1 (Token_QK_Attention, 1 block)
      patch_embed2 → stage2 (Token_QK_Attention, 2 blocks)
      patch_embed3 → stage3 (Spiking_Self_Attention, depths-3 blocks)

    Args:
        T: SNN simulation timesteps.
        snn: Spiking neuron configuration.
        img_size_h, img_size_w: Input image dimensions.
        patch_size: Not used for striding (kept for API compatibility).
        in_channels: Number of input image channels.
        num_classes: If > 0 adds a linear classification head; if 0 the head
                     is nn.Identity (returns raw embed_dims features).
        embed_dims: Final embedding dimension.  Stages use dims//4, dims//2, dims.
        num_heads: Number of attention heads (shared across all stages).
        mlp_ratios: MLP hidden dim multiplier.
        depths: Total number of transformer blocks (1 in stage1, 2 in stage2,
                depths-3 in stage3).
    """

    def __init__(
        self,
        T: int = 4,
        img_size_h: int = 128,
        img_size_w: int = 128,
        patch_size: int = 16,
        in_channels: int = 2,
        num_classes: int = 11,
        embed_dims: int = 256,
        num_heads: int = 4,
        mlp_ratios: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        depths: int = 10,
        sr_ratios: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        patch_embed1 = PatchEmbedInit(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims // 4, snn=snn,
        )
        stage1 = nn.ModuleList([
            TokenSpikingTransformer(
                dim=embed_dims // 4, num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios, snn=snn,
            )
            for j in range(1)
        ])

        patch_embed2 = PatchEmbeddingStage(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims // 2, snn=snn,
        )
        stage2 = nn.ModuleList([
            TokenSpikingTransformer(
                dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios, snn=snn,
            )
            for j in range(2)
        ])

        patch_embed3 = PatchEmbeddingStage(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims, snn=snn,
        )
        stage3 = nn.ModuleList([
            SpikingTransformer(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios, snn=snn,
            )
            for j in range(depths - 3)
        ])

        self.patch_embed1 = patch_embed1
        self.patch_embed2 = patch_embed2
        self.patch_embed3 = patch_embed3
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

        # Classification head; nn.Identity when num_classes == 0 (feature mode)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone on a pre-tiled input.

        Args:
            x: [T, B, C, H, W] tensor (already replicated across T timesteps).

        Returns:
            [T, B, embed_dims] feature tensor (spatial dimensions averaged out).
        """
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        # [T, B, embed_dims, H', W'] → average over spatial → [T, B, embed_dims]
        return x.flatten(3).mean(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward for classification / sanity-check training.

        Args:
            x: [B, C, H, W] image tensor.

        Returns:
            [B, num_classes] logits, or [B, embed_dims] features when
            num_classes == 0.
        """
        T = self.T
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)   # [T, B, C, H, W]
        x = self.forward_features(x)                 # [T, B, D]
        x = self.head(x.mean(0))                     # [B, num_classes or D]
        return x


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def QKFormer_10_512(
    T: int = 4,
    snn: Optional[SNNParams] = None,
    num_classes: int = 0,
    **kwargs,
) -> hierarchical_spiking_transformer:
    """Build QKFormer with embed_dim=512, 10 transformer blocks, 224×224 input.

    Args:
        T: SNN simulation timesteps (default 4).
        snn: Spiking neuron configuration.  Defaults to SNNParams().
        num_classes: Output classes.  Use 0 (default) for CLIP feature mode,
                     1000 for ImageNet classification.

    Returns:
        Configured hierarchical_spiking_transformer instance.
    """
    if snn is None:
        snn = SNNParams()
    return hierarchical_spiking_transformer(
        T=T,
        snn=snn,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=10,
        sr_ratios=1,
        **kwargs,
    )
