"""MSFormer: Multi-Scale Spiking Vision Transformer for image encoding.

Adapted from MSViT/imagenet/msformer.py with the following changes:
- Replaced spikingjelly.clock_driven (old API) with spikingjelly.activation_based
- Removed module-level globals; SNN hyperparameters are passed via SNNParams dataclass
- Removed noisy compute_non_zero_rate debug prints
- Added num_classes=0 support to return raw feature embeddings (no classification head)
- MSFormer_10_512 is pre-configured for 224x224 ImageNet-scale CLIP image encoding
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

from src.models.visual_encoders.lif_node import LIFNode as _LIFNode

__all__ = ["MSFormer_10_512", "SNNParams"]


# ---------------------------------------------------------------------------
# SNN configuration
# ---------------------------------------------------------------------------


@dataclass
class SNNParams:
    """Hyperparameters for spiking neurons throughout MSFormer.

    Attributes:
        neuron_type: Neuron model. One of 'lif', 'plif', 'nlif', 'glif'.
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
        A spikingjelly neuron module in multi-step mode.
    """
    vth = v_threshold if v_threshold is not None else snn.v_threshold

    if snn.neuron_type in ("lif", "nlif"):
        # Use our compile-friendly LIFNode. plif/glif still need spikingjelly.
        return _LIFNode(
            tau=snn.tau,
            v_threshold=vth,
            v_reset=0.0,
            detach_reset=True,
            surrogate="sigmoid",
            surrogate_alpha=4.0,
            step_mode="m",
        )
    # Fallback to spikingjelly for neuron types not yet ported.
    from spikingjelly.activation_based import neuron as _sj_neuron

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
        "Choose from: 'lif', 'plif', 'nlif', 'glif'."
    )


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class Token_MS_Attention(nn.Module):
    """Token-based Multi-Scale Spiking Self-Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        ms_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = _build_lif_node(snn)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = _build_lif_node(snn)

        self.p_conv = nn.Conv1d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.p_bn = nn.BatchNorm1d(dim)
        self.p_lif = _build_lif_node(snn)

        # Attention threshold is 0.5 (half of the default) per original paper
        self.attn_lif = _build_lif_node(snn, v_threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        p_conv_out = self.p_conv(x_for_qkv)
        p_conv_out = self.p_bn(p_conv_out).reshape(T, B, C, N)
        p_conv_out = self.p_lif(p_conv_out)
        p = p_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        q = torch.sum(q, dim=3, keepdim=True)
        p = torch.sum(p, dim=3, keepdim=True)
        attn = self.attn_lif(q + p)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x


class Spiking_Self_Attention(nn.Module):
    """Standard Spiking Self-Attention (used in stage 3)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        ms_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        snn: SNNParams = None,
    ):
        super().__init__()
        if snn is None:
            snn = SNNParams()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}."

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

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)

        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Spiking MLP block (2-layer Conv2d with LIF activations)."""

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

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = _build_lif_node(snn)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, W, H = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, W, H)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, W, H)
        x = self.fc2_lif(x)

        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class TokenSpikingTransformer(nn.Module):
    """Transformer block using Token_MS_Attention (stages 1 & 2)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        ms_scale=None,
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
        self.tssa = Token_MS_Attention(dim, num_heads, snn=snn)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, snn=snn
        )

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
        ms_scale=None,
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
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            ms_scale=ms_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            snn=snn,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, snn=snn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Patch embedding stages
# ---------------------------------------------------------------------------


class PatchEmbedInit(nn.Module):
    """Initial patch embedding: 2 × stride-2 MaxPool, producing H/4 × W/4 patches."""

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
        self.H = img_size_h // patch_size[0]
        self.W = img_size_w // patch_size[1]
        self.num_patches = self.H * self.W

        # Downsample 1: in_channels → embed_dims//2, stride 2
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj_lif = _build_lif_node(snn)

        # Downsample 2: embed_dims//2 → embed_dims, stride 2
        self.proj1_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj1_lif = _build_lif_node(snn)

        self.proj2_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = _build_lif_node(snn)

        # Residual shortcut
        self.proj_res_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4)
        x = self.proj1_lif(x).flatten(0, 1)

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4)
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 4, W // 4)
        x_feat = self.proj_res_lif(x_feat)

        return x + x_feat


class PatchEmbeddingStage(nn.Module):
    """Stage 2 patch embedding: stride-2 downsampling within the same embed dim."""

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
        self.H = img_size_h // patch_size[0]
        self.W = img_size_w // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj3_lif = _build_lif_node(snn)

        self.proj4_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = _build_lif_node(snn)

        # Residual shortcut
        self.proj_res_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1)
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj3_lif(x).flatten(0, 1)

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2)
        x_feat = self.proj_res_lif(x_feat)

        return x + x_feat


class PatchEmbeddingStage3(nn.Module):
    """Stage 3 patch embedding: upsizes embed dim from embed_dims//2 to embed_dims."""

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
        self.H = img_size_h // patch_size[0]
        self.W = img_size_w // patch_size[1]
        self.num_patches = self.H * self.W

        # Input has embed_dims//2 channels; output is embed_dims
        self.proj3_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj3_lif = _build_lif_node(snn)

        self.proj4_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = _build_lif_node(snn)

        # Residual shortcut
        self.proj_res_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = _build_lif_node(snn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1)
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj3_lif(x).flatten(0, 1)

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2)
        x_feat = self.proj_res_lif(x_feat)

        return x + x_feat


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class hierarchical_spiking_transformer(nn.Module):
    """Hierarchical Spiking Vision Transformer (MSFormer).

    3-stage hierarchical architecture with spiking neurons. Each stage
    progressively downsamples the spatial resolution while increasing the
    number of channels. The final output is the temporal mean of spike
    train features, yielding a [B, embed_dims] embedding.

    Args:
        T: Number of SNN simulation timesteps. The same input frame is
           repeated T times to produce a temporal input.
        img_size_h, img_size_w: Input image height/width.
        patch_size: Patch size used in all embedding stages.
        in_channels: Number of input image channels (3 for RGB).
        num_classes: If > 0, appends a linear classification head. If 0,
                     forward() returns raw [B, embed_dims] features.
        embed_dims: Feature dimension of the final stage (stage 3).
        num_heads: Number of attention heads in all transformer blocks.
        mlp_ratios: MLP expansion ratio.
        qkv_bias: Whether to add bias to Q/K/V projections.
        drop_rate: Dropout rate (currently unused in transformer blocks).
        attn_drop_rate: Attention dropout rate (currently unused).
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalisation class for transformer blocks.
        depths: Total number of transformer blocks (stage1=1, stage2=2,
                stage3=depths-3).
        sr_ratios: Spatial reduction ratio (currently unused).
        snn: SNN neuron configuration. Defaults to SNNParams().
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
        mlp_ratios: float = 4,
        qkv_bias: bool = False,
        ms_scale=None,
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
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 2,
            snn=snn,
        )
        stage1 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=embed_dims // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    ms_scale=ms_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    snn=snn,
                )
                for j in range(1)
            ]
        )

        patch_embed2 = PatchEmbeddingStage(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 2,
            snn=snn,
        )
        stage2 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=embed_dims // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    ms_scale=ms_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    snn=snn,
                )
                for j in range(2)
            ]
        )

        patch_embed3 = PatchEmbeddingStage3(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            snn=snn,
        )
        stage3 = nn.ModuleList(
            [
                SpikingTransformer(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    ms_scale=ms_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    snn=snn,
                )
                for j in range(depths - 3)
            ]
        )

        self.patch_embed1 = patch_embed1
        self.patch_embed2 = patch_embed2
        self.patch_embed3 = patch_embed3
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

        # Classification head (Identity when num_classes == 0)
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )

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
        """Run the 3-stage backbone.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            Feature tensor of shape [T, B, embed_dims] (spatial mean pooled).
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

        # Flatten spatial dims and mean-pool → [T, B, embed_dims]
        return x.flatten(3).mean(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification forward pass.

        Expands x to T timesteps, runs the backbone, averages across time,
        and passes through the head.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Logits [B, num_classes] if num_classes > 0, else features [B, embed_dims].
        """
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        x = self.forward_features(x)  # [T, B, embed_dims]
        x = self.head(x.mean(0))  # [B, num_classes or embed_dims]
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def MSFormer_10_512(
    T: int = 4,
    snn: SNNParams = None,
    num_classes: int = 0,
    **kwargs,
) -> hierarchical_spiking_transformer:
    """MSFormer with 10 transformer blocks and 512-dim embeddings.

    Pre-configured for 224×224 RGB input (ImageNet / CLIP scale). With
    num_classes=0 (default), forward_features().mean(0) returns [B, 512]
    embeddings suitable for use as a CLIP image encoder. Pass num_classes=1000
    for standalone ImageNet classification.

    Args:
        T: SNN simulation timesteps.
        snn: Spiking neuron configuration (defaults to SNNParams()).
        num_classes: Output classes. 0 = feature embedding mode (CLIP).
        **kwargs: Additional args forwarded to hierarchical_spiking_transformer.

    Returns:
        Configured hierarchical_spiking_transformer instance.
    """
    if snn is None:
        snn = SNNParams()
    return hierarchical_spiking_transformer(
        T=T,
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
        snn=snn,
        **kwargs,
    )
