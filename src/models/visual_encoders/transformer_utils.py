import math
from typing import Tuple


import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import torchvision.transforms.v2 as T


class SwiGLU(nn.Module):

    def __init__(
        self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()

        hidden = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(dim, hidden, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.w_out = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = x1 * F.silu(x2)
        x = self.drop(x)
        x = self.w_out(x)
        x = self.drop(x)
        return x


class GELU(nn.Module):

    def __init__(
        self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()

        hidden = int(dim * mlp_ratio)
        self.w_in = nn.Linear(dim, hidden, bias=bias)
        self.w_out = nn.Linear(hidden, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_in(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.w_out(x)
        x = self.drop(x)
        return x


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.affine:
            x = x * self.weight
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # scale (1/√head_dim) is applied internally by F.scaled_dot_product_attention

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.attn_dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        x = x.transpose(1, 2).contiguous().view(B, N, D)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        swiglu: bool = True,
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout
        )

        if swiglu:
            self.mlp = SwiGLU(dim, mlp_ratio=mlp_ratio, dropout=dropout, bias=True)
        else:
            self.mlp = GELU(dim, mlp_ratio=mlp_ratio, dropout=dropout, bias=True)

        self.norm1 = RMSNorm(dim, eps=norm_eps, affine=True)
        self.norm2 = RMSNorm(dim, eps=norm_eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        final_norm: bool = True,
        norm_eps: float = 1e-5,
        swiglu: bool = True,
    ):

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_eps=norm_eps,
                    swiglu=swiglu,
                )
                for _ in range(depth)
            ]
        )

        self.norm = (
            RMSNorm(dim, eps=norm_eps, affine=True) if final_norm else nn.Identity()
        )

    def forward(
        self, tokens: torch.Tensor, return_pre_norm: bool = False
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x)
        if return_pre_norm:
            return x
        return self.norm(x)


class RandomMixUp(nn.Module):

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for RandomMixUp.")

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0 for RandomMixUp.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


class RandomCutMix(nn.Module):

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for RandomCutMix.")

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0 for RandomCutMix.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )

        _, H, W = F.get_dimensions(batch)
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, y1:y2, x1:x2] = batch_rolled[:, y1:y2, x1:x2]
        lambda_param = float(1.0 - ((x2 - x1) * (y2 - y1) / (W * H)))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes):

    mixup_cutmix = []

    if mixup_alpha > 0:
        mixup_cutmix.append(T.MixUp(num_classes=num_classes, alpha=mixup_alpha))
    if cutmix_alpha > 0:
        mixup_cutmix.append(T.CutMix(num_classes=num_classes, alpha=cutmix_alpha))

    return mixup_cutmix
