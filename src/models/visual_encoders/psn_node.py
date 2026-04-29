"""Compile-friendly PSN variants and shape-adapting wrapper for QKFormer/MSFormer.

QKFormer and MSFormer call spiking neurons with tensors of shape [T*B, *] —
the batch and timestep dimensions are merged before entering convs/BN, then
split afterward. The spikingjelly PSN family requires [T, N] where the first
dimension equals the fixed T stored in the neuron's weight matrix.

PSNAdapter solves this by:
  1. Receiving [T*B, *] (any trailing dimensions)
  2. Reshaping to [T, B*rest] so the first dim is exactly T
  3. Applying the wrapped PSN (which does H = W @ X, W ∈ R^{T×T})
  4. Reshaping the output back to the original [T*B, *] shape

CompileFriendlyPSN / CompileFriendlyMaskedPSN / CompileFriendlySlidingPSN are
drop-in replacements for spikingjelly's PSN / MaskedPSN / SlidingPSN that
eliminate all torch.compile graph-break sources:

  - No torch.autograd.Function for surrogate gradients.
    Uses the STE trick instead (same pattern as lif_node.py):
      spike = surrogate + (heaviside - surrogate).detach()

  - No backend dispatch (cupy / triton) — pure PyTorch only.

  - No mutable module state during forward.

Interface for all three: [T, N] → [T, N]  (same as spikingjelly variants).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PSNAdapter",
    "CompileFriendlyPSN",
    "CompileFriendlyMaskedPSN",
    "CompileFriendlySlidingPSN",
]


# ---------------------------------------------------------------------------
# STE spike function (shared)
# ---------------------------------------------------------------------------

def _ste_spike(h: torch.Tensor, alpha: float) -> torch.Tensor:
    """Heaviside forward, sigmoid-surrogate backward via STE.

    Forward value = (h >= 0).  Backward gradient = d/dh sigmoid(alpha*h).
    No custom autograd.Function — fully traceable by torch.compile.
    """
    spike = (h >= 0.0).to(h.dtype)
    surrogate = torch.sigmoid(alpha * h)
    return surrogate + (spike - surrogate).detach()


# ---------------------------------------------------------------------------
# Compile-friendly PSN variants  (interface: [T, N] → [T, N])
# ---------------------------------------------------------------------------

class CompileFriendlyPSN(nn.Module):
    """Full T×T parallel spiking neuron — compile-friendly replacement for spikingjelly PSN.

    Computes: H = W @ X - theta,  spike = STE_heaviside(H)

    W ∈ R^{T×T} and theta ∈ R^{T×1} are learnable parameters.
    Initialized as identity + unit threshold to match spikingjelly defaults.

    Args:
        T: Number of SNN timesteps.
        surrogate_alpha: Sigmoid sharpness for the surrogate gradient.
    """

    def __init__(self, T: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.T = T
        self.surrogate_alpha = surrogate_alpha
        self.W = nn.Parameter(torch.eye(T))
        self.theta = nn.Parameter(torch.ones(T, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [T, N] input (N = B * all_other_dims, flattened by PSNAdapter).

        Returns:
            Spike tensor of shape [T, N].
        """
        h = self.W @ x - self.theta
        return _ste_spike(h, self.surrogate_alpha)

    def extra_repr(self) -> str:
        return f"T={self.T}, surrogate_alpha={self.surrogate_alpha}"


class CompileFriendlyMaskedPSN(nn.Module):
    """Causal masked PSN — compile-friendly replacement for spikingjelly MaskedPSN.

    Computes: H = (W * mask) @ X - theta,  spike = STE_heaviside(H)

    The causal mask enforces W[i,j] = 0 unless 0 ≤ i-j ≤ k, so each output
    timestep only depends on the current and k previous input timesteps.

    Args:
        T: Number of SNN timesteps.
        k: Bandwidth (receptive field = k+1 timesteps).
        surrogate_alpha: Sigmoid sharpness for the surrogate gradient.
    """

    def __init__(self, T: int, k: int = 2, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.T = T
        self.k = k
        self.surrogate_alpha = surrogate_alpha
        self.W = nn.Parameter(torch.eye(T))
        self.theta = nn.Parameter(torch.ones(T, 1))
        # Fixed causal mask: mask[i,j] = 1 iff 0 <= i-j <= k
        mask = torch.zeros(T, T)
        for i in range(T):
            mask[i, max(0, i - k): i + 1] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [T, N] input.

        Returns:
            Spike tensor of shape [T, N].
        """
        h = (self.W * self.mask) @ x - self.theta
        return _ste_spike(h, self.surrogate_alpha)

    def extra_repr(self) -> str:
        return f"T={self.T}, k={self.k}, surrogate_alpha={self.surrogate_alpha}"


class CompileFriendlySlidingPSN(nn.Module):
    """Sliding-window PSN — compile-friendly replacement for spikingjelly SlidingPSN.

    Each output timestep h[t] is a weighted sum of inputs x[t], x[t-1], ..., x[t-k]
    using a shared weight vector (same weights for every neuron).  Implemented as
    a causal 1D convolution along the T axis.

    Args:
        T: Number of SNN timesteps.
        k: Sliding window size (kernel = k+1 taps).
        surrogate_alpha: Sigmoid sharpness for the surrogate gradient.
    """

    def __init__(self, T: int, k: int = 2, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.T = T
        self.k = k
        self.surrogate_alpha = surrogate_alpha
        # kernel_size = k+1, shared across all neurons; bias acts as -theta
        self.conv = nn.Conv1d(1, 1, kernel_size=k + 1, bias=True)
        nn.init.ones_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [T, N] input.

        Returns:
            Spike tensor of shape [T, N].
        """
        # Causal padding: pad k zeros on the left of the T dimension
        x_t = x.T.unsqueeze(1)                    # [N, 1, T]
        x_padded = F.pad(x_t, (self.k, 0))        # [N, 1, T+k]
        h = self.conv(x_padded).squeeze(1).T - self.theta   # [T, N]
        return _ste_spike(h, self.surrogate_alpha)

    def extra_repr(self) -> str:
        return f"T={self.T}, k={self.k}, surrogate_alpha={self.surrogate_alpha}"


# ---------------------------------------------------------------------------
# Shape adapter  (unchanged — wraps any [T, N] → [T, N] PSN)
# ---------------------------------------------------------------------------

class PSNAdapter(nn.Module):
    """Drop-in adapter that gives PSN variants a [T*B, *] → [T*B, *] interface.

    Args:
        psn: A PSN module with [T, N] → [T, N] interface.
        T: Number of SNN timesteps (must match the PSN's internal T).
    """

    def __init__(self, psn: nn.Module, T: int) -> None:
        super().__init__()
        self.psn = psn
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape [T*B, *] where the leading dim equals T*batch_size.

        Returns:
            Spike tensor of the same shape as x.
        """
        orig_shape = x.shape
        # Reshape [T*B, *] → [T, B*rest_flat] so PSN sees exactly T timesteps
        x_flat = x.reshape(self.T, -1)
        spike_flat = self.psn(x_flat)
        return spike_flat.reshape(orig_shape)

    def extra_repr(self) -> str:
        return f"T={self.T}"
