"""Compile-friendly LIF neuron for MSFormer / QKFormer.

Drop-in replacement for spikingjelly's LIFNode that eliminates all hard
torch.compile graph-break sources:

  - No ``torch.autograd.Function`` for surrogate gradients.
    Uses the straight-through estimator (STE) trick instead:
    ``spike = surrogate + (heaviside - surrogate).detach()``
    Forward value = heaviside (binary). Backward gradient = surrogate derivative.

  - No ``_memories`` dict dispatch.
    No mutable module state at all — the membrane potential is a local
    variable inside ``forward``, initialised fresh each call.

  - No ``self.v`` rebinding during the forward pass.
    The previous implementation assigned ``self.v = new_tensor`` on every
    timestep.  Under ``torch.compile(mode="reduce-overhead")`` CUDA Graphs
    pin memory addresses at capture time; rebinding ``self.v`` to a new
    tensor on replay raises ``RuntimeError``.  Making ``v`` a local variable
    inside ``forward`` lets the compiler treat each step as SSA and manage
    all intermediate buffers statically.

  - No ``isinstance(self.v, float)`` type guard inside traced code.

  - No backend dispatch (cupy / triton) — pure torch only.

Remaining guards (acceptable):
  - Python loop in ``forward``: Dynamo unrolls for static T (e.g. 4).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = ["LIFNode"]


# ---------------------------------------------------------------------------
# Surrogate gradient functions — STE formulation, no custom autograd
# ---------------------------------------------------------------------------

def sigmoid_surrogate(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Sigmoid surrogate spike function.

    Forward  = heaviside(x)  [binary 0/1]
    Backward = d/dx sigmoid(alpha * x)  =  alpha * sigma * (1 - sigma)

    The STE identity ``f + (target - f).detach()`` has the value of ``target``
    in the forward pass but the gradient of ``f`` in the backward pass.
    """
    spike = (x >= 0.0).to(x.dtype)
    surrogate = torch.sigmoid(alpha * x)
    return surrogate + (spike - surrogate).detach()


def atan_surrogate(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """ATan surrogate spike function.

    Forward  = heaviside(x)
    Backward = d/dx [arctan(pi/2 * alpha * x) / pi + 0.5]
             = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
    """
    spike = (x >= 0.0).to(x.dtype)
    surrogate = 0.5 + torch.atan(math.pi / 2.0 * alpha * x) / math.pi
    return surrogate + (spike - surrogate).detach()


_SURROGATES = {"sigmoid": sigmoid_surrogate, "atan": atan_surrogate}


# ---------------------------------------------------------------------------
# LIF neuron
# ---------------------------------------------------------------------------

class LIFNode(nn.Module):
    """Compile-friendly stateless Leaky Integrate-and-Fire neuron.

    Implements hard reset with decay_input (spikingjelly default config):

        v_charged = v + (x - (v - v_reset)) / tau
        spike     = heaviside(v_charged - v_threshold)
        v_next    = v_reset * spike_d + (1 - spike_d) * v_charged

    where ``spike_d = spike.detach()`` when ``detach_reset=True``.

    Accepts multi-step input ``[T, B, ...]`` and returns ``[T, B, ...]``.
    Membrane potential is initialised to ``v_reset`` at the start of every
    ``forward`` call — no persistent state, no ``reset()`` bookkeeping.

    Args:
        tau: Membrane time constant.
        v_threshold: Spike threshold voltage.
        v_reset: Reset voltage after a spike (hard reset).
        detach_reset: Detach spike from the reset computation graph to prevent
            double-counting gradients through v (matches spikingjelly default).
        surrogate: Surrogate gradient function. ``"sigmoid"`` or ``"atan"``.
        surrogate_alpha: Sharpness parameter for the surrogate. Higher = closer
            to true heaviside in forward, steeper gradient in backward.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        surrogate: str = "sigmoid",
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__()
        if surrogate not in _SURROGATES:
            raise ValueError(f"surrogate must be one of {list(_SURROGATES)}; got {surrogate!r}")
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self._surrogate_fn = _SURROGATES[surrogate]
        self.surrogate_alpha = surrogate_alpha

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _fire(self, v_shifted: torch.Tensor) -> torch.Tensor:
        """Apply surrogate spike function to ``v - v_threshold``."""
        return self._surrogate_fn(v_shifted, self.surrogate_alpha)

    def _step(self, x: torch.Tensor, v_prev: torch.Tensor):
        """Single timestep: charge → fire → reset.

        Args:
            x: Input current for this timestep, shape ``[B, ...]``.
            v_prev: Membrane potential from the previous timestep.

        Returns:
            ``(spike, v_next)`` — both ``[B, ...]``.
        """
        v_charged = v_prev + (x - (v_prev - self.v_reset)) / self.tau
        spike = self._fire(v_charged - self.v_threshold)
        spike_d = spike.detach() if self.detach_reset else spike
        v_next = self.v_reset * spike_d + (1.0 - spike_d) * v_charged
        return spike, v_next

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Process a temporal sequence.

        Args:
            x_seq: Input tensor of shape ``[T, B, ...]``.

        Returns:
            Spike sequence of shape ``[T, B, ...]``.

        The Python loop is unrolled by Dynamo for static T (e.g. T=4), giving
        T fused graph segments with no Python overhead per segment.
        Membrane potential ``v`` is a local variable — no module attribute is
        mutated, satisfying CUDA Graph static-memory requirements.
        """
        v = torch.full_like(x_seq[0], self.v_reset)
        spike_seq = []
        for t in range(x_seq.shape[0]):
            spike, v = self._step(x_seq[t], v)
            spike_seq.append(spike)
        return torch.stack(spike_seq)

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, detach_reset={self.detach_reset}, "
            f"surrogate_alpha={self.surrogate_alpha}"
        )
