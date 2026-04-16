"""Compile-friendly LIF neuron for MSFormer.

Drop-in replacement for spikingjelly's LIFNode that eliminates all hard
torch.compile graph-break sources:

  - No ``torch.autograd.Function`` for surrogate gradients.
    Uses the straight-through estimator (STE) trick instead:
    ``spike = surrogate + (heaviside - surrogate).detach()``
    Forward value = heaviside (binary). Backward gradient = surrogate derivative.

  - No ``_memories`` dict dispatch.
    ``v`` is a plain Python attribute on a vanilla ``nn.Module``.

  - No ``isinstance(self.v, float)`` type guard inside traced code.
    ``v`` is initialised as a 0-dim tensor in ``__init__`` so Dynamo always
    sees a tensor type and never generates a float-equality guard.

  - No backend dispatch (cupy / triton) — pure torch only.

Remaining guards (acceptable):
  - Tensor shape guard in ``_ensure_v``: fires once on the first forward call
    when the 0-dim sentinel is replaced with the correctly-shaped tensor, then
    never again as long as input shape is constant.
  - ``step_mode`` specialisation: Dynamo compiles one version per value.
  - Python loop in ``multi_step_forward``: Dynamo unrolls for static T (e.g. 4).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = ["LIFNode", "reset_lif_states", "init_lif_states"]


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
    """Compile-friendly Leaky Integrate-and-Fire neuron.

    Implements hard reset with decay_input (spikingjelly default config):

        v_charged = v + (x - (v - v_reset)) / tau
        spike     = heaviside(v_charged - v_threshold)
        v_new     = v_reset * spike_d + (1 - spike_d) * v_charged

    where ``spike_d = spike.detach()`` when ``detach_reset=True``.

    Supports both single-step (``step_mode="s"``, input ``[B, ...]``) and
    multi-step (``step_mode="m"``, input ``[T, B, ...]``) modes.

    Args:
        tau: Membrane time constant.
        v_threshold: Spike threshold voltage.
        v_reset: Reset voltage after a spike (hard reset).
        detach_reset: Detach spike from the reset computation graph to prevent
            double-counting gradients through v (matches spikingjelly default).
        surrogate: Surrogate gradient function. ``"sigmoid"`` or ``"atan"``.
        surrogate_alpha: Sharpness parameter for the surrogate. Higher = closer
            to true heaviside in forward, steeper gradient in backward.
        step_mode: ``"s"`` for single-step, ``"m"`` for multi-step.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        surrogate: str = "sigmoid",
        surrogate_alpha: float = 4.0,
        step_mode: str = "m",
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
        self.step_mode = step_mode
        # Register v as a non-persistent buffer so model.to(device) / .cuda()
        # moves it automatically.  Non-persistent means it is excluded from
        # state_dict() — membrane potentials are transient, not model weights.
        # Starts as a 0-dim sentinel; _ensure_v replaces it with the correctly
        # shaped tensor on the first forward (one shape-guard recompile, then
        # stable because device already matches the input).
        self.register_buffer("v", torch.tensor(v_reset), persistent=False)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _ensure_v(self, x: torch.Tensor) -> None:
        """Ensure v is a tensor matching x's shape/device/dtype.

        Re-allocates only when the shape changed (e.g. last partial batch of
        an epoch, or train vs. val batch sizes differ). Re-allocation breaks
        CUDA Graph compatibility for that one call, but is unavoidable when
        the shape genuinely changes.
        """
        if self.v.shape != x.shape:
            self.v = torch.full_like(x, self.v_reset)

    def reset(self) -> None:
        """Reset v to a 0-dim sentinel tensor, preserving device/dtype.

        This discards the existing shaped tensor so the next forward call will
        re-allocate via _ensure_v. Breaks CUDA Graphs — use
        ``inplace_reset()`` when torch.compile with ``reduce-overhead``
        (CUDA Graphs) is active.
        """
        self.v = torch.tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)

    def init_state(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Pre-allocate v as a full-shaped tensor before the first forward.

        Useful for warming up to the correct shape before ``torch.compile``
        so the initial shape-guard recompile happens outside the compiled
        region. The simpler alternative is to run one dummy forward and then
        call ``reset_lif_states(model)``.
        """
        self.v = torch.full(shape, self.v_reset, device=device, dtype=dtype)

    def inplace_reset(self) -> None:
        """Reset v to v_reset in-place, preserving its GPU memory address.

        Required for CUDA Graph compatibility (``torch.compile`` with
        ``mode="reduce-overhead"``).
        """
        self.v.detach_().fill_(self.v_reset)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _fire(self, v_shifted: torch.Tensor) -> torch.Tensor:
        """Apply surrogate spike function to ``v - v_threshold``."""
        return self._surrogate_fn(v_shifted, self.surrogate_alpha)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one timestep. x: [B, ...]"""
        self._ensure_v(x)
        v_charged = self.v + (x - (self.v - self.v_reset)) / self.tau
        spike = self._fire(v_charged - self.v_threshold)
        spike_d = spike.detach() if self.detach_reset else spike
        self.v = self.v_reset * spike_d + (1.0 - spike_d) * v_charged
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Process T timesteps. x_seq: [T, B, ...]

        The Python loop is unrolled by Dynamo for static T (e.g. T=4), giving
        T fused graph segments with no Python overhead per segment.
        """
        self._ensure_v(x_seq[0])
        spike_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(self.single_step_forward(x_seq[t]))
        return torch.stack(spike_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "m":
            return self.multi_step_forward(x)
        return self.single_step_forward(x)

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, detach_reset={self.detach_reset}, "
            f"surrogate_alpha={self.surrogate_alpha}, step_mode={self.step_mode!r}"
        )


# ---------------------------------------------------------------------------
# Module-level reset helpers
# ---------------------------------------------------------------------------

def init_lif_states(
    net: nn.Module,
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    """Pre-allocate membrane potentials on all LIFNodes in ``net``.

    Call once before ``torch.compile`` with the expected single-timestep input
    shape. Because each LIFNode lives at a different feature-map resolution,
    the easiest approach is to run one dummy forward instead::

        with torch.no_grad():
            model(dummy_batch)   # all _ensure_v calls fire, v becomes tensors
        reset_lif_states(model)  # fill with v_reset in-place, keep tensors
        compiled = torch.compile(model, ...)

    This helper is useful when you know the exact tensor shape per node
    (e.g. unit tests or custom pre-warming loops).
    """
    target = getattr(net, "_orig_mod", net)
    for m in target.modules():
        if isinstance(m, LIFNode):
            m.init_state(shape, device=device, dtype=dtype)


def reset_lif_states(net: nn.Module, inplace: bool = True) -> None:
    """Reset membrane potentials of all LIFNode instances in ``net``.

    Args:
        net: Any ``nn.Module`` containing ``LIFNode`` submodules.
        inplace: If True (default), use ``inplace_reset()`` which preserves
            GPU memory addresses — required for CUDA Graph compatibility.
            If False, use ``reset()`` which discards the tensor.
    """
    # Unwrap compiled models so we traverse the original submodules.
    target = getattr(net, "_orig_mod", net)
    for m in target.modules():
        if isinstance(m, LIFNode):
            if inplace:
                m.inplace_reset()
            else:
                m.reset()
