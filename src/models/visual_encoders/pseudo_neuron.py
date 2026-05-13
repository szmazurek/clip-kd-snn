"""PseudoNeuron and helpers ported from PseudoSNN (CVPR 2025).

During training the neuron is a single-pass ReLU + stochastic noise proxy;
no BPTT required. Each neuron owns a learnable logit that maps to a
timestep count T in [min_T, max_T] via STE-rounded sigmoid. At inference
the real spiking IF neuron runs for T discrete timesteps.

Sources:
  PseudoSNN/neuron/neuron.py  (PseudoNeuron, CustomIFNode)
  PseudoSNN/models/module.py  (SeqToANNContainer, pad_and_add)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.neuron import IFNode
from spikingjelly.activation_based import surrogate

__all__ = [
    "pad_and_add",
    "SeqToANNContainer",
    "CustomIFNode",
    "PseudoNeuron",
]


# ---------------------------------------------------------------------------
# Helpers (from PseudoSNN/models/module.py)
# ---------------------------------------------------------------------------

def pad_and_add(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Add two [T1, B, ...] and [T2, B, ...] tensors, padding the shorter one
    along dim-0 via mirror-reflection so shapes match."""
    if tensor1.shape[0] == tensor2.shape[0]:
        return tensor1 + tensor2

    t_short, t_long = sorted([tensor1, tensor2], key=lambda t: t.shape[0])
    diff = t_long.shape[0] - t_short.shape[0]

    indices = torch.arange(
        t_short.shape[0] - 1,
        t_short.shape[0] - 1 - diff,
        -1,
        device=t_short.device,
    ) % t_short.shape[0]
    indices = indices.clamp(0, t_short.shape[0] - 1)
    t_padded = torch.cat([t_short, t_short[indices]], dim=0)
    return t_padded + t_long


class SeqToANNContainer(nn.Sequential):
    """Wrap standard ANN layers so they accept [T, B, ...] input.

    Flattens the T and B dimensions, applies each sub-module in sequence,
    then reshapes back to [T, B, ...].
    """

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T, B = x_seq.shape[:2]
        y = x_seq.flatten(0, 1)  # [T*B, ...]
        for module in self:
            y = module(y)
        return y.view(T, B, *y.shape[1:])


# ---------------------------------------------------------------------------
# Spiking IF neuron (inference-only)
# ---------------------------------------------------------------------------

class CustomIFNode(IFNode):
    """IF neuron with soft reset and a +0.5 fire offset.

    Inherits from spikingjelly.activation_based.neuron.IFNode.
    Used only during inference inside PseudoNeuron._inference_forward().
    step_mode='s' so callers can step through timesteps one-by-one.
    """

    def __init__(self, **kwargs):
        super().__init__(
            v_threshold=1.0,
            v_reset=None,             # soft reset: v -= v_threshold * spike
            surrogate_function=surrogate.Sigmoid(),
            step_mode="s",
            **kwargs,
        )

    def neuronal_fire(self) -> torch.Tensor:
        # Offset by 0.5 so the neuron fires slightly before v reaches 1.0,
        # consistent with the original PseudoSNN paper implementation.
        return self.surrogate_function(self.v + 0.5 - self.v_threshold)


# ---------------------------------------------------------------------------
# PseudoNeuron
# ---------------------------------------------------------------------------

class PseudoNeuron(nn.Module):
    """Proxy spiking neuron that avoids BPTT.

    Training path: mean across T → optional stochastic noise → ReLU.
    Inference path: stateful CustomIFNode unrolled for T timesteps.

    T is a learnable parameter (logit) optimised jointly with the network
    weights via the STE trick.
    """

    def __init__(
        self,
        noise_type: str = "uniform",
        noise_prob: float = 0.5,
        init_T: float = 8.0,
        min_T: int = 1,
        max_T: int = 16,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.noise_prob = noise_prob
        self.min_T = min_T
        self.max_T = max_T

        self.spiking_function = CustomIFNode()

        ratio = (init_T - min_T) / (max_T - min_T)
        assert 0.0 < ratio < 1.0, f"init_T={init_T} must be strictly inside [{min_T}, {max_T}]"
        self.logit = nn.Parameter(torch.logit(torch.tensor(float(ratio))))
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(0.0))

        if noise_type == "uniform":
            self._base_noise = lambda x: torch.rand_like(x) - 0.5
        elif noise_type == "gaussian":
            self._base_noise = lambda x: torch.randn_like(x) / 3.0
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

    def get_timesteps(self) -> torch.Tensor:
        """Differentiable T via STE-rounded sigmoid."""
        T = self.min_T + (self.max_T - self.min_T) * torch.sigmoid(self.logit)
        return torch.round(T) + (T - T.detach())

    def _apply_quantization_noise(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.noise_prob:
            mask = (x > 0) & (x < 1.0)
            x = x + self._base_noise(x) * mask * (1.0 / T)
        return x

    def _apply_clipping_noise(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.noise_prob:
            x = x.clamp(max=1.0)
        return x

    def _inference_forward(self, x_seq: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        T_int = int(T.item())
        if x_seq.shape[0] == T_int:
            input_seq = x_seq
        else:
            x_mean = x_seq.mean(dim=0)
            input_seq = x_mean.unsqueeze(0).expand(T_int, *x_mean.shape)

        self.spiking_function.reset()  # clear membrane state before unrolling
        spikes = [self.spiking_function(input_seq[i]) for i in range(T_int)]
        result = torch.stack(spikes, dim=0)
        self.spiking_function.reset()  # free self.v GPU tensor — prevents VRAM leak across train/val
        return result

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = self.get_timesteps()
        x_seq = x_seq * self.scale + self.bias

        if self.training:
            x_seq = x_seq.mean(dim=0, keepdim=True)
            x_seq = self._apply_quantization_noise(x_seq, T)
            x_seq = self._apply_clipping_noise(x_seq)
            return F.relu(x_seq)
        else:
            return self._inference_forward(x_seq, T)
