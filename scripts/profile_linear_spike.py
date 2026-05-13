"""Torch-profiler traces for isolated Linear + PSN-spike blocks.

Profiles forward+backward of nn.Linear → BatchNorm1d → PSNAdapter(variant) for
each of our three compile-friendly PSN variants at two representative channel
widths (128 and 512), matching QKFormer stage 1 and stage 3 dimensions.

Every module is compiled with torch.compile before the profiler starts.
Separate TensorBoard traces are written per variant so you can compare kernels
side-by-side in the Profiler tab or in chrome://tracing / Perfetto.

Usage:
    # default: all variants, T=4, B=32, dim=128/512
    python scripts/profile_linear_spike.py

    # single variant, custom dims
    python scripts/profile_linear_spike.py \\
        --variants psn masked_psn \\
        --dims 512 \\
        --T 4 --batch-size 64

    # view traces
    tensorboard --logdir ./profiler_traces/linear_spike

    # for Nsight Systems: NVTX markers are emitted automatically around each
    # fwd and bwd call — run with:
    #   nsys profile --trace cuda,nvtx python scripts/profile_linear_spike.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity

from src.models.visual_encoders.psn_node import (
    PSNAdapter,
    CompileFriendlyPSN,
    CompileFriendlyMaskedPSN,
    CompileFriendlySlidingPSN,
)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.assume_static_by_default = True


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

class LinearSpikeBlock(nn.Module):
    """nn.Linear + BN + PSNAdapter — mirrors the Conv1d(k=1)+BN+LIF pattern in QKFormer.

    The Conv1d(kernel_size=1) in attention / MLP blocks is mathematically
    equivalent to this Linear when the spatial tokens are merged into the batch
    dimension, which is exactly what PSNAdapter does anyway.

    Input / output: [T*B, dim]
    """

    def __init__(self, dim: int, T: int, psn_inner: nn.Module) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.spike = PSNAdapter(psn_inner, T=T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spike(self.bn(self.linear(x)))


# ---------------------------------------------------------------------------
# Profiler harness
# ---------------------------------------------------------------------------

def _profile_one(
    name: str,
    block: nn.Module,
    x: torch.Tensor,
    trace_dir: str,
    compile_warmup: int,
    wait: int,
    warmup: int,
    active: int,
) -> None:
    """Compile block, burn through compile_warmup steps, then profile."""
    total_steps = wait + warmup + active

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  input  : {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    print(f"  params : {sum(p.numel() for p in block.parameters()):,}")
    print(f"  schedule: wait={wait} warmup={warmup} active={active} "
          f"({total_steps} total steps)")

    # Fresh compile per block so Dynamo sees isolated graphs
    torch._dynamo.reset()
    compiled = torch.compile(block)

    # --- Compilation warmup (outside profiler, outside NVTX) ---
    print(f"  Compiling … ", end="", flush=True)
    torch.cuda.synchronize()
    for _ in range(compile_warmup):
        block.zero_grad()
        x.grad = None
        out = compiled(x)
        out.sum().backward()
    torch.cuda.synchronize()
    print("done")

    # --- Profiled loop ---
    os.makedirs(trace_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(total_steps):
            block.zero_grad()
            x.grad = None

            # NVTX ranges for Nsight Systems; record_function for TensorBoard
            torch.cuda.nvtx.range_push(f"{name}/forward")
            with torch.profiler.record_function(f"{name}/forward"):
                out = compiled(x)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"{name}/backward")
            with torch.profiler.record_function(f"{name}/backward"):
                out.sum().backward()
            torch.cuda.nvtx.range_pop()

            prof.step()

    print(f"  trace  → {trace_dir}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _sj_psn(T: int, k: int) -> nn.Module:
    from spikingjelly.activation_based.neuron.psn import PSN
    return PSN(T=T)


def _sj_masked_psn(T: int, k: int) -> nn.Module:
    from spikingjelly.activation_based.neuron.psn import MaskedPSN
    return MaskedPSN(k=k, T=T, step_mode="m")


def _sj_sliding_psn(T: int, k: int) -> nn.Module:
    from spikingjelly.activation_based.neuron.psn import SlidingPSN
    return SlidingPSN(k=k, step_mode="m", backend="gemm")


# Spikingjelly variants are wrapped in PSNAdapter just like before — same
# [T*B, dim] → [T, B*dim] shape contract. The key difference is that sj PSN
# uses torch.autograd.Function for its surrogate, which creates graph breaks
# under torch.compile. This is exactly what the traces expose.
_VARIANT_BUILDERS = {
    "psn":            lambda T, k: CompileFriendlyPSN(T=T),
    "masked_psn":     lambda T, k: CompileFriendlyMaskedPSN(T=T, k=k),
    "sliding_psn":    lambda T, k: CompileFriendlySlidingPSN(T=T, k=k),
    "sj_psn":         _sj_psn,
    "sj_masked_psn":  _sj_masked_psn,
    "sj_sliding_psn": _sj_sliding_psn,
}

_ALL_VARIANTS = list(_VARIANT_BUILDERS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Linear+Spike blocks with torch.profiler")
    p.add_argument(
        "--variants",
        nargs="+",
        default=_ALL_VARIANTS,
        choices=_ALL_VARIANTS,
        metavar="V",
        help="PSN variants to profile (default: all six — compile-friendly + spikingjelly)",
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[128, 512],
        metavar="D",
        help="Channel widths to test (default: 128 512 — matching QKFormer stage 1 and 3)",
    )
    p.add_argument("--T", type=int, default=4, help="SNN timesteps")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size B (input is [T*B, dim])")
    p.add_argument("--psn-k", type=int, default=2, help="Bandwidth k for MaskedPSN / SlidingPSN")
    p.add_argument(
        "--profile-wait", type=int, default=0,
        help="Profiler wait steps (0 is fine — compilation already done before profiler starts)",
    )
    p.add_argument("--profile-warmup", type=int, default=1, help="Profiler warmup steps")
    p.add_argument("--profile-active", type=int, default=5, help="Profiler active (captured) steps")
    p.add_argument(
        "--compile-warmup", type=int, default=5,
        help="Forward+backward passes to run before the profiler to trigger Dynamo+Inductor",
    )
    p.add_argument(
        "--trace-dir",
        default="./profiler_traces/linear_spike",
        help="Root directory for TensorBoard traces (one sub-dir per variant×dim)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Input tensor dtype",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.precision]

    print(f"device    : {device}")
    print(f"dtype     : {dtype}")
    print(f"T={args.T}  B={args.batch_size}  T*B={args.T * args.batch_size}")
    print(f"variants  : {args.variants}")
    print(f"dims      : {args.dims}")
    print(f"psn_k     : {args.psn_k}")
    print(f"compile warmup : {args.compile_warmup} steps")
    print(f"profiler       : wait={args.profile_wait} warmup={args.profile_warmup} "
          f"active={args.profile_active}")

    for variant in args.variants:
        build_inner = _VARIANT_BUILDERS[variant]
        for dim in args.dims:
            name = f"{variant}_dim{dim}"
            x = torch.randn(
                args.T * args.batch_size, dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            block = LinearSpikeBlock(
                dim=dim,
                T=args.T,
                psn_inner=build_inner(args.T, args.psn_k),
            ).to(device=device, dtype=dtype).train()

            trace_dir = os.path.join(args.trace_dir, name)
            _profile_one(
                name=name,
                block=block,
                x=x,
                trace_dir=trace_dir,
                compile_warmup=args.compile_warmup,
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
            )

    print(f"\n{'='*60}")
    print(f"All traces written to: {args.trace_dir}")
    print(f"View with:")
    print(f"  tensorboard --logdir {args.trace_dir}")
    print(f"  # or open any .json.gz in chrome://tracing / Perfetto")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
