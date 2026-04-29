"""Benchmark forward+backward time for LIF, PSN, MaskedPSN and SlidingPSN neurons.

Measures wall-clock time using CUDA events for GPU accuracy, with an optional
torch.compile pass for each neuron type to show compilation benefit.

Input shape: [T, B, F]  (timesteps × batch × features)

Usage examples:
    # Default: all neurons, T=4, B=64, F=512, 200 iters
    python scripts/benchmark_neurons.py

    # PSN variants only, larger batch, with compile
    python scripts/benchmark_neurons.py --neuron-types psn masked_psn sliding_psn \\
        --T 4 --batch-size 256 --features 512 --iters 500

    # LIF only, no compile baseline
    python scripts/benchmark_neurons.py --neuron-types lif --no-compile

    # CPU run (no CUDA events — uses time.perf_counter instead)
    python scripts/benchmark_neurons.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Neuron builders
# ---------------------------------------------------------------------------

_ALL_TYPES = ["lif", "psn", "masked_psn", "sliding_psn"]


def build_neuron(
    neuron_type: str,
    T: int,
    k: int = 2,
    psn_backend: str = "gemm",
) -> nn.Module:
    if neuron_type == "lif":
        from src.models.visual_encoders.lif_node import LIFNode
        return LIFNode(tau=2.0, v_threshold=1.0)

    from spikingjelly.activation_based.neuron.psn import PSN, MaskedPSN, SlidingPSN

    if neuron_type == "psn":
        return PSN(T=T)
    if neuron_type == "masked_psn":
        return MaskedPSN(k=k, T=T, step_mode="m")
    if neuron_type == "sliding_psn":
        return SlidingPSN(k=k, step_mode="m", backend=psn_backend)

    raise ValueError(f"Unknown neuron type: {neuron_type!r}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _bench_cuda(
    neuron: nn.Module,
    x_shape: tuple[int, ...],
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) using CUDA events."""
    x = torch.rand(x_shape, device=device, requires_grad=True)
    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Warmup
    for _ in range(warmup):
        out = neuron(x)
        out.sum().backward()
        x.grad = None

    torch.cuda.synchronize()

    for i in range(iters):
        x = torch.rand(x_shape, device=device, requires_grad=True)
        start_evt[i].record()
        out = neuron(x)
        out.sum().backward()
        end_evt[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def _bench_cpu(
    neuron: nn.Module,
    x_shape: tuple[int, ...],
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) using perf_counter."""
    x = torch.rand(x_shape, device=device, requires_grad=True)

    for _ in range(warmup):
        out = neuron(x)
        out.sum().backward()
        x.grad = None

    times = []
    for _ in range(iters):
        x = torch.rand(x_shape, device=device, requires_grad=True)
        t0 = time.perf_counter()
        out = neuron(x)
        out.sum().backward()
        times.append((time.perf_counter() - t0) * 1e3)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def benchmark_one(
    neuron: nn.Module,
    x_shape: tuple[int, ...],
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, float]:
    neuron = neuron.to(device).train()
    if device.type == "cuda":
        return _bench_cuda(neuron, x_shape, iters, warmup, device)
    return _bench_cpu(neuron, x_shape, iters, warmup, device)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    return f"{mean:7.3f} ± {std:5.3f}"


def print_table(rows: list[tuple]) -> None:
    """rows: list of (label, plain_mean, plain_std, compiled_mean|None, compiled_std|None)"""
    w_label = max(len(r[0]) for r in rows) + 2
    header = f"{'Neuron':<{w_label}}  {'Plain (ms)':>20}  {'Compiled (ms)':>20}  {'Speedup':>8}"
    print()
    print(header)
    print("-" * len(header))
    for label, pm, ps, cm, cs in rows:
        plain_s = _fmt(pm, ps)
        if cm is not None:
            comp_s = _fmt(cm, cs)
            speedup = f"{pm / cm:.2f}x" if cm > 0 else "  n/a"
        else:
            comp_s = "      (skipped)"
            speedup = "  n/a"
        print(f"{label:<{w_label}}  {plain_s:>20}  {comp_s:>20}  {speedup:>8}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SNN neurons: fwd+bwd time")
    p.add_argument("--T", type=int, default=4, help="SNN timesteps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--features", type=int, default=512, help="Feature dimension F")
    p.add_argument(
        "--neuron-types",
        nargs="+",
        default=_ALL_TYPES,
        choices=_ALL_TYPES,
        metavar="TYPE",
        help=f"Neuron types to benchmark. Choices: {_ALL_TYPES}",
    )
    p.add_argument("--psn-k", type=int, default=2, help="Order k for masked/sliding PSN")
    p.add_argument(
        "--psn-backend", default="gemm", choices=["gemm", "conv"],
        help="Multi-step backend for SlidingPSN",
    )
    p.add_argument("--iters", type=int, default=200, help="Timed iterations")
    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    p.add_argument(
        "--no-compile", action="store_true",
        help="Skip torch.compile variants",
    )
    p.add_argument(
        "--compile-mode", default="default",
        help="torch.compile mode (default | reduce-overhead | max-autotune)",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    x_shape = (args.T, args.batch_size, args.features)

    print(f"Device : {device}")
    print(f"Shape  : {list(x_shape)}  [T, B, F]")
    print(f"Iters  : {args.iters}  (warmup {args.warmup})")
    print(f"Compile: {'disabled' if args.no_compile else args.compile_mode}")
    print(f"PSN k  : {args.psn_k}  backend={args.psn_backend}")

    rows: list[tuple] = []

    for nt in args.neuron_types:
        label = nt
        try:
            neuron = build_neuron(nt, T=args.T, k=args.psn_k, psn_backend=args.psn_backend)
        except Exception as e:
            print(f"[{nt}] build failed: {e}")
            continue

        print(f"\nBenchmarking {nt} (plain)…", end=" ", flush=True)
        try:
            pm, ps = benchmark_one(neuron, x_shape, args.iters, args.warmup, device)
            print(f"{pm:.3f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append((label, float("nan"), 0.0, None, None))
            continue

        cm, cs = None, None
        if not args.no_compile and device.type == "cuda":
            print(f"Benchmarking {nt} (compiled, mode={args.compile_mode})…", end=" ", flush=True)
            try:
                compiled = torch.compile(
                    build_neuron(nt, T=args.T, k=args.psn_k, psn_backend=args.psn_backend),
                    mode=args.compile_mode,
                )
                cm, cs = benchmark_one(compiled, x_shape, args.iters, args.warmup, device)
                print(f"{cm:.3f} ms")
            except Exception as e:
                print(f"FAILED: {e}")

        rows.append((label, pm, ps, cm, cs))

    print_table(rows)


if __name__ == "__main__":
    main()
