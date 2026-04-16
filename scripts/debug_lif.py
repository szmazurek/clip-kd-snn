"""Minimal LIF compilation debugger.

Creates a single LIFNode, runs it on dummy input, and optionally compiles it.
Use this to isolate and iterate on torch.compile graph breaks.

Usage:
    # Baseline — no compile, just time the raw forward
    python scripts/debug_lif.py

    # Show all graph breaks (allow them, compile what can be compiled)
    python scripts/debug_lif.py --compile

    # Fail immediately on first graph break (shows exact location)
    python scripts/debug_lif.py --compile --fullgraph

    # Static explanation without running (shows all break reasons upfront)
    python scripts/debug_lif.py --explain
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# os.environ.setdefault("TORCH_LOGS", "+dynamo,graph_breaks,recompiles")

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

# ---- Config ----
T = 4  # SNN timesteps
B = 2  # batch size
N = 512  # feature dim


class SingleLIFBlock(nn.Module):
    """One Linear → LIF → Linear block in multi-step mode."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N, N)
        self.lif = neuron.LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, v_threshold=1.0, backend="torch"
        )
        self.fc2 = nn.Linear(N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N]
        x = self.fc1(x)
        x = self.lif(x)
        x = self.fc2(x)
        return x


def make_input(device):
    return torch.randn(T, B, N, device=device)


def run(model, x):
    functional.reset_net(model)
    return model(x)


def benchmark(fn, n_warmup=5, n_iter=20, label=""):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  {label:<30s}  {ms:.2f} ms/iter")
    return ms


def inplace_reset_net(net: nn.Module):
    """Resets SNN memory states in-place to preserve memory pointers for CUDA Graphs."""
    # Handle compiled models by accessing the original module
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod

    for m in net.modules():
        if hasattr(m, "v"):
            if isinstance(m.v, torch.Tensor):
                # Zero out the tensor IN-PLACE. The memory address stays the same!
                m.v.detach().zero_()
            else:
                # Fallback for the very first step before it becomes a tensor
                m.v = 0.0

        # If you are using store_v_seq = True anywhere, you need to handle that too
        if hasattr(m, "v_seq") and isinstance(m.v_seq, torch.Tensor):
            m.v_seq.detach().zero_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the block"
    )
    parser.add_argument(
        "--fullgraph", action="store_true", help="require fullgraph (fail on breaks)"
    )
    parser.add_argument(
        "--explain", action="store_true", help="run torch._dynamo.explain() and exit"
    )
    parser.add_argument(
        "--mode",
        default="reduce-overhead",
        help="torch.compile mode (default: reduce-overhead)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SingleLIFBlock().to(device).train()
    x = make_input(device)

    # --- Static explain ---
    if args.explain:
        print("\n=== torch._dynamo.explain() ===")
        functional.reset_net(model)
        exp = torch._dynamo.explain(model)(x)
        print(exp)
        print(f"\nGraph count  : {exp.graph_count}")
        print(f"Break reasons: {len(exp.break_reasons)}")
        for i, br in enumerate(exp.break_reasons):
            print(f"  [{i+1}] {br.reason}")
            if br.user_stack:
                print(f"       {br.user_stack[-1]}")
        return

    # --- Raw baseline ---
    print(
        f"\ndevice={device}  T={T}  B={B}  N={N}  compile={args.compile}  "
        f"fullgraph={args.fullgraph}  mode={args.mode!r}\n"
    )
    torch.cuda.nvtx.range_push("raw (no compile)")
    raw_ms = benchmark(lambda: run(model, x), label="raw (no compile)")
    torch.cuda.nvtx.range_pop()
    if not args.compile:
        return

    # --- Compile ---
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=args.fullgraph, mode=args.mode)

    print("\n  First call triggers compilation + logs graph breaks...\n")
    functional.reset_net(model)
    compiled(x)  # triggers dynamo tracing
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"compiled ({args.mode})")
    compiled_ms = benchmark(
        lambda: (inplace_reset_net(model), compiled(x)),
        label=f"compiled ({args.mode})",
    )
    torch.cuda.nvtx.range_pop()
    print(f"\nSpeedup: {raw_ms / compiled_ms:.2f}x")

    # Test if next forward passes cause recompiles (they shouldn't if the first compile succeeded)
    print("\n  Subsequent calls should NOT trigger recompiles...\n")
    for i in range(3):
        print(f"  Call #{i+2}...")
        inplace_reset_net(model)
        ms = benchmark(lambda: compiled(x), label=f"compiled ({args.mode})")
        print(f"    New speedup: {raw_ms / ms:.2f}x")
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
