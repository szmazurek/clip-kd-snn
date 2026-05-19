"""Benchmark + Perfetto-annotated profiling for QKFormer building blocks.

Two modes:
  1. Latency benchmark   — torch.utils.benchmark per module (default)
  2. Perfetto profiling  — torch.profiler traces per isolated block (--profile)

Usage:
    # latency only
    python benchmark_qformer_compile.py

    # write Perfetto traces to ./profiler_traces/blocks/
    python benchmark_qformer_compile.py --profile

    # specific blocks only
    python benchmark_qformer_compile.py --profile --blocks psn ssa

    # view traces
    tensorboard --logdir ./profiler_traces/blocks
    # or drag any .json.gz into https://ui.perfetto.dev

    # Nsight Systems (record_function also bridges to NVTX automatically)
    nsys profile --trace cuda,nvtx python benchmark_qformer_compile.py --profile
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ["TORCH_LOGS"] = "+recompiles"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity

torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
torch._dynamo.config.assume_static_by_default = True

from src.models.visual_encoders.qkformer import (
    PatchEmbedInit,
    Token_QK_Attention,
    Spiking_Self_Attention,
    SNNParams,
)
from src.models.visual_encoders.psn_node import (
    PSNAdapter,
    CompileFriendlyPSN,
    CompileFriendlyMaskedPSN,
    CompileFriendlySlidingPSN,
)


# ---------------------------------------------------------------------------
# Isolated building blocks for targeted profiling
# ---------------------------------------------------------------------------

class Conv1dBnPSN(nn.Module):
    """Conv1d(k=1) → BN1d → PSNAdapter — mirrors one attention projection."""

    def __init__(self, dim: int, T: int, psn_inner: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm1d(dim)
        self.lif  = PSNAdapter(psn_inner, T=T)
        self.T    = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T*B, dim, N]
        T = self.T
        B_T, C, N = x.shape
        B = B_T // T
        with torch.profiler.record_function("c1b_psn/conv"):
            out = self.conv(x)
        with torch.profiler.record_function("c1b_psn/bn"):
            out = self.bn(out).reshape(T, B, C, N)
        with torch.profiler.record_function("c1b_psn/lif"):
            out = self.lif(out)
        return out


# ---------------------------------------------------------------------------
# Latency benchmark helper
# ---------------------------------------------------------------------------

def benchmark_module(module: nn.Module, x: torch.Tensor, name: str) -> None:
    print(f"\n{'='*52}\n  {name}\n{'='*52}")
    torch._dynamo.reset()
    compiled = torch.compile(module)

    print("  Compiling ...", end="", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    out = compiled(x); out.sum().backward()
    torch.cuda.synchronize()
    print(f" done in {time.time() - t0:.1f}s")

    def _fwd():
        return compiled(x)

    for _ in range(5):
        _fwd()
    torch.cuda.synchronize()

    timer = benchmark.Timer(
        stmt="_fwd()",
        globals={"_fwd": _fwd},
        num_threads=1,
        label=name,
        sub_label="forward",
    )
    print(timer.blocked_autorange(min_run_time=2.0))

    torch.cuda.reset_peak_memory_stats()
    _fwd()
    torch.cuda.synchronize()
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


# ---------------------------------------------------------------------------
# Perfetto profiling helper
# ---------------------------------------------------------------------------

def profile_block(
    name: str,
    block: nn.Module,
    x: torch.Tensor,
    trace_dir: str,
    compile_warmup: int = 5,
    wait: int = 0,
    warmup: int = 1,
    active: int = 5,
) -> None:
    """Compile block, burn warmup steps, then write a Perfetto-compatible trace."""
    print(f"\n{'='*52}\n  profiling: {name}\n{'='*52}")
    print(f"  input: {tuple(x.shape)}  dtype={x.dtype}")

    torch._dynamo.reset()
    compiled = torch.compile(block, mode="reduce-overhead")

    print("  Compiling ...", end="", flush=True)
    torch.cuda.synchronize()
    for _ in range(compile_warmup):
        block.zero_grad()
        if x.grad is not None:
            x.grad = None
        out = compiled(x)
        out.sum().backward()
    torch.cuda.synchronize()
    print(" done")

    os.makedirs(trace_dir, exist_ok=True)
    total = wait + warmup + active

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(total):
            block.zero_grad()
            if x.grad is not None:
                x.grad = None

            torch.cuda.nvtx.range_push(f"{name}/forward")
            with torch.profiler.record_function(f"{name}/forward"):
                out = compiled(x)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"{name}/backward")
            with torch.profiler.record_function(f"{name}/backward"):
                out.sum().backward()
            torch.cuda.nvtx.range_pop()

            prof.step()

    print(f"  trace → {trace_dir}")


# ---------------------------------------------------------------------------
# Block catalogue
# ---------------------------------------------------------------------------

def _make_psn_inner(psn_impl: str, T: int) -> nn.Module:
    """Instantiate the raw PSN module (before PSNAdapter wrapping)."""
    if psn_impl == "cf":
        return CompileFriendlyPSN(T=T)
    # sj — uses torch.autograd.Function → graph breaks under torch.compile
    from spikingjelly.activation_based.neuron.psn import PSN as _SJ_PSN
    return _SJ_PSN(T=T)


def build_blocks(
    snn: SNNParams,
    T: int,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
    psn_impl: str = "cf",
):
    """Returns dict of {name: (module, input_tensor)}.

    Args:
        psn_impl: 'cf' for CompileFriendlyPSN, 'sj' for native SpikingJelly PSN.
                  Affects psn_raw, psn_adapter, and conv1d_bn_psn_* blocks.
                  Attention blocks (token_qk_attn, spiking_self_attn) pick up the
                  implementation via snn.neuron_type set in parse_args/main.
    """
    blocks = {}

    # --- PSN raw (no conv/bn wrapper) ---
    psn_raw = _make_psn_inner(psn_impl, T).to(device=device, dtype=dtype).train()
    x_psn2 = torch.randn(T, B * 512, device=device, dtype=dtype, requires_grad=True)
    blocks["psn_raw"] = (psn_raw, x_psn2)

    # --- PSNAdapter (adds reshape bookkeeping) ---
    psn_adapter = PSNAdapter(_make_psn_inner(psn_impl, T), T=T).to(device=device, dtype=dtype).train()
    x_adapter = torch.randn(T * B, 512, device=device, dtype=dtype, requires_grad=True)
    blocks["psn_adapter"] = (psn_adapter, x_adapter)

    # --- Conv1d(k=1) → BN1d → PSN, stage-1 dim ---
    c1bp_128 = Conv1dBnPSN(dim=128, T=T, psn_inner=_make_psn_inner(psn_impl, T)).to(device=device, dtype=dtype).train()
    x_c1bp_128 = torch.randn(T * B, 128, 56 * 56, device=device, dtype=dtype,
                              requires_grad=True)
    blocks["conv1d_bn_psn_dim128"] = (c1bp_128, x_c1bp_128)

    # --- Conv1d(k=1) → BN1d → PSN, stage-3 dim ---
    c1bp_512 = Conv1dBnPSN(dim=512, T=T, psn_inner=_make_psn_inner(psn_impl, T)).to(device=device, dtype=dtype).train()
    x_c1bp_512 = torch.randn(T * B, 512, 14 * 14, device=device, dtype=dtype,
                              requires_grad=True)
    blocks["conv1d_bn_psn_dim512"] = (c1bp_512, x_c1bp_512)

    # --- Token_QK_Attention, stage 1 ---
    tqk = Token_QK_Attention(dim=128, num_heads=8, snn=snn).to(device=device, dtype=dtype).train()
    x_tqk = torch.randn(T, B, 128, 56, 56, device=device, dtype=dtype, requires_grad=True)
    blocks["token_qk_attn"] = (tqk, x_tqk)

    # --- Spiking_Self_Attention, stage 3 ---
    ssa = Spiking_Self_Attention(dim=512, num_heads=8, snn=snn).to(device=device, dtype=dtype).train()
    x_ssa = torch.randn(T, B, 512, 14, 14, device=device, dtype=dtype, requires_grad=True)
    blocks["spiking_self_attn"] = (ssa, x_ssa)

    # --- PatchEmbedInit ---
    pei = PatchEmbedInit(img_size_h=224, img_size_w=224, in_channels=3,
                         embed_dims=128, snn=snn).to(device=device, dtype=dtype).train()
    x_pei = torch.randn(T, B, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)
    blocks["patch_embed_init"] = (pei, x_pei)

    return blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true",
                   help="Write Perfetto traces instead of latency benchmark")
    p.add_argument("--blocks", nargs="+", default=None,
                   help="Subset of block names to run (default: all)")
    p.add_argument("--T",          type=int,   default=4)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--psn-impl", default="cf", choices=["cf", "sj"],
                   help="PSN implementation: 'cf' = CompileFriendlyPSN (default), "
                        "'sj' = native SpikingJelly PSN (causes graph breaks under compile)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--dtype",      default="bf16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--trace-dir",  default="./profiler_traces/blocks")
    p.add_argument("--compile-warmup", type=int, default=5)
    p.add_argument("--profile-wait",   type=int, default=0)
    p.add_argument("--profile-warmup", type=int, default=1)
    p.add_argument("--profile-active", type=int, default=5)
    return p.parse_args()


# Maps --psn-impl to the neuron_type string understood by _build_lif_node
_PSN_IMPL_TO_NEURON_TYPE = {"cf": "psn", "sj": "sj_psn"}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype  = {"fp32": torch.float32, "fp16": torch.float16,
              "bf16": torch.bfloat16}[args.dtype]

    neuron_type = _PSN_IMPL_TO_NEURON_TYPE[args.psn_impl]
    snn = SNNParams(neuron_type=neuron_type, T=args.T)

    print(f"device={device}  dtype={dtype}  T={args.T}  B={args.batch_size}  "
          f"psn_impl={args.psn_impl}  neuron_type={neuron_type}")

    blocks = build_blocks(snn, T=args.T, B=args.batch_size,
                          device=device, dtype=dtype, psn_impl=args.psn_impl)

    selected = args.blocks or list(blocks)
    unknown  = set(selected) - set(blocks)
    if unknown:
        print(f"Unknown block(s): {unknown}\nAvailable: {list(blocks)}")
        sys.exit(1)

    for name in selected:
        module, x = blocks[name]
        if args.profile:
            # include impl suffix so cf and sj traces sit in separate dirs
            trace_dir = os.path.join(args.trace_dir, f"{name}_{args.psn_impl}")
            profile_block(
                name=f"{name}/{args.psn_impl}", block=module, x=x, trace_dir=trace_dir,
                compile_warmup=args.compile_warmup,
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
            )
        else:
            benchmark_module(module, x, name)

    if args.profile:
        print(f"\nAll traces → {args.trace_dir}")
        print(f"  tensorboard --logdir {args.trace_dir}")
        print("  or drag any .json.gz into https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
