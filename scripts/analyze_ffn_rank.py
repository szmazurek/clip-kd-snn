#!/usr/bin/env python3
"""
Compute and plot the 95% energy rank of FFN weight matrices across transformer blocks.

The energy rank at threshold p is the smallest r such that:
    sum(sv[:r]) / sum(sv) >= p
where sv are the singular values of the weight matrix in descending order.

Usage – ViT / noloop LoopViT (12 distinct blocks, 1 step):
    python scripts/analyze_ffn_rank.py \\
        --checkpoint outputs/16257472_baseline_loopvit_vitB-16/checkpoints/best-epoch=031-top1=0.3840.ckpt \\
        --label "LoopViT-noloop (ViT-B/16)" \\
        --mode vit --out ffn_rank_vit.png

Usage – global LoopViT (1 shared block × N steps):
    python scripts/analyze_ffn_rank.py \\
        --checkpoint outputs/16323316_kd_loopvit_vitb/checkpoints/last.ckpt \\
        --label "LoopViT-global (1×12)" \\
        --mode loopvit_global --max_loop_steps 12 --out ffn_rank_loopvit_global.png

Usage – per-block LoopViT (loop_core_depth blocks × max_loop_steps iterations):
    python scripts/analyze_ffn_rank.py \\
        --checkpoint outputs/.../last.ckpt \\
        --label "LoopViT-perblock (2×12)" \\
        --mode loopvit_perblock --max_loop_steps 12 --loop_core_depth 2 --out ffn_rank_perblock.png

Multiple --checkpoint / --label pairs overlay on the same figure.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_and_strip(path: str) -> dict:
    """Load a Lightning (or raw) checkpoint and strip all module-wrapper prefixes."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd: dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Strip Lightning student./teacher. prefix
    for prefix in ("student.", "teacher."):
        cand = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if cand:
            sd = cand
            break
    # Strip DDP wrapper
    if sd and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    # Strip torch.compile wrapper
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_arch(sd: dict) -> str:
    """
    Detect FFN architecture from state dict keys.

    Returns one of:
      'openclip'      open_clip ViT  (transformer.resblocks / c_fc / c_proj)
      'custom_gelu'   LoopViT/noloop (encoder.blocks / w_in / w_out)
      'custom_swiglu' LoopViT/noloop (encoder.blocks / w1 / w2 / w_out)
      'perblock_gelu' per-block LoopViT (visual.blocks.i.blocks.0 / w_in / w_out)
      'perblock_swiglu'
    """
    if "model.visual.blocks.0.blocks.0.mlp.w_in.weight" in sd:
        return "perblock_gelu"
    if "model.visual.blocks.0.blocks.0.mlp.w1.weight" in sd:
        return "perblock_swiglu"
    if "model.visual.encoder.blocks.0.mlp.w_in.weight" in sd:
        return "custom_gelu"
    if "model.visual.encoder.blocks.0.mlp.w1.weight" in sd:
        return "custom_swiglu"
    if "model.visual.transformer.resblocks.0.mlp.c_fc.weight" in sd:
        return "openclip"
    raise ValueError(
        "Cannot detect architecture from state dict keys. "
        "Expected one of: open_clip resblocks, custom encoder.blocks, or per-block visual.blocks."
    )


# ---------------------------------------------------------------------------
# Energy rank
# ---------------------------------------------------------------------------

def energy_rank(w: torch.Tensor, threshold: float) -> int:
    sv = torch.linalg.svdvals(w.detach().float())
    cumsum = torch.cumsum(sv, dim=0)
    return int((cumsum / sv.sum() < threshold).sum().item()) + 1


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------

def _max_index(sd: dict, pattern: str) -> int:
    """Return max integer captured by the first group in pattern across all keys."""
    indices = {int(m.group(1)) for k in sd if (m := re.search(pattern, k))}
    if not indices:
        raise ValueError(f"No keys matched pattern {pattern!r}")
    return max(indices)


def _get_in_out_custom(sd: dict, prefix: str, swiglu: bool):
    """
    Return (w_in_tensor, w_out_tensor) for a custom GELU/SwiGLU block at key prefix.
    For SwiGLU concatenates w1 and w2 along dim=0 as the combined in-projection.
    """
    if swiglu:
        w1 = sd[f"{prefix}.mlp.w1.weight"]
        w2 = sd[f"{prefix}.mlp.w2.weight"]
        w_in = torch.cat([w1, w2], dim=0)
    else:
        w_in = sd[f"{prefix}.mlp.w_in.weight"]
    w_out = sd[f"{prefix}.mlp.w_out.weight"]
    return w_in, w_out


def extract_ranks_openclip(sd: dict, threshold: float):
    n = _max_index(sd, r"transformer\.resblocks\.(\d+)\.") + 1
    in_ranks, out_ranks = [], []
    for i in range(n):
        base = f"model.visual.transformer.resblocks.{i}"
        in_ranks.append(energy_rank(sd[f"{base}.mlp.c_fc.weight"], threshold))
        out_ranks.append(energy_rank(sd[f"{base}.mlp.c_proj.weight"], threshold))
    return list(range(n)), in_ranks, out_ranks


def extract_ranks_custom_encoder(sd: dict, threshold: float, swiglu: bool):
    n = _max_index(sd, r"visual\.encoder\.blocks\.(\d+)\.") + 1
    in_ranks, out_ranks = [], []
    for i in range(n):
        w_in, w_out = _get_in_out_custom(sd, f"model.visual.encoder.blocks.{i}", swiglu)
        in_ranks.append(energy_rank(w_in, threshold))
        out_ranks.append(energy_rank(w_out, threshold))
    return list(range(n)), in_ranks, out_ranks


def extract_ranks_global_loopvit(sd: dict, threshold: float, max_loop_steps: int, swiglu: bool):
    """
    Global LoopViT: K blocks in core, each applied at every loop step (shared weights).
    Unrolls to max_loop_steps * K x-positions so the plot x-axis = loop step.
    For the common K=1 case this produces a flat horizontal line.
    """
    xs, per_block_in, per_block_out = extract_ranks_custom_encoder(sd, threshold, swiglu)
    k = len(xs)
    step_indices = list(range(max_loop_steps * k))
    in_ranks = [per_block_in[i % k] for i in range(max_loop_steps * k)]
    out_ranks = [per_block_out[i % k] for i in range(max_loop_steps * k)]
    return step_indices, in_ranks, out_ranks


def extract_ranks_perblock(
    sd: dict, threshold: float, max_loop_steps: int, loop_core_depth: int, swiglu: bool
):
    """
    Per-block LoopViT with constant schedule.
    Each outer block has loop_core_depth inner blocks, all repeated max_loop_steps times.
    Returns x-positions over the entire unrolled sequence, plus outer-block boundaries.
    """
    n = _max_index(sd, r"visual\.blocks\.(\d+)\.") + 1
    step_indices, in_ranks, out_ranks, boundaries = [], [], [], []
    step = 0
    for i in range(n):
        boundaries.append((step, i))
        inner_ranks = []
        for j in range(loop_core_depth):
            w_in, w_out = _get_in_out_custom(
                sd, f"model.visual.blocks.{i}.blocks.{j}", swiglu
            )
            inner_ranks.append((energy_rank(w_in, threshold), energy_rank(w_out, threshold)))
        for _ in range(max_loop_steps):
            for j in range(loop_core_depth):
                step_indices.append(step)
                in_ranks.append(inner_ranks[j][0])
                out_ranks.append(inner_ranks[j][1])
                step += 1
    return step_indices, in_ranks, out_ranks, boundaries


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_LINE_KW = dict(linewidth=1.8, markersize=5)


def _add_vit_lines(ax, xs, in_ranks, out_ranks, label, color):
    ax.plot(xs, in_ranks, "o-", label=f"{label} – in-proj", color=color, **_LINE_KW)
    ax.plot(xs, out_ranks, "s--", label=f"{label} – out-proj", color=color,
            alpha=0.75, **_LINE_KW)


def _add_global_lines(ax, xs, in_ranks, out_ranks, label, color):
    ax.plot(xs, in_ranks, "o-", label=f"{label} – in-proj", color=color,
            markersize=4, linewidth=1.8)
    ax.plot(xs, out_ranks, "s--", label=f"{label} – out-proj", color=color,
            markersize=4, linewidth=1.8, alpha=0.75)


def _add_perblock_lines(ax, xs, in_ranks, out_ranks, boundaries, label, threshold):
    palette = plt.cm.tab10.colors
    total = max(xs) + 1

    region_patches = []
    for j, (start, block_idx) in enumerate(boundaries):
        end = boundaries[j + 1][0] if j + 1 < len(boundaries) else total
        color = palette[block_idx % 10]
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.12, color=color, zorder=0)
        if j > 0:
            ax.axvline(start - 0.5, color="gray", linestyle=":", linewidth=0.8, zorder=1)
        region_patches.append(
            Patch(facecolor=color, alpha=0.4,
                  label=f"Block {block_idx}  (steps {start}–{end - 1})")
        )

    ax.plot(xs, in_ranks, "o-", label=f"{label} – in-proj",
            color="steelblue", markersize=4, linewidth=1.8, zorder=2)
    ax.plot(xs, out_ranks, "s--", label=f"{label} – out-proj",
            color="tomato", markersize=4, linewidth=1.8, alpha=0.85, zorder=2)

    line_legend = ax.legend(loc="upper left")
    ax.add_artist(line_legend)
    ax.legend(handles=region_patches, loc="upper right", fontsize="small")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FFN 95%% energy rank analysis for ViT and LoopViT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--label", nargs="+", default=None, metavar="NAME")
    parser.add_argument(
        "--mode",
        choices=["vit", "loopvit_global", "loopvit_perblock"],
        required=True,
    )
    parser.add_argument(
        "--max_loop_steps",
        type=int,
        default=None,
        help="Number of loop iterations; required when mode=loopvit_global or loopvit_perblock",
    )
    parser.add_argument(
        "--loop_core_depth",
        type=int,
        default=1,
        help="Number of inner blocks per loop cell (loopvit_perblock only, default: 1)",
    )
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Energy fraction threshold (default: 0.95)")
    parser.add_argument("--out", type=str, default="ffn_rank.png",
                        help="Output figure path (default: ffn_rank.png)")
    args = parser.parse_args()

    # Validate mode-specific args
    if args.mode in ("loopvit_global", "loopvit_perblock") and args.max_loop_steps is None:
        parser.error("--max_loop_steps is required when --mode loopvit_global or loopvit_perblock")

    checkpoints = args.checkpoint
    labels = list(args.label) if args.label else []
    while len(labels) < len(checkpoints):
        labels.append(Path(checkpoints[len(labels)]).parent.parent.name)

    pct = int(args.threshold * 100)
    palette = plt.cm.tab10.colors

    is_perblock = args.mode == "loopvit_perblock"
    fig, ax = plt.subplots(figsize=(12 if is_perblock else 10, 4))

    for idx, (ckpt_path, label) in enumerate(zip(checkpoints, labels)):
        print(f"[{idx + 1}/{len(checkpoints)}] Loading {ckpt_path}")
        sd = load_and_strip(ckpt_path)
        arch = detect_arch(sd)
        swiglu = "swiglu" in arch
        print(f"  architecture: {arch}")

        if args.mode == "vit":
            if arch == "openclip":
                xs, ir, or_ = extract_ranks_openclip(sd, args.threshold)
            else:
                xs, ir, or_ = extract_ranks_custom_encoder(sd, args.threshold, swiglu)
            _add_vit_lines(ax, xs, ir, or_, label, palette[idx % 10])

        elif args.mode == "loopvit_global":
            xs, ir, or_ = extract_ranks_global_loopvit(
                sd, args.threshold, args.max_loop_steps, swiglu
            )
            _add_global_lines(ax, xs, ir, or_, label, palette[idx % 10])

        elif args.mode == "loopvit_perblock":
            xs, ir, or_, bounds = extract_ranks_perblock(
                sd, args.threshold, args.max_loop_steps, args.loop_core_depth, swiglu
            )
            _add_perblock_lines(ax, xs, ir, or_, bounds, label, args.threshold)

    ax.set_ylabel(f"{pct}% energy rank")
    if args.mode == "vit":
        xlabel = "Block index"
    elif args.mode == "loopvit_perblock" and args.loop_core_depth > 1:
        xlabel = "Block application (step × core depth)"
    else:
        xlabel = "Loop step"
    ax.set_xlabel(xlabel)
    title_models = " vs ".join(labels) if len(labels) > 1 else labels[0]
    ax.set_title(f"FFN {pct}% energy rank – {title_models}")
    ax.grid(True, alpha=0.3)

    if not is_perblock:
        ax.legend()

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
