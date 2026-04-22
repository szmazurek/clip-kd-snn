import os
import time

os.environ["TORCH_LOGS"] = "+recompiles"

import torch
import torch.utils.benchmark as benchmark
from torch.fx.passes.graph_drawer import FxGraphDrawer

torch.set_num_threads(1)  # Lock this in before compiling!
# Import your classes here (assuming they are in the same file or imported)
# from qkformer import PatchEmbedInit, Token_QK_Attention, Spiking_Self_Attention, SNNParams
from src.models.visual_encoders.qkformer import (
    PatchEmbedInit,
    Token_QK_Attention,
    Spiking_Self_Attention,
    SNNParams,
)
from src.models.visual_encoders.lif_node import LIFNode

torch.set_float32_matmul_precision("high")


# ==========================================
# 1. Custom Backend Factory
# ==========================================
def make_visualizing_backend(module_name):
    """Creates a custom Inductor backend that saves the graph to TXT and PNG."""

    def custom_backend(gm: torch.fx.GraphModule, example_inputs):
        print(f"\n[Compiler] Intercepted FX Graph for {module_name}.")

        # 1. Save clean Python code
        txt_path = f"compiled_{module_name}_graph.txt"
        with open(txt_path, "w") as f:
            f.write(gm.code)
        print(f"  -> Saved text graph to {txt_path}")

        # Pass it back to standard Inductor
        return torch._inductor.compile(gm, example_inputs)

    return custom_backend


# ==========================================
# 2. Testing Harness
# ==========================================
def benchmark_module(module, input_data, name):
    print(f"\n{'='*50}\nTesting Module: {name}\n{'='*50}")

    # Force fresh compilation
    torch._dynamo.reset()

    # Attach backend and compile
    compiled_mod = torch.compile(
        module,
        # mode="reduce-overhead",
        # backend=make_visualizing_backend(name),
    )

    # --- Measure Compilation Time ---
    print(f"Triggering compilation (Triton Autotuning)...")
    torch.cuda.synchronize()
    t0 = time.time()

    out = compiled_mod(input_data)
    out.sum().backward()  # Trigger backward compilation too

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"✅ Compilation finished in {t1 - t0:.2f} seconds.")

    # --- Measure Latency ---
    print(f"Running latency benchmark...")

    def run_forward():
        return compiled_mod(input_data)

    # Warmup
    for _ in range(5):
        _ = run_forward()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="run_forward()",
        globals={"run_forward": run_forward},
        num_threads=1,
        label=name,
        sub_label="Forward Pass",
    )
    stats = timer.blocked_autorange(min_run_time=2.0)
    print(stats)

    # --- Measure Memory ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    _ = run_forward()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM Usage: {peak_mem:.2f} MB\n")


# ==========================================
# 3. Execution Execution
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snn_params = SNNParams()

    T = 4
    B = 32

    # ---------------------------------------------------------
    # Test 1: PatchEmbedInit
    # Input: Raw Images -> [T, B, C=3, H=224, W=224]
    # Output spatial drops to [56, 56]
    # ---------------------------------------------------------
    img_data = torch.randn(T, B, 3, 224, 224, requires_grad=True, device=device)
    patch_embed = PatchEmbedInit(
        img_size_h=224, img_size_w=224, in_channels=3, embed_dims=128, snn=snn_params
    ).to(device)
    benchmark_module(patch_embed, img_data, "PatchEmbedInit")

    # ---------------------------------------------------------
    # Test 2: Token_QK_Attention (Stage 1)
    # Input: After PatchEmbedInit -> [T, B, C=128, H=56, W=56]
    # ---------------------------------------------------------
    stage1_data = torch.randn(T, B, 128, 56, 56, requires_grad=True, device=device)
    tqk_attn = Token_QK_Attention(dim=128, num_heads=8, snn=snn_params).to(device)
    benchmark_module(tqk_attn, stage1_data, "Token_QK_Attention")

    # ---------------------------------------------------------
    # Test 3: Spiking_Self_Attention (Stage 3)
    # Input: Deep in the network -> [T, B, C=512, H=14, W=14]
    # ---------------------------------------------------------
    stage3_data = torch.randn(T, B, 512, 14, 14, requires_grad=True, device=device)
    ssa_attn = Spiking_Self_Attention(dim=512, num_heads=8, snn=snn_params).to(device)
    benchmark_module(ssa_attn, stage3_data, "Spiking_Self_Attention")

    # # trigger compile on backward
    out = patch_embed(img_data)
    out.sum().backward()
    out = tqk_attn(stage1_data)
    out.sum().backward()
    out = ssa_attn(stage3_data)
    out.sum().backward()
    # # NVTX marking and profiling to use nsight systems for detailed analysis
    from torch.cuda import nvtx

    nvtx.range_push("Patch embed forward")
    out = patch_embed(img_data)
    nvtx.range_pop()
    nvtx.range_push("Patch embed backward")
    out.sum().backward()
    nvtx.range_pop()

    nvtx.range_push("Token_QK_Attention forward")
    out = tqk_attn(stage1_data)
    nvtx.range_pop()
    nvtx.range_push("Token_QK_Attention backward")
    out.sum().backward()
    nvtx.range_pop()

    nvtx.range_push("Spiking_Self_Attention forward")
    out = ssa_attn(stage3_data)
    nvtx.range_pop()
    nvtx.range_push("Spiking_Self_Attention backward")
    out.sum().backward()
    nvtx.range_pop()
