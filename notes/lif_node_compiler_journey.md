# From spikingjelly to a compiler-friendly stateless LIF neuron

## What spikingjelly does under the hood

Spikingjelly's `LIFNode` is a rich, feature-complete library class. When you call it,
several layers of indirection fire before a single CUDA kernel runs:

**1. `torch.autograd.Function` subclass for surrogate gradients.**
Every spike operation goes through a custom `Function` with `forward()` and `backward()`
defined in Python. `torch.compile` (Dynamo) cannot inline through an arbitrary Python
`Function`'s `backward`. It sees the boundary as opaque and **breaks the FX graph** there.
Instead of one big fused kernel, you get the graph split into pieces around every spiking
neuron call.

**2. `_memories` dict dispatch.**
Spikingjelly's `MemoryModule` base class stores state in a dict (`_memories`) and
retrieves it via `__getattr__` overloads. Dynamo cannot statically determine what
`self._memories["v"]` contains at trace time because dict lookup is a general Python
operation. This generates **value guards** — checks like "is this tensor still the same
object?" — and recompiles when they fail.

**3. `isinstance(self.v, float)` type guard.**
Early in spikingjelly's forward, state is checked as either a float sentinel or a tensor.
Dynamo sees a Python `isinstance` test on a traced value and must specialise on both
branches, generating **recompile guards** that fire whenever the branch changes (e.g. on
the first batch after a reset).

**4. Backend dispatch.**
Spikingjelly supports cupy and triton backends selected at runtime. The runtime dispatch
path involves Python `if/elif` chains that Dynamo either specialises on (more guards) or
gives up on and breaks the graph.

**5. Per-batch `functional.reset_net` overhead.**
State is stored in the module across batches. The training loop must call
`functional.reset_net(model)` before every batch. That function walks every module in the
tree, checks `isinstance(m, MemoryModule)`, and calls `m.reset()` — a CPU-side loop
that synchronises with the GPU between batches.

Combined effect: `torch.compile` captures many tiny graph fragments separated by
Python-level breaks. Triton autotunes *per fragment* across dozens of neurons. Compile
time is dominated by this fragmented autotuning, and training throughput suffers from
the leftover kernel launch overhead between fragments.

---

## Intermediate version: stateful custom LIFNode

The first custom implementation ([`lif_node.py`](../src/models/visual_encoders/lif_node.py))
eliminated problems 1–4 but kept state in a module buffer:

```python
# __init__
self.register_buffer("v", torch.tensor(v_reset), persistent=False)

# single_step_forward
self._ensure_v(x)
v_charged = self.v + (x - (self.v - self.v_reset)) / self.tau
spike = self._fire(v_charged - self.v_threshold)
spike_d = spike.detach() if self.detach_reset else spike
self.v = self.v_reset * spike_d + (1.0 - spike_d) * v_charged  # ← the problem
return spike
```

### What was fixed

**Surrogate gradient (problem 1):** the STE identity replaces the custom `Function`:
```python
spike = (x >= 0.0).to(x.dtype)
surrogate = torch.sigmoid(alpha * x)
return surrogate + (spike - surrogate).detach()
```
Both `spike` (heaviside) and `surrogate` (differentiable) are plain tensor ops. Dynamo
traces straight through with no graph break. The `.detach()` is a first-class op in the
FX graph; it does not hide a Python backward.

**`_memories` dict (problem 2):** `self.v` is a plain attribute on a vanilla `nn.Module`.
Dynamo knows it is a tensor and generates one stable shape guard when the 0-dim sentinel
is replaced on the first forward.

**`isinstance` guard (problem 3):** the sentinel is initialised as `torch.tensor(v_reset)`,
a 0-dim tensor, not a Python float. Dynamo always sees a tensor type. No type-branch guard.

**Backend dispatch (problem 4):** pure torch, no dispatch.

### What remained broken: `self.v = new_tensor`

On each timestep the line `self.v = ...` **rebinds** the Python attribute to a
freshly-allocated tensor at a new GPU memory address. Dynamo emits a
`set_attr(module, "v", new_tensor)` side-effect node in the FX graph.

Under `torch.compile(mode="reduce-overhead")`, CUDA Graphs record the exact GPU addresses
used during warmup. On replay, `self.v` has been rebound to the tensor from the previous
forward. The address in the CUDA Graph no longer matches the live address → `RuntimeError`.

The workaround was `inplace_reset()` (`self.v.detach_().fill_(v_reset)`) between batches
to preserve the same address. But the within-forward rebinding was still there across the
T-step loop, so CUDA Graph capture over the full unrolled forward was impossible.

This version was already much faster than spikingjelly (one large graph fragment instead
of dozens), but `reduce-overhead` CUDA Graphs were still blocked.

---

## Current version: fully stateless LIF

The only change to the numerical computation is where `v` lives:

```python
def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
    v = torch.full_like(x_seq[0], self.v_reset)   # local variable, not self.v
    spike_seq = []
    for t in range(x_seq.shape[0]):
        spike, v = self._step(x_seq[t], v)
        spike_seq.append(spike)
    return torch.stack(spike_seq)

def _step(self, x, v_prev):
    v_charged = v_prev + (x - (v_prev - self.v_reset)) / self.tau
    spike     = self._fire(v_charged - self.v_threshold)
    spike_d   = spike.detach() if self.detach_reset else spike
    v_next    = self.v_reset * spike_d + (1.0 - spike_d) * v_charged
    return spike, v_next
```

`v` is a **local Python variable**. When Dynamo traces `spike, v = self._step(x_seq[t], v)`,
it sees `v` as a value flowing through the Python stack — a node in the FX graph passed
as an argument and returned as a result. No `set_attr` is emitted.

For T=4 Dynamo unrolls the loop and produces a pure SSA (Static Single Assignment) graph:

```
v0 = full_like(x_seq[0], 0.0)
spike0, v1 = _step(x_seq[0], v0)
spike1, v2 = _step(x_seq[1], v1)
spike2, v3 = _step(x_seq[2], v2)
spike3, v4 = _step(x_seq[3], v3)
return stack([spike0, spike1, spike2, spike3])
```

Each `v_i` is defined exactly once. There are no side-effects. The Inductor backend sees a
directed acyclic graph of tensor operations with:

- **Statically known live set:** Inductor pre-allocates the full buffer pool once at
  capture time. Every intermediate tensor (`v_charged`, `spike`, `v_next`) has a single
  known producer and consumer.
- **CUDA Graph replay:** on every subsequent call the GPU re-issues the same kernel launch
  commands against the same pre-allocated buffers. No Python executes. No kernel is
  launched individually. The GPU runs the entire T-step sequence autonomously.
- **Cross-timestep fusion opportunity:** because the graph is a single connected DAG,
  Inductor can consider pointwise ops from adjacent timesteps for fusion.

### Why compile time dropped ~10x

Previously Triton had to autotune a separate kernel for each graph fragment — one per
graph break. Now there is one graph. Triton autotuning fires once over the full fused
kernel.

### Why training throughput increased

The primary gain is CUDA Graphs. On `reduce-overhead` mode the CPU dispatches one replay
command and moves on to the next batch before the GPU has finished the current one.
With the stateful version the CPU had to:

1. Execute Python for each of the T timesteps (Dynamo's `set_attr` side-effects require
   Python-side bookkeeping even inside a compiled region).
2. Call `reset_lif_states(model)` before each batch — a CPU-side module tree walk
   that also blocked the CPU-GPU pipeline.

Both costs are now zero.

### Correctness improvement

With the stateful version, forgetting to call `reset_lif_states` before a batch caused
membrane potential from the previous batch to leak silently into the current one (training
converged, just slightly worse). With the stateless version this is impossible. Each
`forward` call starts from a deterministic `v_reset` initial condition with no dependency
on history.

---

## Summary

| Problem | spikingjelly | stateful custom LIF | stateless custom LIF |
|---|---|---|---|
| Graph breaks from `autograd.Function` | Many | None (STE trick) | None |
| Value guards from `_memories` dict | Many | None | None |
| Type guards from float sentinel | Present | None (0-dim tensor) | None |
| Backend dispatch Python branches | Present | None | None |
| `set_attr` side-effect in graph | N/A | Present (kills CUDA Graphs) | **None** |
| Per-batch CPU reset overhead | `functional.reset_net` walk | `reset_lif_states` walk | **Nothing** |
| CUDA Graph capture possible | No | No | **Yes** |
| Triton compilation units | ~dozens | ~1 | 1 |
| Leaked state risk | High | High (reset required) | **Impossible** |
