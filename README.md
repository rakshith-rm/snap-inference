# SnapInfer

Small vLLM prototype for weight snapshotting and fast model activation on a
single GPU.

## What It Does

- Starts a real vLLM engine and loads model weights normally.
- Captures GPU model weights into a `.snap` file.
- Restores those weights into a dummy-loaded vLLM engine.
- Keeps one warm vLLM slot per model for fast second-pass switching.
- Verifies correctness by comparing generated output before and after restore.

This is weights-only snapshotting. It does not snapshot the full container,
CUDA context, or vLLM runtime state.

## Demo Command

```bash
python bench.py --model distilgpt2 --models distilgpt2 gpt2 --tokens 32
```

## Latest Result

Run on:

```text
NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, WSL
```

Single-model result for `distilgpt2`:

```text
vLLM cold start      13.539 s
Snapshot capture      1.084 s
Restore cold slot    11.730 s
Restore warm slot     0.744 s
Speedup              18.2x
Outputs match        yes
```

Multi-model switching after warm slots were created:

```text
distilgpt2 restore    773 ms
gpt2 restore         1229 ms
```

## What This Demonstrates

SnapInfer shows that once vLLM slots are warm, model activation can be reduced
from full cold load time to sub-second or near-sub-second weight restore.

The current prototype demonstrates fast multi-model switching on one GPU by
keeping warm vLLM slots alive for each model.

## Current Limits

- First restore for each model still pays vLLM engine bootstrap cost.
- Warm slots reserve GPU memory while idle.
- This is not full serverless cold start from zero GPU state.
- WSL disables pinned memory, so restore throughput is slower than on a native
  Linux NVIDIA machine.

## Next Steps

Planned work is to move from weights-only snapshots to full vLLM runtime state
capture. The goal is to snapshot an already-initialized vLLM container, including
the Python process, vLLM engine state, CUDA context, GPU memory, and model state,
so restoring the container can begin serving immediately without running the
normal vLLM cold-start path.

Areas to explore:

- Use CRIU to checkpoint and restore the Linux process/container state.
- Use NVIDIA CUDA checkpointing support to preserve CUDA context and GPU memory.
- Explore CUDA Graph capture/restore behavior for reducing warmup and replay
  overhead after restore.
- Combine runtime-state restore with the current weight snapshot path to move
  toward true serverless-style inference startup.
