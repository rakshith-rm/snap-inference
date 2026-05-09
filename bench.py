import argparse, time, torch
from transformers import AutoTokenizer
from engine import SnapshotEngine

B, G, Y, C, W = "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def pre_download(model_id):
    """Download model files so cold-start timing excludes network."""
    print(f"  downloading {model_id} ...", end=" ", flush=True)
    AutoTokenizer.from_pretrained(model_id)
    print("cached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--models", nargs="*",
                    default=["distilgpt2", "gpt2", "gpt2-medium"])
    ap.add_argument("--prompt",
                    default="The future of artificial intelligence is")
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()

    gpu_name = torch.cuda.get_device_name()
    vram = torch.cuda.get_device_properties(0).total_memory // 2**20

    # In phase 2 we keep one warm vLLM slot per model. Each slot reserves a
    # share of VRAM (mostly KV cache), so we shrink the per-slot fraction
    # based on how many models we plan to keep warm simultaneously.
    n_slots = max(1, len(a.models))
    slot_gmu = min(0.4, 0.8 / n_slots)
    e = SnapshotEngine(slot_gpu_memory_utilization=slot_gmu)

    print(f"\n{B}{'═' * 58}")
    print(f"  SnapInfer Benchmark")
    print(f"  {gpu_name} ({vram} MB)")
    print(f"{'═' * 58}{W}\n")

    # ensure everything is cached locally first
    all_models = {a.model, *a.models}
    for mid in all_models:
        pre_download(mid)
    print()

    # ── phase 1: single-model cold-start comparison ──────────────
    m = a.model
    print(f"{C}▸ {m}{W}\n")

    load_s = e.load(m)
    snap_s = e.snapshot()
    sz = e.snaps[m].size_bytes

    # baseline inference (model already on GPU)
    _, text1 = e.complete(m, a.prompt, a.tokens, temperature=0)

    # First restore creates the dummy vLLM slot, so it includes vLLM bootstrap.
    e.evict()
    time.sleep(0.3)
    cold_slot_restore_ms = e.restore(m)

    # verify inference after restore
    _, text2 = e.complete(m, a.prompt, a.tokens, temperature=0)

    # Second restore reuses the existing vLLM slot, so it measures just the
    # snapshotted model load into an already-running engine.
    e.evict()
    time.sleep(0.1)
    warm_slot_restore_ms = e.restore(m)
    _, text3 = e.complete(m, a.prompt, a.tokens, temperature=0)

    ok = text1.strip() == text2.strip() == text3.strip()
    speedup = (
        load_s / (warm_slot_restore_ms / 1000)
        if warm_slot_restore_ms > 0 else float("inf")
    )
    bw = (sz / 2**30) / (warm_slot_restore_ms / 1000) if warm_slot_restore_ms > 0 else 0

    print(f"  Weights           {sz / 2**20:>8.0f} MB   fp16")
    print(f"  vLLM cold start   {Y}{load_s:>8.3f} s{W}    engine + disk weights")
    print(f"  Snapshot capture  {C}{snap_s:>8.3f} s{W}    offline prep, not counted")
    print(f"  Restore cold slot {C}{cold_slot_restore_ms / 1000:>8.3f} s{W}    engine + snapshot weights")
    print(f"  Restore warm slot {G}{warm_slot_restore_ms / 1000:>8.3f} s{W}    snapshot weights only")
    print(f"  PCIe throughput   {bw:>8.1f} GB/s")
    print()
    tag = f"{G}✓ outputs match{W}" if ok else "\033[31m✗ MISMATCH\033[0m"
    print(f"  {B}⚡ {speedup:,.1f}× faster cold start{W}   {tag}")
    print(f'  "{text1[:72]}{"…" if len(text1) > 72 else ""}"')

    e.evict()

    # ── phase 2: multi-model rapid switching ─────────────────────
    if a.models:
        print(f"\n{C}▸ Rapid model switching "
              f"({len(a.models)} models, 1 GPU, slot gmu={slot_gmu:.2f}){W}\n")

        for mid in a.models:
            ls, ss = e.warmup(mid)
            smb = e.snaps[mid].size_bytes / 2**20
            print(f"  {mid:30s} {smb:>6.0f} MB   "
                  f"loaded {Y}{ls:.1f}s{W}  snapped {C}{ss:.2f}s{W}")

        # First pass: each switch creates a fresh dummy vLLM slot, so each
        # restore pays the engine bootstrap cost the first time.
        print(f"\n  {B}First pass (slot creation per model){W}")
        for mid in a.models:
            ms = e.restore(mid)
            _, txt = e.complete(mid, "Hello world", 15, temperature=0)
            print(f"  {mid:30s} restore {Y}{ms:>7.0f} ms{W}  "
                  f'"{txt.strip()[:40]}"')

        # Second pass: the slot for each model is now warm, so switching only
        # restores weights into an already-running vLLM engine.
        print(f"\n  {B}Second pass (warm slots reused){W}")
        for mid in a.models:
            e.evict()
            ms = e.restore(mid)
            _, txt = e.complete(mid, "Hello world", 15, temperature=0)
            print(f"  {mid:30s} restore {G}{ms:>7.0f} ms{W}  "
                  f'"{txt.strip()[:40]}"')

    print(f"\n{B}{'═' * 58}{W}\n")


if __name__ == "__main__":
    main()
