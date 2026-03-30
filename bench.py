import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import SnapshotEngine

B, G, Y, C, W = "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def pre_download(model_id):
    """Download model files so cold-start timing excludes network."""
    print(f"  downloading {model_id} ...", end=" ", flush=True)
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
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
    e = SnapshotEngine()

    print(f"\n{B}{'═' * 58}")
    print(f"  SnapInfer Benchmark")
    print(f"  {gpu_name} ({vram} MB)")
    print(f"{'═' * 58}{W}\n")

    # ensure everything is cached locally first
    all_models = set([a.model] + (a.models or []))
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

    # evict → restore from pinned memory
    e.evict()
    time.sleep(0.3)
    restore_ms = e.restore(m)

    # verify inference after restore
    _, text2 = e.complete(m, a.prompt, a.tokens, temperature=0)

    ok = text1.strip() == text2.strip()
    speedup = load_s / (restore_ms / 1000) if restore_ms > 0 else float("inf")
    bw = (sz / 2**30) / (restore_ms / 1000) if restore_ms > 0 else 0

    print(f"  Weights           {sz / 2**20:>8.0f} MB   fp16")
    print(f"  Cold start (disk) {Y}{load_s:>8.3f} s{W}")
    print(f"  Snapshot capture  {C}{snap_s:>8.3f} s{W}    GPU → pinned RAM")
    print(f"  Snapshot restore  {G}{restore_ms / 1000:>8.3f} s{W}    pinned RAM → GPU")
    print(f"  PCIe throughput   {bw:>8.1f} GB/s")
    print()
    tag = f"{G}✓ outputs match{W}" if ok else "\033[31m✗ MISMATCH\033[0m"
    print(f"  {B}⚡ {speedup:,.1f}× faster cold start{W}   {tag}")
    print(f'  "{text1[:72]}{"…" if len(text1) > 72 else ""}"')

    e.evict()

    # ── phase 2: multi-model rapid switching ─────────────────────
    if a.models:
        print(f"\n{C}▸ Rapid model switching "
              f"({len(a.models)} models, 1 GPU){W}\n")

        for mid in a.models:
            ls, ss = e.warmup(mid)
            smb = e.snaps[mid].size_bytes / 2**20
            print(f"  {mid:20s} {smb:>6.0f} MB   "
                  f"loaded {Y}{ls:.1f}s{W}  snapped {C}{ss:.2f}s{W}")

        print()
        for mid in a.models:
            ms = e.restore(mid)
            _, txt = e.complete(mid, "Hello world", 15, temperature=0)
            e.evict()
            print(f"  {mid:20s} restore {G}{ms:>6.0f} ms{W}  "
                  f'"{txt.strip()[:50]}"')

    print(f"\n{B}{'═' * 58}{W}\n")


if __name__ == "__main__":
    main()
