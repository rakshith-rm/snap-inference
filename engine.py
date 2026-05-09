import gc
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from snap_worker import worker_restore_snapshot, worker_write_snapshot


@dataclass(slots=True)
class Snapshot:
    model_id: str
    tokenizer: object
    snapshot_path: str
    dtype: str
    size_bytes: int = 0


class SnapshotEngine:
    """Capture vLLM weights to snapshot files and restore quickly."""

    def __init__(
        self,
        device="cuda:0",
        snapshot_dir="snapshots",
        gpu_memory_utilization=0.85,
        slot_gpu_memory_utilization=None,
    ):
        self.device = device
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_memory_utilization = gpu_memory_utilization
        # Each warm slot is its own dummy vLLM engine with its own KV cache; the
        # per-slot fraction must be small enough that N slots fit in VRAM.
        self.slot_gpu_memory_utilization = (
            slot_gpu_memory_utilization or gpu_memory_utilization
        )
        self.snaps: dict[str, Snapshot] = {}
        self.model = None
        self.model_id = None
        self._lock = threading.Lock()
        self._active_snapshot = None
        self._slots: dict[str, "LLM"] = {}

    def _snapshot_path(self, model_id):
        safe = model_id.replace("/", "__").replace("\\", "__")
        return str(self.snapshot_dir / f"{safe}.snap")

    def _drop_all_slots(self):
        if self._slots:
            self._slots.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def load(self, model_id, hf_name=None, dtype="float16"):
        """Normal vLLM load from model files. Returns seconds."""
        hf_name = hf_name or model_id
        self.evict(keep_slots=False)

        tok = AutoTokenizer.from_pretrained(hf_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        t0 = time.perf_counter()
        llm = LLM(
            model=hf_name,
            dtype=dtype,
            enforce_eager=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        load_s = time.perf_counter() - t0

        self.model, self.model_id = llm, model_id
        self._active_snapshot = None
        if model_id not in self.snaps:
            self.snaps[model_id] = Snapshot(
                model_id, tok, self._snapshot_path(model_id), dtype, 0
            )
        return load_s

    def snapshot(self):
        """Capture active vLLM model weights into a snapshot file. Returns seconds."""
        assert self.model is not None, "no active model"
        t0 = time.perf_counter()
        snap = self.snaps[self.model_id]
        result = self.model.collective_rpc(worker_write_snapshot, args=(snap.snapshot_path,))
        snap.size_bytes = result[0]["total_bytes"]
        self._active_snapshot = snap.snapshot_path
        return time.perf_counter() - t0

    def evict(self, keep_slots=True):
        """Deactivate current model; keep_slots=False also drops the warm pool."""
        if self.model is not None:
            self.model = self.model_id = None
            self._active_snapshot = None
        if keep_slots:
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self._drop_all_slots()

    def restore(self, model_id):
        """Restore weights into a warm dummy vLLM slot for this model. Returns ms."""
        if self.model_id == model_id:
            return 0.0
        snap = self.snaps[model_id]
        assert os.path.exists(snap.snapshot_path), f"no snapshot file for '{model_id}'"
        self.evict(keep_slots=True)

        t0 = time.perf_counter()
        if model_id not in self._slots:
            self._slots[model_id] = LLM(
                model=snap.model_id,
                dtype=snap.dtype,
                enforce_eager=True,
                load_format="dummy",
                gpu_memory_utilization=self.slot_gpu_memory_utilization,
            )

        self.model = self._slots[model_id]
        self.model_id = model_id
        self.model.collective_rpc(worker_restore_snapshot, args=(snap.snapshot_path,))

        self._active_snapshot = snap.snapshot_path
        return (time.perf_counter() - t0) * 1000

    def warmup(self, model_id, hf_name=None, dtype="float16"):
        """Load → snapshot → evict. Returns (load_s, snap_s)."""
        ls = self.load(model_id, hf_name, dtype)
        ss = self.snapshot()
        self.evict()
        return ls, ss

    def _ensure(self, model_id):
        with self._lock:
            return self.restore(model_id)

    def _prompt(self, model_id, messages):
        tok = self.snaps[model_id].tokenizer
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    @staticmethod
    def _sampling(temperature, max_tokens):
        t = float(temperature) if temperature and temperature > 0 else 0.0
        return SamplingParams(temperature=t, max_tokens=max_tokens)

    def _gen_text(self, prompt, max_tokens, temperature):
        out = self.model.generate([prompt], self._sampling(temperature, max_tokens))
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def generate(self, model_id, messages, max_tokens=256, temperature=0.7):
        cold_ms = self._ensure(model_id)
        prompt = self._prompt(model_id, messages)
        return cold_ms, self._gen_text(prompt, max_tokens, temperature)

    def generate_stream(self, model_id, messages, max_tokens=256, temperature=0.7):
        cold_ms, text = self.generate(model_id, messages, max_tokens, temperature)
        return cold_ms, iter([text])

    def complete(self, model_id, prompt, max_tokens=256, temperature=0.7):
        """Plain text completion (no chat template)."""
        cold_ms = self._ensure(model_id)
        return cold_ms, self._gen_text(prompt, max_tokens, temperature)

    def gpu_mb(self):
        return torch.cuda.memory_allocated(self.device) / 1024**2
