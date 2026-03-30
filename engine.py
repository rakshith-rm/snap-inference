import torch, time, gc, warnings, threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Snapshot:
    __slots__ = ("model_id", "config", "tokenizer", "tensors",
                 "dtype", "size_bytes", "gen_config")

    def __init__(self, model_id, config, tokenizer, tensors, dtype, gen_config):
        self.model_id = model_id
        self.config = config
        self.tokenizer = tokenizer
        self.tensors = tensors
        self.dtype = dtype
        self.size_bytes = sum(t.nbytes for t in tensors.values()) if tensors else 0
        self.gen_config = gen_config


class SnapshotEngine:
    """Capture GPU model state → CPU pinned memory. Restore at PCIe bandwidth."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.snaps: dict[str, Snapshot] = {}
        self.model = None
        self.model_id = None
        self._lock = threading.Lock()

    # ── load ─────────────────────────────────────────────────────────
    def load(self, model_id, hf_name=None, dtype=torch.float16):
        """Normal cold start: disk/HF cache → GPU. Returns seconds."""
        hf_name = hf_name or model_id
        self.evict()

        tok = AutoTokenizer.from_pretrained(hf_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=dtype,
        ).to(self.device).eval()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0

        self.model, self.model_id = m, model_id
        self.snaps[model_id] = Snapshot(
            model_id, m.config, tok, {},
            dtype, getattr(m, "generation_config", None),
        )
        return load_s

    # ── snapshot ─────────────────────────────────────────────────────
    def snapshot(self):
        """Capture active GPU model → pinned CPU memory. Returns seconds."""
        assert self.model is not None, "no active model"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tensors = {}
        for k, v in self.model.state_dict().items():
            p = torch.empty_like(v, device="cpu", pin_memory=True)
            p.copy_(v, non_blocking=True)
            tensors[k] = p
        torch.cuda.synchronize()
        snap_s = time.perf_counter() - t0

        s = self.snaps[self.model_id]
        s.tensors = tensors
        s.size_bytes = sum(t.nbytes for t in tensors.values())
        s.config = self.model.config
        s.gen_config = getattr(self.model, "generation_config", None)
        s.dtype = next(self.model.parameters()).dtype
        return snap_s

    # ── evict ────────────────────────────────────────────────────────
    def evict(self):
        """Free active model from GPU."""
        if self.model is not None:
            del self.model
            self.model = self.model_id = None
            gc.collect()
            torch.cuda.empty_cache()

    # ── restore ──────────────────────────────────────────────────────
    def restore(self, model_id):
        """Restore from pinned snapshot → GPU. Returns cold-start ms."""
        if self.model_id == model_id:
            return 0.0
        snap = self.snaps[model_id]
        assert snap.tensors, f"no snapshot for '{model_id}'"
        self.evict()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. build model skeleton on meta device (zero memory, zero compute)
        with warnings.catch_warnings(), torch.device("meta"):
            warnings.simplefilter("ignore")
            m = AutoModelForCausalLM.from_config(snap.config, torch_dtype=snap.dtype)

        # 2. DMA: pinned CPU → GPU at PCIe bandwidth (the fast path)
        gpu_state = {k: v.to(self.device, non_blocking=True)
                     for k, v in snap.tensors.items()}
        torch.cuda.synchronize()

        # 3. assign gpu tensors into the meta shell (pointer swap, no copy)
        m.load_state_dict(gpu_state, assign=True)
        m.eval()
        if snap.gen_config:
            m.generation_config = snap.gen_config

        ms = (time.perf_counter() - t0) * 1000
        self.model, self.model_id = m, model_id
        return ms

    # ── warmup (convenience) ─────────────────────────────────────────
    def warmup(self, model_id, hf_name=None, dtype=torch.float16):
        """Load → snapshot → evict. Returns (load_s, snap_s)."""
        ls = self.load(model_id, hf_name, dtype)
        ss = self.snapshot()
        self.evict()
        return ls, ss

    # ── inference ────────────────────────────────────────────────────
    def _ensure(self, model_id):
        with self._lock:
            return self.restore(model_id)

    def _prompt(self, model_id, messages):
        tok = self.snaps[model_id].tokenizer
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            ) + "\nassistant:"

    def _gkw(self, tok, temperature, max_tokens):
        kw = dict(max_new_tokens=max_tokens,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw["do_sample"] = False
        return kw

    def generate(self, model_id, messages, max_tokens=256, temperature=0.7):
        cold_ms = self._ensure(model_id)
        tok = self.snaps[model_id].tokenizer
        ids = tok(self._prompt(model_id, messages),
                  return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, **self._gkw(tok, temperature, max_tokens))
        text = tok.decode(out[0][ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        return cold_ms, text

    def generate_stream(self, model_id, messages, max_tokens=256, temperature=0.7):
        cold_ms = self._ensure(model_id)
        tok = self.snaps[model_id].tokenizer
        ids = tok(self._prompt(model_id, messages),
                  return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                       skip_special_tokens=True)

        def _run():
            with torch.no_grad():
                self.model.generate(
                    **ids, **self._gkw(tok, temperature, max_tokens),
                    streamer=streamer)

        threading.Thread(target=_run, daemon=True).start()
        return cold_ms, streamer

    def complete(self, model_id, prompt, max_tokens=256, temperature=0.7):
        """Plain text completion (no chat template)."""
        cold_ms = self._ensure(model_id)
        tok = self.snaps[model_id].tokenizer
        ids = tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, **self._gkw(tok, temperature, max_tokens))
        return cold_ms, tok.decode(out[0][ids.input_ids.shape[1]:],
                                   skip_special_tokens=True)

    def gpu_mb(self):
        return torch.cuda.memory_allocated(self.device) / 1024**2
