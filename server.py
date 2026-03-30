import os, json, time, uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from engine import SnapshotEngine

app = FastAPI(title="SnapInfer")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
engine = SnapshotEngine()


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = "gpt2"
    messages: list[Msg]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


@app.on_event("startup")
def startup():
    for m in filter(None, os.getenv("MODELS", "").split(",")):
        m = m.strip()
        if m:
            print(f"[warmup] {m} ...", end=" ", flush=True)
            ls, ss = engine.warmup(m)
            print(f"loaded {ls:.1f}s, snapped {ss:.2f}s")
    print(f"[ready] {len(engine.snaps)} model(s) snapshotted")


@app.get("/v1/models")
def models():
    return {"data": [
        {"id": k, "size_mb": round(v.size_bytes / 2**20)}
        for k, v in engine.snaps.items()
    ]}


@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    # first request for an unknown model → slow warmup (once)
    if req.model not in engine.snaps:
        print(f"[first-load] {req.model}")
        engine.warmup(req.model)

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream(req.model, msgs, req.max_tokens, req.temperature),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    cold_ms, text = engine.generate(
        req.model, msgs, req.max_tokens, req.temperature)
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "id": f"snap-{uuid.uuid4().hex[:8]}",
        "model": req.model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "timing": {
            "cold_start_ms": round(cold_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


async def _stream(model, msgs, max_tok, temp):
    cold_ms, streamer = engine.generate_stream(model, msgs, max_tok, temp)
    yield f"data: {json.dumps({'cold_start_ms': round(cold_ms, 1)})}\n\n"
    for tok in streamer:
        yield ("data: " + json.dumps({
            "choices": [{"delta": {"content": tok}}]
        }) + "\n\n")
    yield "data: [DONE]\n\n"


@app.get("/status")
def status():
    return {
        "active": engine.model_id,
        "gpu_mb": round(engine.gpu_mb(), 1),
        "snapshots": {
            k: round(v.size_bytes / 2**20, 1)
            for k, v in engine.snaps.items()
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
