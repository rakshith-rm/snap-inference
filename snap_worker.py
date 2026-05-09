import struct

import torch

MAGIC = b"SNAP"
VERSION = 1
HEADER_BYTES = 16
TABLE_ENTRY_BYTES = 24


def _worker_model(worker):
    return worker.model_runner.model


def worker_write_snapshot(worker, path: str):
    model = _worker_model(worker)
    params = [p.data.contiguous() for _, p in model.named_parameters() if p.is_cuda]
    sizes = [p.nbytes for p in params]
    n = len(params)

    payload_start = HEADER_BYTES + n * TABLE_ENTRY_BYTES
    offsets, off = [], payload_start
    for s in sizes:
        offsets.append(off)
        off += s
    total_bytes = sum(sizes)

    flat = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    cur = 0
    for p in params:
        nbytes = p.nbytes
        flat[cur : cur + nbytes].copy_(p.reshape(-1).view(torch.uint8), non_blocking=True)
        cur += nbytes
    torch.cuda.synchronize()

    header = MAGIC + struct.pack("<IQ", VERSION, n)
    table = b"".join(
        struct.pack("<QQQ", i, s, o) for i, (s, o) in enumerate(zip(sizes, offsets))
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(table)
        f.write(flat.numpy())
    return {"num_regions": n, "total_bytes": total_bytes}


def _read_snapshot(path: str):
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("invalid snapshot magic")
        version = struct.unpack("<I", f.read(4))[0]
        if version != VERSION:
            raise ValueError(f"unsupported snapshot version {version}")
        num_regions = struct.unpack("<Q", f.read(8))[0]
        entries = [
            struct.unpack("<QQQ", f.read(TABLE_ENTRY_BYTES)) for _ in range(num_regions)
        ]
        total = sum(s for _, s, _ in entries)
        payload_start = entries[0][2] if entries else HEADER_BYTES
        f.seek(payload_start)
        flat = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        got = f.readinto(flat.numpy())
        if got != total:
            raise ValueError(f"truncated snapshot payload: expected {total}, got {got}")
    return entries, flat, total


def worker_restore_snapshot(worker, path: str):
    model = _worker_model(worker)
    params = [p for _, p in model.named_parameters() if p.is_cuda]
    entries, flat, total = _read_snapshot(path)
    if len(entries) != len(params):
        raise ValueError(f"snapshot has {len(entries)} regions, model has {len(params)}")

    flat_gpu = flat.to(params[0].device, non_blocking=True)
    torch.cuda.synchronize()
    cur = 0
    for i, p in enumerate(params):
        _, size, _ = entries[i]
        if size != p.data.nbytes:
            raise ValueError(f"size mismatch at param {i}: {size} != {p.data.nbytes}")
        p.data.copy_(flat_gpu[cur : cur + size].view(p.dtype).reshape(p.shape))
        cur += size
    torch.cuda.synchronize()
    del flat_gpu
    return {"num_regions": len(entries), "total_bytes": total}

