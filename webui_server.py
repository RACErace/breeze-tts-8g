#!/usr/bin/env python3
"""Breeze TTS 2 WebUI server.

Reuses the official streaming inference API (``breeze_infer.api``) in a single
process and serves a lightweight single-page frontend.  No Gradio, no Node,
no build step.

Memory / performance design:
  * one process, one model copy, loaded once by the API lifespan;
  * no ASR / speaker-encoder / extra models on the GPU;
  * raw PCM is streamed to the browser chunk by chunk (no server-side buffering);
  * the checkpoint digest used by ``/v1/model`` is cached on disk, so restarts
    skip re-hashing ~7 GB of weights;
  * single inference lock (409 while busy) -> the client queues requests.

Run:
  cd <repo>
  python webui_server.py <checkpoint> --port 7860
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import breeze_infer.api as api

HERE = Path(__file__).resolve().parent
WEBUI_HTML = HERE / "webui" / "index.html"
CACHE_DIR = Path.home() / ".cache" / "breeze-webui"
CACHE_FILE = CACHE_DIR / "model_info.json"
HISTORY_DIR = Path.home() / ".local" / "share" / "breeze-webui" / "takes"
TAKE_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
MAX_TAKE_TEXT_CHARS = 20000


def _checkpoint_fingerprint(model_dir: Path) -> dict[str, object]:
    """Cheap stat-based fingerprint of the checkpoint files (no hashing)."""
    index = model_dir / "model.safetensors.index.json"
    shards: list[str] = []
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            shards = []
    if not shards:
        shards = ["model.safetensors"]
    names = ["config.json", index.name, *shards]
    fingerprint: dict[str, object] = {}
    for name in names:
        path = model_dir / name
        try:
            stat = path.stat()
        except OSError:
            fingerprint[name] = None
            continue
        fingerprint[name] = [stat.st_size, stat.st_mtime_ns]
    return fingerprint


def cached_model_info(model_dir: Path) -> dict:
    """Return ``build_model_info`` result, cached until the checkpoint changes."""
    fingerprint = _checkpoint_fingerprint(model_dir)
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cached.get("model_dir") == str(model_dir) and cached.get("fingerprint") == fingerprint:
            return cached["info"]
    except (OSError, ValueError, KeyError):
        pass
    info = api.build_model_info(model_dir)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {"model_dir": str(model_dir), "fingerprint": fingerprint, "info": info},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return info


app = api.app


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    try:
        html = WEBUI_HTML.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - deployment guard
        raise HTTPException(status_code=500, detail=f"WebUI HTML missing: {exc}") from exc
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> JSONResponse:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    import psutil
    import torch

    gpu_used = gpu_total = None
    try:
        gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
        gpu_used = int(torch.cuda.memory_allocated(0))
    except Exception:  # pragma: no cover - informational only
        pass
    ram = psutil.virtual_memory()
    settings = api._settings
    return JSONResponse(
        {
            "status": "ok",
            "model_dir": str(settings.model) if settings else None,
            "sample_rate": runtime.sample_rate,
            "frame_rate": settings.model_info.get("frame_rate") if settings else None,
            "max_new_tokens": api.MAX_NEW_TOKENS,
            "gpu_used": gpu_used,
            "gpu_total": gpu_total,
            "ram_used": int(ram.used),
            "ram_total": int(ram.total),
            "busy": api._request_lock.locked(),
        }
    )


def _take_meta_path(take_id: str) -> Path:
    return HISTORY_DIR / f"{take_id}.json"


def _take_wav_path(take_id: str) -> Path:
    return HISTORY_DIR / f"{take_id}.wav"


def _read_take(take_id: str) -> dict | None:
    if not TAKE_ID_RE.match(take_id):
        return None
    try:
        record = json.loads(_take_meta_path(take_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not _take_wav_path(take_id).is_file():
        return None
    record["url"] = f"/api/takes/{take_id}/audio"
    return record


def _list_takes() -> list[dict]:
    if not HISTORY_DIR.is_dir():
        return []
    records: list[dict] = []
    for meta_path in HISTORY_DIR.glob("*.json"):
        record = _read_take(meta_path.stem)
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records


@app.get("/api/takes")
def list_takes() -> JSONResponse:
    return JSONResponse({"takes": _list_takes()})


@app.post("/api/takes")
async def create_take(
    file: UploadFile = File(...),
    text: str = Form(""),
    mode: str = Form("plain"),
    sample_rate: int = Form(24000),
    duration_s: float = Form(0.0),
) -> JSONResponse:
    data = await file.read()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="expected a RIFF/WAVE file")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    existing = _list_takes()
    number = 1 + max((int(item.get("number", 0)) for item in existing), default=0)
    take_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record = {
        "id": take_id,
        "number": number,
        "text": text[:MAX_TAKE_TEXT_CHARS],
        "mode": mode[:32],
        "sample_rate": int(sample_rate),
        "duration_s": round(float(duration_s), 3),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bytes": len(data),
    }
    _take_wav_path(take_id).write_bytes(data)
    _take_meta_path(take_id).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record["url"] = f"/api/takes/{take_id}/audio"
    return JSONResponse(record)


@app.get("/api/takes/{take_id}/audio")
def take_audio(take_id: str) -> FileResponse:
    if not TAKE_ID_RE.match(take_id):
        raise HTTPException(status_code=404, detail="take not found")
    path = _take_wav_path(take_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="take not found")
    return FileResponse(path, media_type="audio/wav", filename=f"breeze-take-{take_id}.wav")


@app.delete("/api/takes/{take_id}")
def delete_take(take_id: str) -> JSONResponse:
    if not TAKE_ID_RE.match(take_id):
        raise HTTPException(status_code=404, detail="take not found")
    removed = False
    for path in (_take_wav_path(take_id), _take_meta_path(take_id)):
        if path.is_file():
            path.unlink()
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="take not found")
    return JSONResponse({"ok": True, "id": take_id})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Breeze TTS 2 WebUI")
    parser.add_argument("model", help="Local checkpoint directory or Hugging Face repo id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--fast-all", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-codec", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=api.MAX_SEQ_LEN,
        help="Maximum context length (prompt + generated frames).",
    )
    parser.add_argument(
        "--int8-linears",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MLP projections with weight-only int8 kernels.",
    )
    args = parser.parse_args()
    if args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2")

    try:
        model_dir = api.resolve_model_dir(args.model)
    except api.ModelResolutionError as exc:
        parser.error(str(exc))

    api._settings = api.ApiSettings(
        model=model_dir,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        model_info=cached_model_info(model_dir),
        max_seq_len=args.max_seq_len,
        int8_linears=args.int8_linears,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
