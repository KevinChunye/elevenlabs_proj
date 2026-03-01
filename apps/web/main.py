from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.core.config import load_config
from packages.core.elevenlabs_stt import ElevenLabsError, request_realtime_token
from packages.core.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv()

app = FastAPI(title="Emotion Inference MVP")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    video: UploadFile = File(...),
    taxonomy: str = Form("ekman"),
):
    suffix = Path(video.filename or "input.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=400, detail="Only mp4/mov are supported")

    upload_dir = OUTPUTS_DIR / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    local_path = upload_dir / f"upload_{video.filename or 'input.mp4'}"

    with local_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    result = run_pipeline(video_path=str(local_path), output_root=str(OUTPUTS_DIR), taxonomy=taxonomy, allow_mock_stt=True)

    run_id = result["run_id"]
    viewer_rel = f"/outputs/{run_id}/viewer.html"
    transcript_rel = f"/outputs/{run_id}/transcript.json"
    emotions_rel = f"/outputs/{run_id}/emotions.json"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "run_id": run_id,
            "viewer_url": viewer_rel,
            "transcript_url": transcript_rel,
            "emotions_url": emotions_rel,
            "stt_error": result.get("stt_error"),
        },
    )


@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page(request: Request):
    return templates.TemplateResponse("realtime.html", {"request": request})


@app.post("/api/realtime-token")
async def realtime_token():
    cfg = load_config("ekman")
    if not cfg.elevenlabs_api_key:
        raise HTTPException(status_code=400, detail="ELEVENLABS_API_KEY is not set")

    try:
        payload = request_realtime_token(cfg.elevenlabs_api_key)
    except ElevenLabsError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return JSONResponse(payload)
