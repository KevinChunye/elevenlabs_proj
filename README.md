# Multimodal Emotion Enhanced Transcription Bot

A lightweight Python MVP for video-based emotion timeline generation.

## What it does
- Accepts local `mp4`/`mov` via CLI or web UI
- Extracts audio with `ffmpeg`
- Transcribes speech with ElevenLabs STT (batch)
- Runs multimodal inference:
  - audio prosody-based emotion baseline
  - face + pose signals (MediaPipe)
- Fuses results into per-segment labels using selectable taxonomy:
  - `ekman` (default)
  - `plutchik`
  - `pucek`
- Generates outputs per run:
  - `outputs/<run_id>/transcript.json`
  - `outputs/<run_id>/emotions.json`
  - `outputs/<run_id>/viewer.html`

## Project layout
- `apps/cli` CLI runner (`run`, `smoke`, `realtime`)
- `apps/web` tiny FastAPI upload UI + realtime demo page
- `packages/core` pipeline, feature extraction, fusion, schemas
- `tests` minimal schema/fusion tests

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install `ffmpeg` / `ffprobe` first (e.g., `brew install ffmpeg`).

Set environment:
- `ELEVENLABS_API_KEY`
- Optional: `ELEVENLABS_MODEL_ID` (defaults to `scribe_v2`)

## CLI
Analyze a local video:
```bash
python apps/cli/main.py run --video /absolute/path/video.mp4 --taxonomy ekman
```

Smoke test:
```bash
python apps/cli/main.py smoke
```

Realtime demo mode:
```bash
python apps/cli/main.py run --realtime --host 127.0.0.1 --port 8000
```

## Web UI
```bash
uvicorn apps.web.main:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000`.

## Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Security
- No API keys are committed.
- `.env` is ignored by git.
