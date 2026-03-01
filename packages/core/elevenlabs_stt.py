from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .ffmpeg_utils import get_video_duration_seconds
from .schemas import TranscriptWord

BATCH_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
REALTIME_TOKEN_URL = "https://api.elevenlabs.io/v1/speech-to-text/token"


class ElevenLabsError(RuntimeError):
    pass


def create_batch_transcript(
    media_path: str,
    api_key: str,
    model_id: str = "scribe_v2",
    language_code: Optional[str] = None,
) -> Dict[str, Any]:
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")

    with open(media_path, "rb") as f:
        files = {"file": (Path(media_path).name, f, "application/octet-stream")}
        data = {
            "model_id": model_id,
            "diarize": "true",
            "timestamps_granularity": "word",
        }
        if language_code:
            data["language_code"] = language_code

        headers = {"xi-api-key": api_key}
        resp = requests.post(BATCH_STT_URL, headers=headers, data=data, files=files, timeout=600)

    # Fallback for older model naming if v2 is unavailable on account.
    if resp.status_code >= 400 and model_id == "scribe_v2":
        with open(media_path, "rb") as f:
            files = {"file": (Path(media_path).name, f, "application/octet-stream")}
            retry_data = dict(data)
            retry_data["model_id"] = "scribe_v1"
            resp = requests.post(BATCH_STT_URL, headers={"xi-api-key": api_key}, data=retry_data, files=files, timeout=600)

    if resp.status_code >= 400:
        raise ElevenLabsError(f"Batch STT failed ({resp.status_code}): {resp.text[:400]}")

    return resp.json()


def parse_words_from_stt_response(payload: Dict[str, Any]) -> List[TranscriptWord]:
    words_raw = payload.get("words") or payload.get("data", {}).get("words") or []
    parsed: List[TranscriptWord] = []

    for w in words_raw:
        if not isinstance(w, dict):
            continue
        text = str(w.get("text", "")).strip()
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue

        speaker = w.get("speaker_id") or w.get("speaker")
        token_type = w.get("type")
        parsed.append(
            TranscriptWord(
                text=text,
                start=float(start),
                end=float(end),
                speaker_id=str(speaker) if speaker is not None else None,
                token_type=str(token_type) if token_type is not None else None,
            )
        )

    parsed.sort(key=lambda x: (x.start, x.end))
    return parsed


def request_realtime_token(api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")
    headers = {"xi-api-key": api_key}
    resp = requests.post(REALTIME_TOKEN_URL, headers=headers, timeout=30)
    if resp.status_code >= 400:
        raise ElevenLabsError(f"Realtime token request failed ({resp.status_code}): {resp.text[:400]}")
    return resp.json()


def mock_stt_response(video_path: str, default_text: str = "Hello. This is a fallback smoke transcript.") -> Dict[str, Any]:
    duration = max(get_video_duration_seconds(video_path), 1.0)
    tokens = [t for t in default_text.replace(".", " .").split(" ") if t]
    per = duration / max(len(tokens), 1)

    t = 0.0
    words = []
    for token in tokens:
        end = min(duration, t + per)
        words.append(
            {
                "text": token,
                "start": round(t, 3),
                "end": round(end, 3),
                "speaker_id": "spk_0",
                "type": "word" if token != "." else "punctuation",
            }
        )
        t = end

    return {
        "text": default_text,
        "language_code": "en",
        "words": words,
        "mocked": True,
    }
