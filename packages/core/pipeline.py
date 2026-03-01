from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .audio_emotion import analyze_voice_emotions
from .config import load_config
from .elevenlabs_stt import ElevenLabsError, create_batch_transcript, mock_stt_response, parse_words_from_stt_response
from .ffmpeg_utils import copy_video_to_output, extract_audio_wav, get_video_duration_seconds
from .fusion import fuse_segments
from .schemas import EmotionTimeline, FusedEmotionSegment, TranscriptSegment, TranscriptWord, to_dict
from .segmentation import build_segments
from .viewer import write_viewer_html
from .vision_emotion import analyze_vision_emotions


def _new_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def _fallback_single_segment(payload: Dict[str, Any], duration: float) -> List[TranscriptSegment]:
    text = payload.get("text") or payload.get("transcript") or ""
    if not text:
        text = "(no transcript text returned)"
    return [
        TranscriptSegment(
            start=0.0,
            end=max(0.5, duration),
            text=str(text).strip(),
            speaker_id=None,
            words=[],
        )
    ]


def run_pipeline(
    video_path: str,
    output_root: str = "outputs",
    taxonomy: str = "ekman",
    allow_mock_stt: bool = True,
) -> Dict[str, Any]:
    config = load_config(taxonomy=taxonomy)

    run_id = _new_run_id()
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    video_copy = copy_video_to_output(video_path, str(run_dir))
    audio_path = str(run_dir / "audio.wav")
    extract_audio_wav(video_path, audio_path, sample_rate=16000)

    stt_payload = None
    stt_error = None

    try:
        stt_payload = create_batch_transcript(media_path=audio_path, api_key=config.elevenlabs_api_key, model_id=config.elevenlabs_model_id)
    except Exception as exc:
        stt_error = str(exc)
        if not allow_mock_stt:
            raise
        stt_payload = mock_stt_response(video_path)

    words = parse_words_from_stt_response(stt_payload)
    segments = build_segments(words)
    if not segments:
        segments = _fallback_single_segment(stt_payload, get_video_duration_seconds(video_path))

    voice_items = analyze_voice_emotions(audio_path, segments)
    vision_items = analyze_vision_emotions(video_path, segments, sample_fps=6.0)

    fused_segments = fuse_segments(segments, voice_items, vision_items, taxonomy=taxonomy)

    transcript_path = run_dir / "transcript.json"
    emotions_path = run_dir / "emotions.json"

    transcript_json = {
        "run_id": run_id,
        "source_video": str(Path(video_path).resolve()),
        "taxonomy": taxonomy,
        "stt_error": stt_error,
        "stt_payload": stt_payload,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker_id": s.speaker_id,
                "text": s.text,
                "words": [to_dict(w) for w in s.words],
            }
            for s in segments
        ],
    }
    transcript_path.write_text(json.dumps(transcript_json, indent=2), encoding="utf-8")

    timeline = EmotionTimeline(run_id=run_id, taxonomy=taxonomy, segments=fused_segments)
    emotions_path.write_text(json.dumps(to_dict(timeline), indent=2), encoding="utf-8")

    viewer_path = write_viewer_html(str(run_dir), fused_segments, taxonomy, Path(video_copy).name)

    return {
        "run_id": run_id,
        "output_dir": str(run_dir.resolve()),
        "transcript_json": str(transcript_path.resolve()),
        "emotions_json": str(emotions_path.resolve()),
        "viewer_html": str(Path(viewer_path).resolve()),
        "segments": [to_dict(s) for s in fused_segments],
        "stt_error": stt_error,
    }
