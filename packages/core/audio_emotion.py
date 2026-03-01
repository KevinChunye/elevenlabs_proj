from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .schemas import TranscriptSegment, VoiceEmotion

CANONICAL = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "confidence",
    "anxiety",
    "boredom",
    "engagement",
    "confusion",
    "frustration",
    "neutral",
]


@dataclass
class _RawAudioStats:
    pitch_hz: float
    energy: float
    speaking_rate_wps: float
    pause_ratio: float


def _read_wav_mono_f32(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise RuntimeError("Expected 16-bit PCM WAV from ffmpeg extraction")

    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, sr


def _estimate_pitch_hz(signal: np.ndarray, sr: int) -> float:
    if signal.size < sr // 4:
        return 0.0
    x = signal - np.mean(signal)
    energy = np.sqrt(np.mean(x * x) + 1e-9)
    if energy < 1e-3:
        return 0.0

    corr = np.correlate(x, x, mode="full")[x.size - 1 :]
    min_lag = max(1, int(sr / 400))
    max_lag = min(len(corr) - 1, int(sr / 50))
    if max_lag <= min_lag:
        return 0.0

    segment = corr[min_lag:max_lag]
    lag = int(np.argmax(segment)) + min_lag
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def _words_per_second(segment: TranscriptSegment) -> float:
    dur = max(segment.end - segment.start, 1e-6)
    count = sum(1 for w in segment.words if w.token_type != "punctuation" and w.text.strip())
    return float(count / dur)


def _pause_ratio(segment: TranscriptSegment) -> float:
    dur = max(segment.end - segment.start, 1e-6)
    speech = 0.0
    for w in segment.words:
        speech += max(0.0, w.end - w.start)
    return float(max(0.0, min(1.0, 1.0 - (speech / dur))))


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr) + 1e-6)
    return [float((v - mean) / std) for v in values]


def _empty_scores() -> Dict[str, float]:
    return {k: 0.0 for k in CANONICAL}


def _scale_scores(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    if m <= 1e-9:
        return scores
    return {k: max(0.0, min(1.0, v / m)) for k, v in scores.items()}


def _classify(stats: _RawAudioStats, nz_pitch: float, nz_energy: float, nz_rate: float) -> VoiceEmotion:
    scores = _empty_scores()
    cues: List[str] = []

    if nz_energy > 0.6 and nz_rate > 0.5:
        scores["engagement"] += 0.35
        scores["confidence"] += 0.30
        scores["joy"] += 0.20
        cues.append("high energy and fast speaking rate")

    if nz_energy > 0.8 and nz_pitch > 0.6:
        scores["anger"] += 0.30
        scores["surprise"] += 0.22
        cues.append("simultaneously high pitch and intensity")

    if nz_pitch > 0.7 and stats.pause_ratio > 0.22:
        scores["anxiety"] += 0.40
        scores["fear"] += 0.25
        cues.append("high pitch with frequent pauses")

    if nz_energy < -0.6 and nz_rate < -0.5:
        scores["sadness"] += 0.40
        scores["boredom"] += 0.35
        cues.append("low energy and slow speaking rate")

    if stats.pause_ratio > 0.32 and nz_rate > 0.35:
        scores["confusion"] += 0.30
        scores["frustration"] += 0.25
        cues.append("fast bursts separated by pauses")

    if stats.pause_ratio < 0.10 and nz_energy > 0.3:
        scores["confidence"] += 0.20
        cues.append("few pauses while maintaining intensity")

    if max(scores.values()) < 0.12:
        scores["neutral"] = 0.35
        cues.append("prosody near baseline")

    scores = _scale_scores(scores)
    label = max(scores, key=scores.get)
    conf = float(scores[label])

    return VoiceEmotion(
        label=label,
        confidence=conf,
        scores=scores,
        cues=cues,
        pitch_hz=float(stats.pitch_hz),
        energy=float(stats.energy),
        speaking_rate_wps=float(stats.speaking_rate_wps),
        pause_ratio=float(stats.pause_ratio),
    )


def analyze_voice_emotions(wav_path: str, segments: List[TranscriptSegment]) -> List[VoiceEmotion]:
    signal, sr = _read_wav_mono_f32(wav_path)

    raw: List[_RawAudioStats] = []
    for seg in segments:
        i0 = max(0, int(seg.start * sr))
        i1 = min(signal.shape[0], int(seg.end * sr))
        chunk = signal[i0:i1]

        if chunk.size == 0:
            pitch = 0.0
            energy = 0.0
        else:
            pitch = _estimate_pitch_hz(chunk, sr)
            energy = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))

        raw.append(
            _RawAudioStats(
                pitch_hz=pitch,
                energy=energy,
                speaking_rate_wps=_words_per_second(seg),
                pause_ratio=_pause_ratio(seg),
            )
        )

    n_pitch = _normalize([r.pitch_hz for r in raw])
    n_energy = _normalize([r.energy for r in raw])
    n_rate = _normalize([r.speaking_rate_wps for r in raw])

    out: List[VoiceEmotion] = []
    for idx, stats in enumerate(raw):
        out.append(_classify(stats, n_pitch[idx], n_energy[idx], n_rate[idx]))
    return out
