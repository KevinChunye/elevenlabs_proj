from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


Taxonomy = str


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    speaker_id: Optional[str] = None
    token_type: Optional[str] = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    words: List[TranscriptWord] = field(default_factory=list)


@dataclass
class VoiceEmotion:
    label: str
    confidence: float
    scores: Dict[str, float]
    cues: List[str]
    pitch_hz: float
    energy: float
    speaking_rate_wps: float
    pause_ratio: float


@dataclass
class VisionEmotion:
    label: str
    confidence: float
    scores: Dict[str, float]
    face_cues: List[str]
    gesture_cues: List[str]
    smile: float
    brow_furrow: float
    eye_openness: float
    gesture_intensity: float


@dataclass
class FusedEmotionSegment:
    start: float
    end: float
    speaker_id: Optional[str]
    transcript: str
    emotion_label: str
    pucek_label: str
    confidence: float
    valence: float
    arousal: float
    cues: Dict[str, List[str]]
    modality_scores: Dict[str, Dict[str, float]]


@dataclass
class EmotionTimeline:
    run_id: str
    taxonomy: str
    segments: List[FusedEmotionSegment]


def to_dict(value):
    return asdict(value)
