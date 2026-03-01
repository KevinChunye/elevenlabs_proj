from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .schemas import FusedEmotionSegment, TranscriptSegment, VisionEmotion, VoiceEmotion

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

LANGUAGE_LEXICON = {
    "joy": {"great", "awesome", "happy", "glad", "love", "excellent", "amazing", "fun"},
    "sadness": {"sad", "sorry", "down", "unhappy", "hurt", "bad"},
    "anger": {"angry", "mad", "annoyed", "ridiculous", "unacceptable", "hate"},
    "fear": {"afraid", "scared", "terrified", "threat", "panic"},
    "disgust": {"gross", "disgusting", "nasty", "repulsive"},
    "surprise": {"wow", "unexpected", "shocked", "suddenly"},
    "confidence": {"definitely", "certain", "confident", "sure", "absolutely"},
    "anxiety": {"worried", "nervous", "uneasy", "anxious"},
    "boredom": {"boring", "whatever", "meh", "dull"},
    "engagement": {"interesting", "curious", "let's", "focus", "explore"},
    "confusion": {"maybe", "perhaps", "unclear", "not sure", "i guess", "i think"},
    "frustration": {"stuck", "why", "again", "can't", "cannot", "difficult", "frustrated"},
}

VALENCE_AROUSAL = {
    "joy": (0.8, 0.7),
    "sadness": (-0.8, 0.3),
    "anger": (-0.8, 0.8),
    "fear": (-0.9, 0.8),
    "disgust": (-0.7, 0.6),
    "surprise": (0.1, 0.9),
    "confidence": (0.6, 0.6),
    "anxiety": (-0.7, 0.7),
    "boredom": (-0.4, 0.2),
    "engagement": (0.5, 0.6),
    "confusion": (-0.2, 0.5),
    "frustration": (-0.7, 0.7),
    "neutral": (0.0, 0.4),
}

PUCEK_MAP = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "confidence": "confidence",
    "anxiety": "anxiety",
    "boredom": "boredom",
    "engagement": "engagement",
    "confusion": "confusion",
    "frustration": "frustration",
    "neutral": "engagement",
}

EKMAN_MAP = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "confidence": "joy",
    "anxiety": "fear",
    "boredom": "sadness",
    "engagement": "joy",
    "confusion": "surprise",
    "frustration": "anger",
    "neutral": "neutral",
}

PLUTCHIK_MAP = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "confidence": "trust",
    "anxiety": "fear",
    "boredom": "sadness",
    "engagement": "anticipation",
    "confusion": "surprise",
    "frustration": "anger",
    "neutral": "neutral",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _empty_scores() -> Dict[str, float]:
    return {k: 0.0 for k in CANONICAL}


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    if m <= 1e-9:
        return scores
    return {k: _clamp(v / m, 0.0, 1.0) for k, v in scores.items()}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def language_scores(text: str) -> Tuple[Dict[str, float], List[str]]:
    scores = _empty_scores()
    cues: List[str] = []
    tokens = _tokenize(text)
    tok_set = set(tokens)
    joined = " ".join(tokens)

    for emo, lex in LANGUAGE_LEXICON.items():
        hits = []
        for item in lex:
            if " " in item:
                if re.search(r"\b" + re.escape(item) + r"\b", joined):
                    hits.append(item)
            elif item in tok_set:
                hits.append(item)
        if hits:
            scores[emo] += 0.22 * len(hits)
            cues.append(f"keywords for {emo}: {', '.join(sorted(set(hits)))}")

    if "!" in text:
        scores["surprise"] += 0.15
        cues.append("exclamation intensity")
    if "?" in text:
        scores["confusion"] += 0.18
        cues.append("questioning language")

    if max(scores.values()) < 1e-6:
        scores["neutral"] = 0.2

    return _normalize(scores), cues


def _map_taxonomy(label: str, taxonomy: str) -> str:
    taxonomy = (taxonomy or "ekman").lower()
    if taxonomy == "plutchik":
        return PLUTCHIK_MAP.get(label, label)
    if taxonomy == "pucek":
        return PUCEK_MAP.get(label, label)
    return EKMAN_MAP.get(label, label)


def _weighted_valence_arousal(scores: Dict[str, float]) -> Tuple[float, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 1e-9:
        return 0.0, 0.0
    valence = 0.0
    arousal = 0.0
    for emo, w in scores.items():
        if w <= 0:
            continue
        v, a = VALENCE_AROUSAL[emo]
        valence += (w / total) * v
        arousal += (w / total) * a
    return _clamp(valence, -1.0, 1.0), _clamp(arousal, 0.0, 1.0)


def fuse_segments(
    segments: List[TranscriptSegment],
    voice_items: List[VoiceEmotion],
    vision_items: List[VisionEmotion],
    taxonomy: str = "ekman",
) -> List[FusedEmotionSegment]:
    out: List[FusedEmotionSegment] = []

    # Per-speaker smooth state.
    smooth_state: Dict[str, Dict[str, float]] = {}

    for i, seg in enumerate(segments):
        voice = voice_items[i]
        vision = vision_items[i]
        lang_scores, lang_cues = language_scores(seg.text)

        scores = _empty_scores()
        for emo in CANONICAL:
            scores[emo] += 0.42 * voice.scores.get(emo, 0.0)
            scores[emo] += 0.33 * vision.scores.get(emo, 0.0)
            scores[emo] += 0.25 * lang_scores.get(emo, 0.0)

        speaker = seg.speaker_id or "__global__"
        if speaker in smooth_state:
            prev = smooth_state[speaker]
            for emo in CANONICAL:
                scores[emo] = 0.66 * scores[emo] + 0.34 * prev[emo]

        scores = _normalize(scores)
        smooth_state[speaker] = dict(scores)

        top = max(scores, key=scores.get)
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        agreement = 0
        if voice.label == top:
            agreement += 1
        if vision.label == top:
            agreement += 1
        if max(lang_scores, key=lang_scores.get) == top:
            agreement += 1

        confidence = _clamp(0.55 * top_score + 0.20 * (agreement / 3.0) + 0.25 * max(0.0, top_score - second_score), 0.0, 1.0)
        if top_score < 0.32 or (top_score - second_score) < 0.06:
            confidence = min(confidence, 0.49)

        valence, arousal = _weighted_valence_arousal(scores)

        cues = {
            "voice": list(voice.cues),
            "face": list(vision.face_cues),
            "gesture": list(vision.gesture_cues),
            "language": lang_cues,
        }

        out.append(
            FusedEmotionSegment(
                start=seg.start,
                end=seg.end,
                speaker_id=seg.speaker_id,
                transcript=seg.text,
                emotion_label=_map_taxonomy(top, taxonomy),
                pucek_label=PUCEK_MAP.get(top, "engagement"),
                confidence=round(confidence, 3),
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                cues=cues,
                modality_scores={
                    "voice": {k: round(v, 3) for k, v in voice.scores.items()},
                    "vision": {k: round(v, 3) for k, v in vision.scores.items()},
                    "language": {k: round(v, 3) for k, v in lang_scores.items()},
                    "fused": {k: round(v, 3) for k, v in scores.items()},
                },
            )
        )

    return out
