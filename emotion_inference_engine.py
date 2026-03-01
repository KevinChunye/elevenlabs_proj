#!/usr/bin/env python3
"""Rule-based multimodal emotion inference using Pucek's taxonomy.

Input JSON schema:
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker_id": "spk1",
      "transcript": "...",
      "audio": {
        "pitch_z": 0.8,
        "energy": 0.7,
        "speaking_rate_wps": 2.9,
        "pause_ratio": 0.08
      },
      "vision": {
        "face": {
          "aus": ["AU6", "AU12"],
          "gaze": "direct",
          "head_movement": "steady"
        },
        "gesture": {
          "posture": "open",
          "hand_motion": "expansive",
          "fidgeting": 0.1
        }
      }
    }
  ]
}

Usage:
  python emotion_inference_engine.py input.json
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from typing import Dict, List, Tuple

EMOTIONS = [
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
]

MODALITY_WEIGHTS = {
    "voice": 0.30,
    "face": 0.30,
    "gesture": 0.15,
    "language": 0.25,
}

EMOTION_DIMS = {
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
}

LEXICONS = {
    "joy": {
        "great",
        "awesome",
        "happy",
        "glad",
        "love",
        "excited",
        "good",
        "nice",
        "amazing",
        "fantastic",
        "yay",
        "wonderful",
    },
    "sadness": {
        "sad",
        "sorry",
        "down",
        "upset",
        "hurt",
        "unhappy",
        "depressed",
        "hopeless",
        "bad",
        "lost",
        "miss",
    },
    "anger": {
        "angry",
        "mad",
        "furious",
        "annoyed",
        "ridiculous",
        "unacceptable",
        "hate",
        "stupid",
        "outrageous",
    },
    "fear": {
        "scared",
        "afraid",
        "terrified",
        "worried",
        "panic",
        "nervous",
        "threat",
        "risk",
    },
    "disgust": {
        "disgusting",
        "gross",
        "nasty",
        "repulsive",
        "sick",
        "ew",
    },
    "surprise": {
        "wow",
        "suddenly",
        "unexpected",
        "surprised",
        "shocked",
        "what",
    },
    "confidence": {
        "definitely",
        "certain",
        "sure",
        "clear",
        "confident",
        "absolutely",
    },
    "anxiety": {
        "anxious",
        "uneasy",
        "tense",
        "concerned",
        "apprehensive",
    },
    "boredom": {
        "boring",
        "whatever",
        "meh",
        "dull",
        "tired",
    },
    "engagement": {
        "interesting",
        "curious",
        "let's",
        "let",
        "focus",
        "explore",
    },
    "confusion": {
        "confused",
        "unclear",
        "maybe",
        "perhaps",
        "not sure",
        "i think",
        "i guess",
        "hmm",
    },
    "frustration": {
        "frustrated",
        "stuck",
        "can't",
        "cannot",
        "difficult",
        "why",
        "again",
    },
}

HEDGES = {"maybe", "perhaps", "possibly", "might", "i think", "i guess", "not sure"}
INTENSIFIERS = {"very", "extremely", "really", "so", "too", "absolutely"}
NEGATIONS = {"not", "never", "no", "none", "cannot", "can't"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def blank_scores() -> Dict[str, float]:
    return {emotion: 0.0 for emotion in EMOTIONS}


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 1e-9:
        return scores
    return {k: clamp(v / max_score, 0.0, 1.0) for k, v in scores.items()}


def get_num(source: dict, keys: List[str], default=None):
    for key in keys:
        if key in source and isinstance(source[key], (int, float)):
            return float(source[key])
    return default


def contains_any(value: str, needles: List[str]) -> bool:
    text = (value or "").lower()
    return any(n in text for n in needles)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def phrase_hit(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def score_language(transcript: str) -> Tuple[Dict[str, float], List[str], bool]:
    scores = blank_scores()
    cues: List[str] = []
    text = (transcript or "").strip().lower()
    if not text:
        return scores, cues, False

    tokens = tokenize(text)
    token_set = set(tokens)

    has_not_sure = phrase_hit(text, "not sure")

    for emotion, words in LEXICONS.items():
        hits = []
        for w in words:
            if " " in w:
                if phrase_hit(text, w):
                    hits.append(w)
            elif w in token_set:
                if emotion == "confidence" and w == "sure" and has_not_sure:
                    continue
                hits.append(w)
        if hits:
            scores[emotion] += 0.18 * len(hits)
            cues.append(f"{emotion} keywords: {', '.join(sorted(set(hits)))}")

    hedge_hits = []
    for h in HEDGES:
        if " " in h:
            if phrase_hit(text, h):
                hedge_hits.append(h)
        elif h in token_set:
            hedge_hits.append(h)
    if hedge_hits:
        scores["confusion"] += 0.25
        scores["anxiety"] += 0.10
        cues.append(f"hedging language: {', '.join(sorted(set(hedge_hits)))}")

    intensifier_count = sum(1 for t in tokens if t in INTENSIFIERS)
    if intensifier_count:
        scores["surprise"] += 0.06 * intensifier_count
        scores["anger"] += 0.04 * intensifier_count
        cues.append(f"intensity words count={intensifier_count}")

    negation_count = sum(1 for t in tokens if t in NEGATIONS)
    if negation_count:
        scores["frustration"] += 0.08 * negation_count
        cues.append(f"negations count={negation_count}")

    exclam = text.count("!")
    if exclam:
        scores["surprise"] += 0.08 * exclam
        scores["engagement"] += 0.05 * exclam
        cues.append(f"exclamation marks={exclam}")

    qmarks = text.count("?")
    if qmarks:
        scores["confusion"] += 0.10 * qmarks
        scores["engagement"] += 0.05 * qmarks
        cues.append(f"question marks={qmarks}")

    return normalize_scores(scores), cues, True


def score_voice(audio: dict) -> Tuple[Dict[str, float], List[str], bool]:
    scores = blank_scores()
    cues: List[str] = []
    if not isinstance(audio, dict) or not audio:
        return scores, cues, False

    pitch_z = get_num(audio, ["pitch_z", "f0_z", "pitch_norm"]) 
    pitch = get_num(audio, ["pitch", "f0", "pitch_mean"]) 
    energy_z = get_num(audio, ["energy_z", "loudness_z"]) 
    energy = get_num(audio, ["energy", "intensity", "rms_energy"]) 
    rate = get_num(audio, ["speaking_rate_wps", "speaking_rate", "speech_rate"]) 
    pause_ratio = get_num(audio, ["pause_ratio", "silence_ratio", "pause_fraction"]) 

    available = any(v is not None for v in [pitch_z, pitch, energy_z, energy, rate, pause_ratio])
    if not available:
        return scores, cues, False

    high_pitch = (pitch_z is not None and pitch_z >= 0.6) or (pitch is not None and pitch >= 220)
    low_pitch = (pitch_z is not None and pitch_z <= -0.6) or (pitch is not None and pitch <= 140)

    high_energy = (energy_z is not None and energy_z >= 0.5) or (energy is not None and energy >= 0.65)
    low_energy = (energy_z is not None and energy_z <= -0.5) or (energy is not None and energy <= 0.35)

    fast_rate = rate is not None and rate >= 2.8
    slow_rate = rate is not None and rate <= 1.8

    long_pauses = pause_ratio is not None and pause_ratio >= 0.25
    short_pauses = pause_ratio is not None and pause_ratio <= 0.10

    if high_pitch and high_energy and fast_rate:
        scores["anger"] += 0.40
        scores["joy"] += 0.30
        scores["surprise"] += 0.25
        scores["engagement"] += 0.25
        cues.append("high pitch + high energy + fast speech")

    if high_pitch and long_pauses:
        scores["anxiety"] += 0.30
        scores["fear"] += 0.20
        cues.append("high pitch with frequent pauses")

    if low_energy and slow_rate and long_pauses:
        scores["sadness"] += 0.45
        scores["boredom"] += 0.35
        cues.append("low energy + slow speech + long pauses")

    if high_energy and short_pauses:
        scores["confidence"] += 0.22
        scores["engagement"] += 0.20
        cues.append("high intensity with few pauses")

    if low_pitch and low_energy:
        scores["sadness"] += 0.18
        cues.append("low pitch and low intensity")

    if fast_rate and long_pauses:
        scores["frustration"] += 0.22
        scores["confusion"] += 0.15
        cues.append("fast bursts with noticeable pausing")

    if slow_rate and short_pauses:
        scores["boredom"] += 0.15
        cues.append("slow monotone with little variation")

    return normalize_scores(scores), cues, True


def _au_present(aus: List[str], target: str) -> bool:
    return target.upper() in {au.upper() for au in aus}


def score_face(face: dict) -> Tuple[Dict[str, float], List[str], bool]:
    scores = blank_scores()
    cues: List[str] = []
    if not isinstance(face, dict) or not face:
        return scores, cues, False

    aus = face.get("aus") if isinstance(face.get("aus"), list) else []
    gaze = str(face.get("gaze", "")).lower()
    head = str(face.get("head_movement", face.get("head_pose", ""))).lower()

    available = bool(aus or gaze or head)
    if not available:
        return scores, cues, False

    if _au_present(aus, "AU6") and _au_present(aus, "AU12"):
        scores["joy"] += 0.55
        cues.append("AU6+AU12 smile pattern")

    if _au_present(aus, "AU1") and _au_present(aus, "AU4") and _au_present(aus, "AU15"):
        scores["sadness"] += 0.55
        cues.append("AU1+AU4+AU15 sadness pattern")

    if _au_present(aus, "AU4") and (
        _au_present(aus, "AU5") or _au_present(aus, "AU7") or _au_present(aus, "AU23")
    ):
        scores["anger"] += 0.50
        scores["frustration"] += 0.20
        cues.append("brow lowerer with tightened eyelids/lips")

    if _au_present(aus, "AU9") or _au_present(aus, "AU10"):
        scores["disgust"] += 0.55
        cues.append("nose wrinkle / upper lip raiser")

    if (_au_present(aus, "AU1") and _au_present(aus, "AU2") and _au_present(aus, "AU5")) or (
        _au_present(aus, "AU26") and _au_present(aus, "AU5")
    ):
        scores["surprise"] += 0.55
        cues.append("raised brows + widened eyes")

    if _au_present(aus, "AU1") and _au_present(aus, "AU2") and _au_present(aus, "AU20"):
        scores["fear"] += 0.50
        scores["anxiety"] += 0.20
        cues.append("fear-linked AU combination")

    if contains_any(gaze, ["direct", "center", "stable"]):
        scores["confidence"] += 0.22
        scores["engagement"] += 0.25
        cues.append("direct gaze")

    if contains_any(gaze, ["avert", "away", "down", "avoid"]):
        scores["anxiety"] += 0.25
        scores["sadness"] += 0.15
        cues.append("gaze aversion")

    if contains_any(gaze, ["scan", "shifting", "dart"]):
        scores["confusion"] += 0.20
        scores["anxiety"] += 0.15
        cues.append("shifting gaze")

    if contains_any(head, ["steady", "upright"]):
        scores["confidence"] += 0.20

    if contains_any(head, ["nod", "forward"]):
        scores["engagement"] += 0.20
        cues.append("affirmative head movement")

    if contains_any(head, ["tilt"]):
        scores["confusion"] += 0.20
        cues.append("head tilt")

    if contains_any(head, ["restless", "jitter", "shake", "rapid"]):
        scores["anxiety"] += 0.20
        scores["frustration"] += 0.15
        cues.append("restless head movement")

    return normalize_scores(scores), cues, True


def score_gesture(gesture: dict) -> Tuple[Dict[str, float], List[str], bool]:
    scores = blank_scores()
    cues: List[str] = []
    if not isinstance(gesture, dict) or not gesture:
        return scores, cues, False

    posture = str(gesture.get("posture", "")).lower()
    hand_motion = str(gesture.get("hand_motion", gesture.get("movement", ""))).lower()
    movement_level = get_num(gesture, ["movement_level", "motion_intensity"], default=None)
    fidgeting = get_num(gesture, ["fidgeting", "fidget_score"], default=None)

    available = bool(posture or hand_motion or movement_level is not None or fidgeting is not None)
    if not available:
        return scores, cues, False

    if contains_any(posture, ["open", "upright", "expanded"]):
        scores["confidence"] += 0.25
        cues.append("open posture")

    if contains_any(posture, ["lean forward", "forward", "attentive"]):
        scores["engagement"] += 0.30
        cues.append("forward-leaning posture")

    if contains_any(posture, ["slouch", "collapsed"]):
        scores["boredom"] += 0.30
        scores["sadness"] += 0.15
        cues.append("slouched posture")

    if contains_any(hand_motion, ["expansive", "open", "illustrative"]):
        scores["engagement"] += 0.22
        scores["confidence"] += 0.15
        cues.append("expansive hand gestures")

    if contains_any(hand_motion, ["still", "minimal", "limited"]):
        scores["boredom"] += 0.20

    if contains_any(hand_motion, ["abrupt", "clench", "jerky"]):
        scores["anger"] += 0.20
        scores["frustration"] += 0.30
        cues.append("abrupt or tense gestures")

    if contains_any(hand_motion, ["self touch", "self-touch"]):
        scores["anxiety"] += 0.25
        cues.append("self-touch gesture")

    if fidgeting is not None:
        if fidgeting >= 0.55:
            scores["anxiety"] += 0.35
            scores["confusion"] += 0.15
            cues.append(f"high fidgeting={fidgeting:.2f}")
        elif fidgeting <= 0.20:
            scores["confidence"] += 0.10

    if movement_level is not None:
        if movement_level >= 0.65:
            scores["engagement"] += 0.15
        elif movement_level <= 0.25:
            scores["boredom"] += 0.15

    return normalize_scores(scores), cues, True


def modality_top(scores: Dict[str, float]) -> Tuple[str, float]:
    label = max(scores, key=scores.get)
    return label, scores[label]


def compute_valence_arousal(fused: Dict[str, float]) -> Tuple[float, float]:
    total = sum(max(v, 0.0) for v in fused.values())
    if total <= 1e-9:
        return 0.0, 0.0

    valence = 0.0
    arousal = 0.0
    for emotion, weight in fused.items():
        if weight <= 0.0:
            continue
        v, a = EMOTION_DIMS[emotion]
        valence += (weight / total) * v
        arousal += (weight / total) * a
    return clamp(valence, -1.0, 1.0), clamp(arousal, 0.0, 1.0)


def blend(prev_scores: Dict[str, float], current_scores: Dict[str, float], alpha: float) -> Dict[str, float]:
    if not prev_scores:
        return current_scores
    out = {}
    for emotion in EMOTIONS:
        out[emotion] = alpha * current_scores[emotion] + (1.0 - alpha) * prev_scores[emotion]
    return out


def infer_segment(segment: dict, state: dict) -> dict:
    transcript = segment.get("transcript", "")
    audio = segment.get("audio", {})

    vision = segment.get("vision", {}) if isinstance(segment.get("vision"), dict) else {}
    face_input = vision.get("face", segment.get("face", {}))
    gesture_input = vision.get("gesture", segment.get("gesture", {}))

    language_scores, language_cues, language_ok = score_language(transcript)
    voice_scores, voice_cues, voice_ok = score_voice(audio)
    face_scores, face_cues, face_ok = score_face(face_input)
    gesture_scores, gesture_cues, gesture_ok = score_gesture(gesture_input)

    modality_results = {
        "language": (language_scores, language_cues, language_ok),
        "voice": (voice_scores, voice_cues, voice_ok),
        "face": (face_scores, face_cues, face_ok),
        "gesture": (gesture_scores, gesture_cues, gesture_ok),
    }

    available_modalities = [m for m, (_, _, ok) in modality_results.items() if ok]
    weight_sum = sum(MODALITY_WEIGHTS[m] for m in available_modalities)

    fused = blank_scores()
    if weight_sum > 0:
        for modality in available_modalities:
            scores, _, _ = modality_results[modality]
            w = MODALITY_WEIGHTS[modality] / weight_sum
            for emotion in EMOTIONS:
                fused[emotion] += w * scores[emotion]

    speaker_id = segment.get("speaker_id") or "__global__"
    prev_scores = state.get(speaker_id, {}).get("scores")
    prev_primary = state.get(speaker_id, {}).get("primary")

    # Per-speaker temporal smoothing to reduce frame jitter.
    smoothed = blend(prev_scores, fused, alpha=0.68)

    top_label = max(smoothed, key=smoothed.get)
    top_score = smoothed[top_label]

    # Stickiness: avoid quick emotion flips when scores are close.
    if prev_primary and prev_primary != top_label:
        prev_score = smoothed.get(prev_primary, 0.0)
        if prev_score + 0.04 >= top_score:
            top_label = prev_primary
            top_score = prev_score

    sorted_scores = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
    second_label, second_score = sorted_scores[1]

    modality_tops = {}
    top_votes = 0
    usable_votes = 0
    for modality in available_modalities:
        scores, _, _ = modality_results[modality]
        m_label, m_score = modality_top(scores)
        modality_tops[modality] = {"label": m_label, "score": round(m_score, 3)}
        if m_score >= 0.20:
            usable_votes += 1
            if m_label == top_label:
                top_votes += 1

    agreement = (top_votes / usable_votes) if usable_votes > 0 else 0.0
    coverage = len(available_modalities) / 4.0
    separation = top_score - second_score

    uncertain_reasons = []
    if top_score < 0.30:
        uncertain_reasons.append("weak multimodal evidence")
    if separation < 0.06:
        uncertain_reasons.append("top emotions are too close")
    if usable_votes >= 2 and agreement < 0.5:
        uncertain_reasons.append("modalities disagree")
    if coverage < 0.5 and top_score < 0.45:
        uncertain_reasons.append("limited modality coverage")

    uncertain = len(uncertain_reasons) > 0

    confidence = clamp(0.45 * top_score + 0.30 * agreement + 0.25 * coverage, 0.0, 1.0)
    if uncertain:
        confidence = min(confidence, 0.49)

    valence, arousal = compute_valence_arousal(smoothed)

    emotions_out = [{"label": top_label, "confidence": round(clamp(top_score, 0.0, 1.0), 3)}]
    if second_score >= 0.25 and separation < 0.18:
        emotions_out.append({"label": second_label, "confidence": round(clamp(second_score, 0.0, 1.0), 3)})

    primary = "uncertain" if uncertain else top_label

    out = {
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker_id": segment.get("speaker_id"),
        "primary_emotion": primary,
        "emotions": emotions_out,
        "confidence": round(confidence, 3),
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "supporting_cues": {
            "voice": voice_cues,
            "face": face_cues,
            "gesture": gesture_cues,
            "language": language_cues,
        },
        "diagnostics": {
            "available_modalities": available_modalities,
            "agreement": round(agreement, 3),
            "coverage": round(coverage, 3),
            "score_separation": round(separation, 3),
            "modality_tops": modality_tops,
        },
        "uncertainty_reason": "; ".join(uncertain_reasons) if uncertain else None,
    }

    state[speaker_id] = {
        "scores": deepcopy(smoothed),
        "primary": top_label,
    }

    return out


def run_inference(data: dict) -> dict:
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Input must contain a list field: segments")

    state = {}
    results = []
    for segment in sorted(segments, key=lambda s: (s.get("start", 0.0), s.get("end", 0.0))):
        if not isinstance(segment, dict):
            continue
        results.append(infer_segment(segment, state))

    return {
        "taxonomy": "Pucek",
        "version": "1.0",
        "segments": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer emotions from multimodal segments.")
    parser.add_argument("input", help="Path to input JSON")
    parser.add_argument("--output", help="Optional output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = run_inference(data)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
