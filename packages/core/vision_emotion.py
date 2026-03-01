from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import TranscriptSegment, VisionEmotion

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


def _empty_scores() -> Dict[str, float]:
    return {k: 0.0 for k in CANONICAL}


def _norm_scores(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    if m <= 1e-9:
        return scores
    return {k: max(0.0, min(1.0, v / m)) for k, v in scores.items()}


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


@dataclass
class _Agg:
    frames_total: int = 0
    face_frames: int = 0
    pose_frames: int = 0
    smile_vals: List[float] = field(default_factory=list)
    brow_furrow_vals: List[float] = field(default_factory=list)
    eye_open_vals: List[float] = field(default_factory=list)
    brow_raise_vals: List[float] = field(default_factory=list)
    gesture_vals: List[float] = field(default_factory=list)
    head_motion_vals: List[float] = field(default_factory=list)


def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _classify(agg: _Agg) -> VisionEmotion:
    scores = _empty_scores()
    face_cues: List[str] = []
    gesture_cues: List[str] = []

    smile = _avg(agg.smile_vals)
    furrow = _avg(agg.brow_furrow_vals)
    eye_open = _avg(agg.eye_open_vals)
    brow_raise = _avg(agg.brow_raise_vals)
    gesture = _avg(agg.gesture_vals)
    head_motion = _avg(agg.head_motion_vals)

    face_cov = agg.face_frames / max(agg.frames_total, 1)
    pose_cov = agg.pose_frames / max(agg.frames_total, 1)

    if smile > 0.58:
        scores["joy"] += 0.55
        scores["confidence"] += 0.20
        face_cues.append("smile landmarks consistently elevated")

    if furrow > 0.62:
        scores["anger"] += 0.45
        scores["frustration"] += 0.25
        face_cues.append("brow furrow pattern present")

    if brow_raise > 0.58 and eye_open > 0.5:
        scores["surprise"] += 0.45
        scores["fear"] += 0.22
        face_cues.append("raised brows with wider eyes")

    if eye_open < 0.34 and smile < 0.45:
        scores["sadness"] += 0.30
        scores["boredom"] += 0.28
        face_cues.append("reduced eye openness with low positive affect")

    if head_motion > 0.25 and furrow > 0.55:
        scores["anxiety"] += 0.25
        face_cues.append("restless head movement")

    if gesture > 0.55:
        scores["engagement"] += 0.45
        gesture_cues.append("high hand/upper-body motion intensity")

    if gesture > 0.62 and furrow > 0.6:
        scores["frustration"] += 0.30
        gesture_cues.append("tense motion paired with brow tension")

    if gesture < 0.2 and face_cov > 0.3:
        scores["boredom"] += 0.18
        gesture_cues.append("very limited movement")

    if max(scores.values()) < 0.15:
        scores["neutral"] = 0.35

    # Penalize confidence when detection coverage is weak.
    cov_penalty = 0.8 if (face_cov + pose_cov) < 0.35 else 1.0
    scores = {k: v * cov_penalty for k, v in scores.items()}
    scores = _norm_scores(scores)

    label = max(scores, key=scores.get)
    confidence = scores[label]

    if face_cov < 0.2:
        face_cues.append("limited face detection coverage")
    if pose_cov < 0.2:
        gesture_cues.append("limited pose detection coverage")

    return VisionEmotion(
        label=label,
        confidence=float(confidence),
        scores=scores,
        face_cues=face_cues,
        gesture_cues=gesture_cues,
        smile=float(smile),
        brow_furrow=float(furrow),
        eye_openness=float(eye_open),
        gesture_intensity=float(gesture),
    )


def analyze_vision_emotions(video_path: str, segments: List[TranscriptSegment], sample_fps: float = 6.0) -> List[VisionEmotion]:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except Exception:
        fallback: List[VisionEmotion] = []
        for _ in segments:
            fallback.append(
                VisionEmotion(
                    label="neutral",
                    confidence=0.15,
                    scores={k: (0.35 if k == "neutral" else 0.0) for k in CANONICAL},
                    face_cues=["mediapipe/opencv unavailable"],
                    gesture_cues=["mediapipe/opencv unavailable"],
                    smile=0.0,
                    brow_furrow=0.0,
                    eye_openness=0.0,
                    gesture_intensity=0.0,
                )
            )
        return fallback

    if not segments:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0
    step = max(1, int(round(native_fps / max(sample_fps, 1.0))))

    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    aggs = [_Agg() for _ in segments]

    last_wrists = None
    last_nose = None

    seg_idx = 0
    frame_idx = 0

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            t = frame_idx / native_fps
            frame_idx += 1

            while seg_idx < len(segments) and t > segments[seg_idx].end:
                seg_idx += 1
            if seg_idx >= len(segments):
                break
            if t < segments[seg_idx].start:
                continue

            agg = aggs[seg_idx]
            agg.frames_total += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(rgb)
            pose_res = pose.process(rgb)

            if face_res.multi_face_landmarks:
                agg.face_frames += 1
                lm = face_res.multi_face_landmarks[0].landmark

                mouth_w = _dist(lm[61], lm[291])
                mouth_h = _dist(lm[13], lm[14]) + 1e-6
                smile_ratio = max(0.0, min(1.0, (mouth_w / mouth_h - 1.2) / 2.2))

                inner_brow_dist = _dist(lm[70], lm[300])
                brow_furrow = max(0.0, min(1.0, (0.10 - inner_brow_dist) / 0.08))

                eye_open_l = _dist(lm[159], lm[145])
                eye_open_r = _dist(lm[386], lm[374])
                eye_open = max(0.0, min(1.0, ((eye_open_l + eye_open_r) * 0.5 - 0.015) / 0.04))

                brow_raise_l = max(0.0, lm[159].y - lm[70].y)
                brow_raise_r = max(0.0, lm[386].y - lm[300].y)
                brow_raise = max(0.0, min(1.0, ((brow_raise_l + brow_raise_r) * 0.5 - 0.02) / 0.08))

                agg.smile_vals.append(smile_ratio)
                agg.brow_furrow_vals.append(brow_furrow)
                agg.eye_open_vals.append(eye_open)
                agg.brow_raise_vals.append(brow_raise)

            if pose_res.pose_landmarks:
                agg.pose_frames += 1
                p = pose_res.pose_landmarks.landmark

                wrists = ((p[15].x, p[15].y), (p[16].x, p[16].y))
                nose = (p[0].x, p[0].y)

                if last_wrists is not None:
                    wl = math.sqrt((wrists[0][0] - last_wrists[0][0]) ** 2 + (wrists[0][1] - last_wrists[0][1]) ** 2)
                    wr = math.sqrt((wrists[1][0] - last_wrists[1][0]) ** 2 + (wrists[1][1] - last_wrists[1][1]) ** 2)
                    gesture = max(0.0, min(1.0, ((wl + wr) * 0.5) / 0.03))
                    agg.gesture_vals.append(gesture)

                if last_nose is not None:
                    hn = math.sqrt((nose[0] - last_nose[0]) ** 2 + (nose[1] - last_nose[1]) ** 2)
                    agg.head_motion_vals.append(max(0.0, min(1.0, hn / 0.02)))

                last_wrists = wrists
                last_nose = nose

    cap.release()

    out = [_classify(a) for a in aggs]
    return out
