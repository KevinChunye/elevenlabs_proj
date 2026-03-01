# Pucek Emotion Inference Engine

This repository contains a rule-based multimodal emotion inference engine that maps time segments to **Pucek taxonomy** labels.

## Taxonomy
- Core: `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`
- Social/complex: `confidence`, `anxiety`, `boredom`, `engagement`, `confusion`, `frustration`
- Dimensional: `valence` in `[-1,1]`, `arousal` in `[0,1]`

## Input
Provide JSON with a `segments` list. Each segment can include:
- `start`, `end`, `speaker_id`
- `transcript`
- `audio` (`pitch_z`, `energy`, `speaking_rate_wps`, `pause_ratio`, ...)
- `vision.face` (`aus`, `gaze`, `head_movement`)
- `vision.gesture` (`posture`, `hand_motion`, `fidgeting`, `movement_level`)

Use [`sample_input.json`](/Users/haochuanwang/Desktop/Research/sundai_hack/audio_mar1/sample_input.json) as a template.

## Run
```bash
python emotion_inference_engine.py sample_input.json
```

Optional file output:
```bash
python emotion_inference_engine.py sample_input.json --output output.json
```

## Output format
Machine-readable JSON:
- `primary_emotion` (`"uncertain"` if weak/conflicting evidence)
- `emotions` (top candidates + confidence)
- `confidence` (0-1)
- `supporting_cues` split by `voice`, `face`, `gesture`, `language`
- `valence`, `arousal`
- `diagnostics` (`agreement`, modality coverage, score separation)
- `uncertainty_reason` when applicable

## Inference behavior
- Uses weighted multimodal fusion (voice/face/language/gesture)
- Prefers cross-modality agreement
- Applies per-speaker temporal smoothing to reduce jitter
- Caps confidence when evidence is weak or contradictory
