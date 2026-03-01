from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .schemas import FusedEmotionSegment

COLOR_MAP = {
    "joy": "#16a34a",
    "sadness": "#2563eb",
    "anger": "#dc2626",
    "fear": "#7c3aed",
    "disgust": "#059669",
    "surprise": "#d97706",
    "confidence": "#0ea5e9",
    "anxiety": "#9333ea",
    "boredom": "#64748b",
    "engagement": "#0891b2",
    "confusion": "#c2410c",
    "frustration": "#be123c",
    "neutral": "#6b7280",
    "trust": "#0284c7",
    "anticipation": "#0f766e",
}


def _build_html(data: Dict, video_filename: str) -> str:
    data_json = json.dumps(data)
    color_json = json.dumps(COLOR_MAP)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Emotion Timeline Viewer</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 16px; color: #0f172a; background: #f8fafc; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; margin-bottom: 16px; }}
    #rows {{ display: grid; gap: 8px; }}
    .row {{ display: grid; grid-template-columns: 90px 1fr auto; gap: 10px; align-items: center; padding: 8px; border: 1px solid #e2e8f0; border-radius: 8px; cursor: pointer; }}
    .row.active {{ outline: 2px solid #0ea5e9; }}
    .tag {{ color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; }}
    .time {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #334155; }}
    .tline {{ height: 16px; width: 100%; background: #e2e8f0; border-radius: 6px; overflow: hidden; position: relative; }}
    .bar {{ height: 100%; position: absolute; top: 0; border-radius: 4px; opacity: 0.9; }}
    .small {{ font-size: 12px; color: #475569; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h2>Emotion Timeline Viewer</h2>
      <video id=\"player\" controls width=\"100%\" src=\"./{video_filename}\"></video>
      <input id=\"scrubber\" type=\"range\" min=\"0\" max=\"1000\" value=\"0\" style=\"width:100%;margin-top:8px\" />
      <div class=\"small\" id=\"cursor\"></div>
    </div>
    <div class=\"card\">
      <div class=\"small\">Transcript timeline</div>
      <div class=\"tline\" id=\"timeline\"></div>
    </div>
    <div class=\"card\">
      <div id=\"rows\"></div>
    </div>
  </div>

<script>
const data = {data_json};
const colors = {color_json};
const segments = data.segments || [];
const player = document.getElementById('player');
const scrubber = document.getElementById('scrubber');
const rows = document.getElementById('rows');
const timeline = document.getElementById('timeline');
const cursor = document.getElementById('cursor');

function fmt(t) {{
  const s = Math.max(0, t || 0);
  const m = Math.floor(s / 60);
  const r = (s % 60).toFixed(1).padStart(4, '0');
  return `${{m}}:${{r}}`;
}}

const duration = Math.max(1, ...segments.map(s => s.end || 0));

segments.forEach((s, idx) => {{
  const row = document.createElement('div');
  row.className = 'row';
  row.dataset.idx = String(idx);
  const c = colors[s.emotion_label] || '#334155';
  row.innerHTML = `
    <div class=\"time\">${{fmt(s.start)}}-${{fmt(s.end)}}</div>
    <div>
      <div>${{s.transcript || ''}}</div>
      <div class=\"small\">valence=${{s.valence}}, arousal=${{s.arousal}}, conf=${{s.confidence}}</div>
    </div>
    <span class=\"tag\" style=\"background:${{c}}\">${{s.emotion_label}}</span>
  `;
  row.onclick = () => {{
    player.currentTime = s.start || 0;
    player.play().catch(() => {{}});
  }};
  rows.appendChild(row);

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.style.left = `${{(s.start / duration) * 100}}%`;
  bar.style.width = `${{Math.max(0.5, ((s.end - s.start) / duration) * 100)}}%`;
  bar.style.background = c;
  bar.title = `${{s.emotion_label}} (${{fmt(s.start)}}-${{fmt(s.end)}})`;
  bar.onclick = () => {{ player.currentTime = s.start || 0; }};
  timeline.appendChild(bar);
}});

function highlight() {{
  const t = player.currentTime || 0;
  scrubber.value = String(Math.round((t / Math.max(duration, player.duration || duration)) * 1000));
  cursor.textContent = `time=${{fmt(t)}}`;
  const idx = segments.findIndex(s => t >= s.start && t <= s.end);
  document.querySelectorAll('.row').forEach(r => r.classList.remove('active'));
  if (idx >= 0) {{
    const target = document.querySelector(`.row[data-idx=\"${{idx}}\"]`);
    if (target) target.classList.add('active');
  }}
}}

player.addEventListener('timeupdate', highlight);
scrubber.addEventListener('input', () => {{
  const d = Math.max(duration, player.duration || duration);
  player.currentTime = (Number(scrubber.value) / 1000) * d;
  highlight();
}});

highlight();
</script>
</body>
</html>
"""


def write_viewer_html(output_dir: str, segments: List[FusedEmotionSegment], taxonomy: str, video_filename: str) -> str:
    payload = {
        "taxonomy": taxonomy,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker_id": s.speaker_id,
                "transcript": s.transcript,
                "emotion_label": s.emotion_label,
                "pucek_label": s.pucek_label,
                "confidence": s.confidence,
                "valence": s.valence,
                "arousal": s.arousal,
                "cues": s.cues,
            }
            for s in segments
        ],
    }

    path = Path(output_dir) / "viewer.html"
    path.write_text(_build_html(payload, video_filename), encoding="utf-8")
    return str(path)
