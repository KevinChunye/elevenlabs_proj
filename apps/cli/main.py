from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from packages.core.pipeline import run_pipeline

SAMPLE_MP4_URL = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
load_dotenv()


def _print_segments(segments, n=5):
    for seg in segments[:n]:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        label = seg.get("emotion_label", "unknown")
        conf = seg.get("confidence", 0.0)
        text = (seg.get("transcript", "") or "").strip().replace("\n", " ")
        print(f"- [{start:.2f}-{end:.2f}] {label} (conf={conf:.2f}) :: {text}")


def cmd_run(args):
    if getattr(args, "realtime", False):
        return cmd_realtime(args)
    if not args.video:
        raise SystemExit("run mode requires --video unless --realtime is set")

    result = run_pipeline(
        video_path=args.video,
        output_root=args.output_root,
        taxonomy=args.taxonomy,
        allow_mock_stt=args.allow_mock_stt,
    )

    print(f"run_id: {result['run_id']}")
    if result.get("stt_error"):
        print(f"stt_warning: {result['stt_error']}")
    print(f"transcript: {result['transcript_json']}")
    print(f"emotions: {result['emotions_json']}")
    print(f"viewer: {result['viewer_html']}")
    _print_segments(result.get("segments", []), n=5)


def _download_sample(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    urllib.request.urlretrieve(SAMPLE_MP4_URL, str(target))


def _generate_local_sample(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x360:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=16000",
        "-t",
        "8",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(target),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def cmd_smoke(args):
    smoke_dir = Path(args.output_root) / "_smoke"
    smoke_video = smoke_dir / "sample.mp4"

    try:
        _download_sample(smoke_video)
    except Exception as exc:
        print(f"sample_download_failed: {exc}")
        if not args.video:
            print("Falling back to a locally generated synthetic sample clip.")
            _generate_local_sample(smoke_video)

    video = args.video or str(smoke_video)

    result = run_pipeline(
        video_path=video,
        output_root=args.output_root,
        taxonomy=args.taxonomy,
        allow_mock_stt=True,
    )

    print("smoke_result:")
    if result.get("stt_error"):
        print(f"stt_warning: {result['stt_error']}")
    _print_segments(result.get("segments", []), n=5)
    print(f"viewer: {result['viewer_html']}")


def cmd_realtime(args):
    import uvicorn

    host = args.host
    port = args.port
    print(f"Starting realtime demo server at http://{host}:{port}/realtime")
    uvicorn.run("apps.web.main:app", host=host, port=port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal emotion inference MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Analyze a local video")
    run_p.add_argument("--video", help="Path to local mp4/mov")
    run_p.add_argument("--realtime", action="store_true", help="Launch realtime browser demo instead of batch video analysis")
    run_p.add_argument("--host", default="127.0.0.1")
    run_p.add_argument("--port", type=int, default=8000)
    run_p.add_argument("--taxonomy", default="ekman", choices=["ekman", "plutchik", "pucek"])
    run_p.add_argument("--output-root", default="outputs")
    run_p.add_argument("--allow-mock-stt", action="store_true", default=False)
    run_p.set_defaults(func=cmd_run)

    smoke_p = sub.add_parser("smoke", help="Smoke test with a public sample mp4")
    smoke_p.add_argument("--video", help="Optional local video override")
    smoke_p.add_argument("--taxonomy", default="ekman", choices=["ekman", "plutchik", "pucek"])
    smoke_p.add_argument("--output-root", default="outputs")
    smoke_p.set_defaults(func=cmd_smoke)

    rt_p = sub.add_parser("realtime", help="Run realtime STT browser demo")
    rt_p.add_argument("--host", default="127.0.0.1")
    rt_p.add_argument("--port", type=int, default=8000)
    rt_p.set_defaults(func=cmd_realtime)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
