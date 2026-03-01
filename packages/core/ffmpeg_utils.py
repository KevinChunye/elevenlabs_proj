from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required binary: {name}. Please install it and retry.")


def get_video_duration_seconds(video_path: str) -> float:
    require_binary("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_audio_wav(video_path: str, wav_path: str, sample_rate: int = 16000) -> str:
    require_binary("ffmpeg")
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path


def copy_video_to_output(video_path: str, output_dir: str) -> str:
    src = Path(video_path)
    ext = src.suffix.lower() or ".mp4"
    dst = Path(output_dir) / f"input_video{ext}"
    dst.write_bytes(src.read_bytes())
    return str(dst)
