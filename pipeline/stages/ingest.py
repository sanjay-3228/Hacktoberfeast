"""Stage 1: copy/transcode the video for the web and extract 16 kHz mono audio."""
import json
import os
import shutil
import subprocess
from pathlib import Path


def _executable(name: str, env_name: str) -> str:
    configured = os.environ.get(env_name)
    executable = configured or shutil.which(name)
    if executable:
        return executable
    raise RuntimeError(
        f"{name} is required for video ingest but was not found on PATH. "
        "Install FFmpeg and add its bin directory to PATH, or set "
        f"{env_name} to the full path of {name}."
    )


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def get_duration(path: Path) -> float:
    ffprobe = _executable("ffprobe", "FFPROBE_BIN")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def ingest(video_path: str, job_dir: Path) -> dict:
    """Returns {"video": Path, "audio": Path, "duration": float}."""
    job_dir.mkdir(parents=True, exist_ok=True)
    video = job_dir / "video.mp4"
    audio = job_dir / "audio.wav"

    if not video.exists():
        ffmpeg = _executable("ffmpeg", "FFMPEG_BIN")
        # H.264 + AAC so every browser can play it
        _run([ffmpeg, "-y", "-i", str(video_path),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
              "-movflags", "+faststart", str(video)])
    if not audio.exists():
        ffmpeg = _executable("ffmpeg", "FFMPEG_BIN")
        _run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1",
              "-ar", "16000", str(audio)])

    return {"video": video, "audio": audio, "duration": get_duration(video)}
