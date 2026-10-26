"""Stage 1: copy/transcode the video for the web and extract 16 kHz mono audio."""
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
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
        # H.264 + AAC so every browser can play it
        _run(["ffmpeg", "-y", "-i", str(video_path),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
              "-movflags", "+faststart", str(video)])
    if not audio.exists():
        _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
              "-ar", "16000", str(audio)])

    return {"video": video, "audio": audio, "duration": get_duration(video)}
