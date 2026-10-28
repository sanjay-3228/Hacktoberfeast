"""End-to-end audio-visual speaker diarization pipeline.

Turns a short (2-5 minute) video into a self-contained job folder::

    jobs/{job_id}/
        video.mp4            # web-friendly copy of the input
        audio.wav            # 16 kHz mono PCM
        transcript.json      # faster-whisper output (word timestamps)
        diarization.json     # pyannote speaker turns
        messages.json        # merged speaker-labelled messages
        faces_tracks.json    # face tracks + mouth-openness samples
        faces_match.json     # speaker -> face assignment
        llm.json             # topic / chapters / takeaways
        chunks.json          # ~45 s search chunks + embeddings
        result.json          # final contract served by the backend
        faces/T{n}.jpg       # best crop per face track
        faces/S{n}.jpg       # avatar crop per on-camera speaker

Every stage is file-idempotent: when its output file already exists the stage
is skipped, so a crashed run resumes where it stopped and never repeats heavy
GPU compute. Delete a stage output to force that stage to run again.

Usage::

    python vke_pipeline.py /path/to/clip.mp4 demo1 --hf-token hf_xxx

    from vke_pipeline import process_video
    process_video("/path/to/clip.mp4", "demo1", hf_token="hf_xxx")

Requirements: ffmpeg/ffprobe on PATH, plus faster-whisper, pyannote.audio,
mediapipe, opencv-python-headless, sentence-transformers, numpy and openai.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypedDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_JOBS_DIR = Path(__file__).resolve().parent / "jobs"


def _load_env_file() -> None:
    """Load the nearest .env if python-dotenv is installed (never overrides)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd() / ".env", here / ".env", here.parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


_load_env_file()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
WHISPER_BEAM_SIZE = max(_env_int("WHISPER_BEAM_SIZE", 5), 1)
# "auto" prefers CUDA when CTranslate2 sees a device; "cpu" forces the CPU
# path (useful when the CUDA runtime DLLs are missing).
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
# faster-whisper's batched pipeline decodes VAD chunks in parallel. Measured on
# an RTX-class GPU: ~2x faster on a 3-minute talk (4.1s -> 2.0s) and ~5x on a
# 60s clip, with output equivalent to sequential decoding (see
# docs/PERFORMANCE.md). Set to 0 to force the sequential path.
WHISPER_BATCH_SIZE = max(_env_int("WHISPER_BATCH_SIZE", 8), 0)
# Batched decoding splits audio into VAD chunks of this many seconds. The 30s
# default can merge several utterances into one clip, and on repetitive speech
# the tiny model then drops repeats (measured: 80 words vs 104 at 10-20s).
# 20s keeps repeats and timestamps while still batching well.
WHISPER_CHUNK_LENGTH = min(max(_env_int("WHISPER_CHUNK_LENGTH", 20), 1), 30)

# Decoder penalty against autoregressive token repetition loops.
# Prevents decoder attractor loops on pauses/hesitations while preserving legitimate speech repeats.
WHISPER_REPETITION_PENALTY = max(_env_float("WHISPER_REPETITION_PENALTY", 1.1), 1.0)
WHISPER_NO_REPEAT_NGRAM_SIZE = max(_env_int("WHISPER_NO_REPEAT_NGRAM_SIZE", 0), 0)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
WHISPER_HOTWORDS = os.getenv("WHISPER_HOTWORDS", "").strip() or None
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "").strip() or None

# Transcription language: "auto" enables automatic detection (default);
# ISO codes like "ta" (Tamil), "en" (English) force specific language transcription.
_raw_lang = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()
WHISPER_LANGUAGE: str | None = None if _raw_lang in ("auto", "none", "") else _raw_lang

ENABLE_EMBEDDINGS = _env_bool("ENABLE_EMBEDDINGS", True)
ENABLE_FACE_TRACKING = _env_bool("ENABLE_FACE_TRACKING", True)

DIARIZATION_MODEL = os.getenv(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
).strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()

JOBS_DIR = Path(os.getenv("JOBS_DIR", str(_DEFAULT_JOBS_DIR))).expanduser()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-10-21"
).strip()

MERGE_GAP_SEC = max(_env_float("MERGE_GAP_SEC", 1.0), 0.0)
FACE_FPS = max(_env_float("FACE_FPS", 5.0), 0.5)
FACE_SAMPLE_SEC = 1.0 / FACE_FPS
ON_CAMERA_THRESHOLD = _env_float("ON_CAMERA_THRESHOLD", 0.15)
MIN_TRACK_SAMPLES = max(_env_int("MIN_TRACK_SAMPLES", 8), 1)
TRACK_MAX_GAP_SEC = max(_env_float("TRACK_MAX_GAP_SEC", 2.0), 0.0)
TRACK_IOU_THRESHOLD = min(max(_env_float("TRACK_IOU_THRESHOLD", 0.3), 0.0), 1.0)
CHUNK_SECONDS = max(_env_float("CHUNK_SECONDS", 45.0), 5.0)

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_NATURAL_SPEAKER_RE = re.compile(r"^S(\d+)$")

# ---------------------------------------------------------------------------
# Typed structures (the on-disk data contract)
# ---------------------------------------------------------------------------


class Word(TypedDict):
    """One Whisper word with its own timestamps."""

    word: str
    start: float
    end: float


class TranscriptSegment(TypedDict):
    """One Whisper segment; ``words`` carries the word-level timing."""

    start: float
    end: float
    text: str
    words: list[Word]


class Transcript(TypedDict, total=False):
    """Shape of ``transcript.json``."""

    language: str
    language_mode: str
    detected_language: str
    language_probability: float
    transcription_model: str
    segments: list[TranscriptSegment]


class Turn(TypedDict):
    """One pyannote speaker turn."""

    speaker: str
    start: float
    end: float


class Message(TypedDict):
    """One continuous, speaker-labelled utterance."""

    speaker: str
    start: float
    end: float
    text: str


class TrackSample(TypedDict):
    """Mouth-openness sample of one face track at one instant."""

    t: float
    mouth: float


class FaceTrack(TypedDict):
    """Shape of one entry in ``faces_tracks.json``."""

    track: int
    samples: list[TrackSample]
    crop: str


class SpeakerMeta(TypedDict, total=False):
    """Result of matching one speaker to a face track."""

    on_camera: bool
    avatar: str | None
    score: float
    track: int | None
    visual_status: str


class Chapter(TypedDict):
    """One timestamped chapter."""

    title: str
    start: float
    end: float
    summary: str


class SpeakerOut(TypedDict, total=False):
    """One speaker entry in ``result.json``."""

    id: str
    name: str
    on_camera: bool
    avatar: str | None
    talk_time_pct: float
    visual_status: str


class Result(TypedDict, total=False):
    """Shape of ``result.json`` served to the frontend."""

    topic: str
    duration: float
    speakers: list[SpeakerOut]
    messages: list[Message]
    chapters: list[Chapter]
    takeaways: list[str]
    notes: list[str]
    language: str
    language_mode: str
    detected_language: str
    language_probability: float
    transcription_model: str


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


@contextmanager
def _timed(name: str) -> Iterator[None]:
    """Print how long a pipeline stage takes, even when it raises."""
    start = time.perf_counter()
    print(f"[{name}] started", flush=True)
    try:
        yield
    except BaseException:
        print(f"[{name}] failed after {time.perf_counter() - start:.1f}s", flush=True)
        raise
    else:
        print(f"[{name}] done in {time.perf_counter() - start:.1f}s", flush=True)


def _save_json(obj: Any, path: Path) -> None:
    """Atomically write JSON to disk (never leaves a half-written file).

    A temporary sibling is written, flushed and fsynced, then atomically
    renamed over the target.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    """Read JSON with a clearer error when the file is corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON file: {path}: {exc}") from exc


def _is_non_empty_file(path: Path) -> bool:
    """True when *path* exists, is a file, and has at least one byte."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _free_gpu() -> None:
    """Release Python references and CUDA's cached allocator after a model."""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _cuda_device_count() -> int:
    """CUDA devices visible to CTranslate2 (faster-whisper's own backend).

    faster-whisper runs on CTranslate2, not PyTorch, so CUDA availability is
    detected through ``ctranslate2.get_cuda_device_count()`` rather than
    importing torch just to call ``torch.cuda.is_available()``. Returns 0 when
    CTranslate2 is missing or reports no usable device.
    """
    try:
        import ctranslate2

        return max(int(ctranslate2.get_cuda_device_count()), 0)
    except Exception:
        return 0


def _run_ffmpeg(command: Sequence[str]) -> None:
    """Run FFmpeg with a useful error message, without flooding the log."""
    try:
        subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install FFmpeg and add it to PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or "").strip().splitlines()
        tail = "\n".join(lines[-6:]) if lines else "no stderr output"
        raise RuntimeError(
            f"ffmpeg failed with exit code {exc.returncode}:\n{tail}"
        ) from exc


def _probe_media_details(path: Path) -> dict[str, Any]:
    """Inspect media container and streams in a single ffprobe call.

    Returns:
        {
            "duration": float,
            "width": int,
            "height": int,
            "resolution": str,
            "fps": float,
            "video_codec": str | None,
            "audio_codec": str | None,
            "file_size_bytes": int,
            "file_size_mb": float,
        }
    """
    path = Path(path).resolve()
    file_size = path.stat().st_size if path.is_file() else 0

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_entries",
                "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        meta = json.loads(completed.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install FFmpeg and add it to PATH.") from exc
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        meta = {}

    duration = 0.0
    format_info = meta.get("format", {})
    if "duration" in format_info:
        try:
            d = float(format_info["duration"])
            if math.isfinite(d) and d > 0:
                duration = d
        except (TypeError, ValueError):
            pass

    video_codec = None
    audio_codec = None
    width = 0
    height = 0
    fps = 0.0

    for stream in meta.get("streams", []):
        ctype = stream.get("codec_type")
        cname = stream.get("codec_name")
        if ctype == "video" and video_codec is None:
            video_codec = cname
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            r_fps = stream.get("r_frame_rate", "")
            if "/" in r_fps:
                try:
                    num, den = r_fps.split("/")
                    if float(den) > 0:
                        fps = round(float(num) / float(den), 2)
                except (ValueError, ZeroDivisionError):
                    pass
        elif ctype == "audio" and audio_codec is None:
            audio_codec = cname

    res_str = f"{width}x{height}" if width and height else "unknown"

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "resolution": res_str,
        "fps": fps,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }


def _probe_media(path: Path) -> tuple[str | None, str | None]:
    """Return the first video and audio codec names, or ``(None, None)``."""
    details = _probe_media_details(path)
    return details["video_codec"], details["audio_codec"]


def _remux_to_mp4(source: Path, destination: Path) -> None:
    """Copy H.264/AAC streams from another container into MP4."""
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(destination),
        ]
    )


def _transcode_to_mp4(source: Path, destination: Path) -> None:
    """Re-encode any input into browser-safe H.264/AAC MP4."""
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(destination),
        ]
    )


def _get_duration(path: Path) -> float:
    """Return the exact container duration in seconds via ffprobe."""
    details = _probe_media_details(path)
    duration = details["duration"]
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Video has an invalid duration: {duration!r}")
    return duration


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy *source* to *destination* through a temporary sibling."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if not _is_non_empty_file(temporary):
            raise RuntimeError(f"Copy produced an empty file: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: ingest
# ---------------------------------------------------------------------------


def ingest(video_path: str | Path, job_dir: Path) -> dict[str, Any]:
    """Produce a web-playable ``video.mp4`` and extract 16 kHz mono audio.

    H.264/AAC MP4 inputs are copied byte-for-byte. H.264/AAC streams in other
    containers are remuxed without re-encoding. Anything else is transcoded to
    H.264/AAC, so the served video plays in every browser.

    Args:
        video_path: Input video file.
        job_dir: Destination job folder (created if needed).

    Returns:
        ``{"video": Path, "audio": Path, "duration": float,
        "has_audio": bool}`` where duration is the exact container duration
        reported by ffprobe and has_audio is False when the source had no
        audio stream (a silent placeholder track is generated in that case).

    Raises:
        FileNotFoundError: The source video does not exist.
        RuntimeError: FFmpeg is missing, or it failed to produce output.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    video = job_dir / "video.mp4"
    audio = job_dir / "audio.wav"

    if not _is_non_empty_file(video):
        if source.resolve() == video.resolve():
            # The caller already placed the file inside the job folder.
            pass
        else:
            source_video, source_audio = _probe_media(source)
            # Browsers reliably play H.264 + AAC; copy those MP4s untouched.
            web_safe = source_video == "h264" and source_audio in (None, "aac")
            if source.suffix.lower() == ".mp4" and web_safe:
                _atomic_copy(source, video)
            else:
                temporary = video.with_name(f".{video.name}.tmp")
                try:
                    if web_safe:
                        _remux_to_mp4(source, temporary)
                    else:
                        _transcode_to_mp4(source, temporary)
                    if not _is_non_empty_file(temporary):
                        raise RuntimeError("ffmpeg produced no video output")
                    os.replace(temporary, video)
                except Exception as exc:
                    temporary.unlink(missing_ok=True)
                    if not web_safe:
                        raise
                    print(
                        f"[ingest] remux failed ({exc}); "
                        "re-encoding to H.264/AAC",
                        flush=True,
                    )
                    _transcode_to_mp4(source, temporary)
                    if not _is_non_empty_file(temporary):
                        raise RuntimeError("ffmpeg produced no video output")
                    os.replace(temporary, video)
                finally:
                    temporary.unlink(missing_ok=True)

    has_audio = True
    meta_path = job_dir / "ingest.json"
    if not _is_non_empty_file(audio):
        _video_codec, audio_codec = _probe_media(video)
        temporary = audio.with_name(f".{audio.name}.tmp")
        try:
            if audio_codec is None:
                # Video-only file: there is no speech to extract. Generate a
                # silent 16 kHz mono track so the pipeline layout stays
                # consistent, and report has_audio=False so no stage claims a
                # transcript of speech that never existed.
                silent_duration = _get_duration(video)
                print(
                    "[ingest] no audio stream found; generating a silent "
                    "placeholder track",
                    flush=True,
                )
                _run_ffmpeg(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=mono:sample_rate=16000",
                        "-t",
                        f"{silent_duration:.3f}",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        "-f",
                        "wav",
                        str(temporary),
                    ]
                )
                has_audio = False
            else:
                _run_ffmpeg(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(video),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        "-f",
                        "wav",
                        str(temporary),
                    ]
                )
            if not _is_non_empty_file(temporary):
                raise RuntimeError("ffmpeg produced no audio output")
            os.replace(temporary, audio)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        # Reuse a previously extracted track; prefer the sidecar written on
        # the first run so a silent placeholder is still reported honestly.
        try:
            cached_meta = _load_json(meta_path)
            if isinstance(cached_meta, dict) and "has_audio" in cached_meta:
                has_audio = bool(cached_meta["has_audio"])
        except (OSError, ValueError):
            pass

    duration = _get_duration(video)
    try:
        _save_json({"has_audio": has_audio}, meta_path)
    except OSError:
        pass
    print(
        f"[ingest] duration={duration:.2f}s has_audio={has_audio}", flush=True
    )
    return {
        "video": video,
        "audio": audio,
        "duration": duration,
        "has_audio": has_audio,
    }


# ---------------------------------------------------------------------------
# Stage 2: transcribe
# ---------------------------------------------------------------------------

_whisper_cache: dict[tuple[str, str, str], Any] = {}
_whisper_lock = threading.Lock()
_cuda_available: bool | None = None


def clear_whisper_cache() -> None:
    """Clear cached Whisper models and release GPU/CPU memory."""
    global _cuda_available
    with _whisper_lock:
        _whisper_cache.clear()
        _cuda_available = None
        _free_gpu()


# Bookkeeping for the per-job timing report: how long the most recent model
# load took and whether it came from the process cache.
_model_load_info: dict[str, Any] = {
    "model_load_sec": 0.0,
    "model_cached": False,
}


_dll_dir_handles: list[Any] = []
_nvidia_dll_dirs_added = False


def _add_nvidia_dll_dirs() -> list[str]:
    """Register NVIDIA runtime DLL folders from pip packages (Windows only).

    CTranslate2's Windows wheel links against CUDA 12 runtime DLLs such as
    ``cublas64_12.dll``, which the wheel does not bundle. Those DLLs come from
    the ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` pip packages, whose files
    live under ``site-packages/nvidia/<component>/bin``. Windows does not
    search those directories by default, so they are registered on first use,
    before CTranslate2 performs any CUDA call. Idempotent.

    Returns:
        The directories registered on this call (empty once already done).
    """
    global _nvidia_dll_dirs_added
    if _nvidia_dll_dirs_added or os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    _nvidia_dll_dirs_added = True

    added: list[str] = []
    try:
        import site

        roots: list[str] = []
        try:
            roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            roots.append(site.getusersitepackages())
        except Exception:
            pass
        for root in roots:
            nvidia_dir = Path(root) / "nvidia"
            if not nvidia_dir.is_dir():
                continue
            for bin_dir in sorted(nvidia_dir.glob("*/bin")):
                try:
                    handle = os.add_dll_directory(str(bin_dir))
                    _dll_dir_handles.append(handle)
                except OSError:
                    pass
                # CTranslate2 loads cuBLAS with a plain LoadLibrary call, which
                # only consults PATH (not add_dll_directory), so prepend it too.
                current_path = os.environ.get("PATH", "")
                entry = str(bin_dir)
                if entry.lower() not in current_path.lower():
                    os.environ["PATH"] = entry + os.pathsep + current_path
                added.append(entry)
    except Exception:
        pass
    if added:
        print(
            f"[transcribe] registered NVIDIA runtime DLL dirs: {added}",
            flush=True,
        )
    return added


# Register before any CTranslate2 CUDA call (the import alone is harmless).
_add_nvidia_dll_dirs()


def _mark_cuda_unavailable(reason: str) -> None:
    """Record that CUDA cannot run here and drop any cached CUDA model."""
    global _cuda_available
    with _whisper_lock:
        _cuda_available = False
        for key in [k for k in _whisper_cache if k[1] == "cuda"]:
            _whisper_cache.pop(key, None)
        for key in [k for k in _batched_cache if k[1] == "cuda"]:
            _batched_cache.pop(key, None)
    print(f"[transcribe] CUDA unavailable ({reason}); using CPU", flush=True)


_batched_cache: dict[tuple[str, str, str, int], Any] = {}


def _batch_size_in_use() -> int:
    """Batched decoding is used only when it is configured and supported."""
    if WHISPER_BATCH_SIZE <= 0:
        return 0
    try:
        from faster_whisper import BatchedInferencePipeline  # noqa: F401
    except Exception:
        return 0
    return WHISPER_BATCH_SIZE


def _get_transcriber(
    model_name: str,
    device: str,
    compute_type: str,
) -> tuple[Any, str, str, int]:
    """Return a cached (possibly batched) transcriber plus device details.

    ``BatchedInferencePipeline`` decodes VAD-delimited chunks in parallel. On
    this project's GPU it measured ~2x faster than sequential decoding for a
    3-minute talk (4.07s -> 1.95s) with equivalent output (similarity to a
    ``base``-model reference: 0.850 batched vs 0.847 sequential). It runs the
    same Whisper weights and still yields word timestamps; batch size and
    chunk length are configurable through ``WHISPER_BATCH_SIZE`` and
    ``WHISPER_CHUNK_LENGTH``.
    """
    model, actual_device, actual_compute = _get_whisper_model(
        model_name, device=device, compute_type=compute_type
    )
    batch_size = _batch_size_in_use()
    if batch_size <= 0:
        return model, actual_device, actual_compute, 0

    key = (model_name, actual_device, actual_compute, batch_size)
    with _whisper_lock:
        pipeline = _batched_cache.get(key)
        if pipeline is None:
            from faster_whisper import BatchedInferencePipeline

            pipeline = BatchedInferencePipeline(model=model)
            _batched_cache[key] = pipeline
    return pipeline, actual_device, actual_compute, batch_size


def _get_whisper_model(
    model_name: str,
    device: str,
    compute_type: str,
) -> tuple[Any, str, str]:
    """Retrieve or initialize a cached WhisperModel instance.

    Handles CUDA runtime fallback to CPU, memoizes device availability, and caches
    the initialized model instance across jobs so weights are not re-read from disk.

    A CUDA model can construct successfully and still fail at the first
    inference call when the CUDA runtime DLLs are missing, so the caller must
    treat a failed ``transcribe()`` call as a CUDA failure too; see
    :func:`transcribe`.
    """
    global _cuda_available

    _add_nvidia_dll_dirs()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for transcription "
            "(pip install faster-whisper); torch is not required"
        ) from exc

    target_device = device
    target_compute = compute_type

    if target_device == "cuda" and _cuda_available is False:
        target_device = "cpu"
        target_compute = "int8"

    cache_key = (model_name, target_device, target_compute)
    with _whisper_lock:
        if cache_key in _whisper_cache:
            _model_load_info["model_load_sec"] = 0.0
            _model_load_info["model_cached"] = True
            return _whisper_cache[cache_key], target_device, target_compute

        _model_load_info["model_cached"] = False

        if target_device == "cuda":
            load_start = time.perf_counter()
            try:
                print(
                    f"[transcribe] loading {model_name} on cuda (compute_type={target_compute})",
                    flush=True,
                )
                model = WhisperModel(model_name, device="cuda", compute_type=target_compute)
            except Exception as exc:
                print(
                    f"[transcribe] CUDA model load failed ({exc}); falling back to CPU",
                    flush=True,
                )
                _cuda_available = False
                target_device = "cpu"
                target_compute = "int8"
                cache_key = (model_name, "cpu", "int8")
                if cache_key in _whisper_cache:
                    _model_load_info["model_load_sec"] = round(
                        time.perf_counter() - load_start, 3
                    )
                    _model_load_info["model_cached"] = True
                    return _whisper_cache[cache_key], "cpu", "int8"
            else:
                _cuda_available = True
                _whisper_cache[cache_key] = model
                _model_load_info["model_load_sec"] = round(
                    time.perf_counter() - load_start, 3
                )
                return model, "cuda", target_compute

        load_start = time.perf_counter()
        print(
            f"[transcribe] loading {model_name} on cpu (compute_type={target_compute})",
            flush=True,
        )
        model = WhisperModel(model_name, device="cpu", compute_type=target_compute)
        _whisper_cache[cache_key] = model
        _model_load_info["model_load_sec"] = round(time.perf_counter() - load_start, 3)
        return model, "cpu", target_compute


def _is_misidentified_english(
    detected_lang: str, segments: list[TranscriptSegment]
) -> bool:
    """Detect when auto-detection picked Hindi or Urdu for English speech.

    Whisper base's 30-second acoustic classifier frequently misclassifies South
    Asian accented English as Hindi ('hi') or Urdu ('ur'). In that failure mode,
    the sequence-to-sequence decoder outputs predominantly Latin alphabet words
    interspersed with Urdu/Arabic and Devanagari script fragments, and enters
    pathological repetition loops. If the output text is predominantly Latin or
    mixes Urdu script into Hindi, it is an acoustic misclassification.
    """
    if detected_lang not in ("hi", "ur"):
        return False
    full_text = " ".join(str(s.get("text", "")) for s in segments)
    latin_count = sum(1 for c in full_text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in full_text if c.isalpha())
    if total_alpha == 0:
        return False
    # If 40%+ of alphabetic characters are Latin, or Urdu script is in 'hi':
    has_urdu_in_hi = detected_lang == "hi" and any(
        "\u0600" <= c <= "\u06ff" for c in full_text
    )
    return (latin_count / total_alpha) > 0.4 or has_urdu_in_hi


def transcribe(
    audio_path: Path,
    job_dir: Path,
    timing_info: dict[str, Any] | None = None,
    hotwords: str | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
) -> Transcript:
    """Transcribe 16 kHz mono audio with faster-whisper and save word timestamps.

    Uses a cached WhisperModel instance to eliminate model reload overhead on
    subsequent jobs. If CUDA runtime libraries are missing, falls back cleanly to
    CPU int8 and preserves the model for reuse.

    Args:
        audio_path: 16 kHz mono WAV file.
        job_dir: Job folder; output is ``transcript.json``.
        timing_info: Optional dictionary populated with transcription execution metrics.
        hotwords: Optional vocabulary hints / glossary for proper nouns.
        initial_prompt: Optional initial prompt to guide decoder context.
        language: Optional language code (e.g. "en"). Falls back to ``WHISPER_LANGUAGE``.

    Returns:
        The saved transcript document.
    """
    out_path = Path(job_dir) / "transcript.json"
    if out_path.exists():
        return _load_json(out_path)

    use_cuda = WHISPER_DEVICE != "cpu" and _cuda_device_count() > 0
    device = "cuda" if use_cuda else "cpu"
    compute_type = WHISPER_COMPUTE_TYPE if use_cuda else "int8"

    if language is not None and str(language).strip().lower() in ("auto", "none", ""):
        language = None
    active_language = language if language is not None else WHISPER_LANGUAGE
    active_hotwords = hotwords if hotwords is not None else WHISPER_HOTWORDS
    active_prompt = (
        initial_prompt if initial_prompt is not None else WHISPER_INITIAL_PROMPT
    )

    def _collect_segments(iterator) -> list[TranscriptSegment]:
        collected: list[TranscriptSegment] = []
        for segment in iterator:
            words: list[Word] = []
            for word in segment.words or []:
                try:
                    start = float(word.start)
                    end = float(word.end)
                except (TypeError, ValueError):
                    continue
                text = str(word.word or "")
                if not text:
                    continue
                start = max(0.0, start)
                end = max(start, end)
                # Safeguard: discard degenerate zero-duration consecutive duplicates
                # emitted when a decoder collapses into a single-point attractor loop.
                # Legitimate repeated words ("this, this", "yeah, yeah") span distinct non-zero intervals.
                if (
                    words
                    and text.strip().lower() == words[-1]["word"].strip().lower()
                    and abs(end - start) < 1e-4
                ):
                    continue
                words.append(
                    {"word": text, "start": round(start, 3), "end": round(end, 3)}
                )
            collected.append(
                {
                    "start": round(max(0.0, float(segment.start)), 3),
                    "end": round(max(0.0, float(segment.end)), 3),
                    "text": str(segment.text or "").strip(),
                    "words": words,
                }
            )
        return collected

    transcriber, actual_device, actual_compute, batch_size = _get_transcriber(
        WHISPER_MODEL, device=device, compute_type=compute_type
    )
    model_load_sec = float(_model_load_info.get("model_load_sec", 0.0))
    model_cached = bool(_model_load_info.get("model_cached", False))
    cuda_fallback_reason: str | None = None

    def _run(run_lang: str | None) -> tuple[list[TranscriptSegment], Any]:
        options: dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": WHISPER_BEAM_SIZE,
            "repetition_penalty": WHISPER_REPETITION_PENALTY,
            "condition_on_previous_text": WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        }
        if run_lang is not None:
            options["language"] = run_lang
        if WHISPER_NO_REPEAT_NGRAM_SIZE > 0:
            options["no_repeat_ngram_size"] = WHISPER_NO_REPEAT_NGRAM_SIZE
        if active_hotwords:
            options["hotwords"] = active_hotwords
        if active_prompt:
            options["initial_prompt"] = active_prompt
        if batch_size > 0:
            options["batch_size"] = batch_size
            options["chunk_length"] = WHISPER_CHUNK_LENGTH
        iterator, info_obj = transcriber.transcribe(str(audio_path), **options)
        return _collect_segments(iterator), info_obj

    t_start = time.perf_counter()
    try:
        segments, info = _run(active_language)
    except Exception as exc:
        if actual_device != "cuda":
            raise
        # A CUDA model can construct and then fail on the first real inference
        # call when the CUDA runtime DLLs are missing. Fall back once to CPU.
        cuda_fallback_reason = f"{type(exc).__name__}: {exc}"[:200]
        _mark_cuda_unavailable(cuda_fallback_reason)
        transcriber, actual_device, actual_compute, batch_size = _get_transcriber(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        t_start = time.perf_counter()
        segments, info = _run(active_language)
    t_elapsed = time.perf_counter() - t_start

    detected_lang = str(getattr(info, "language", "") or active_language or "")
    if active_language is None and _is_misidentified_english(detected_lang, segments):
        print(
            f"[transcribe] warning: auto-detected language '{detected_lang}' produced "
            f"predominantly Latin/English text with non-Latin script corruption; "
            f"re-transcribing with language='en'",
            flush=True,
        )
        t_retry_start = time.perf_counter()
        segments, info = _run("en")
        t_elapsed += time.perf_counter() - t_retry_start
        detected_lang = "en"

    audio_duration = float(getattr(info, "duration", 0.0) or 0.0)
    if audio_duration <= 0.0 and segments:
        audio_duration = max(float(s["end"]) for s in segments)

    rtf = (
        round(t_elapsed / max(audio_duration, 0.001), 3)
        if audio_duration > 0
        else 0.0
    )

    final_language = active_language or detected_lang
    language_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    language_mode = "manual" if active_language else "auto"

    if timing_info is not None:
        timing_info["device"] = actual_device
        timing_info["compute_type"] = actual_compute
        timing_info["model"] = WHISPER_MODEL
        timing_info["language"] = final_language
        timing_info["language_mode"] = language_mode
        timing_info["detected_language"] = detected_lang
        timing_info["language_probability"] = round(language_prob, 4)
        timing_info["model_load_sec"] = model_load_sec
        timing_info["model_cached"] = model_cached
        timing_info["batch_size"] = batch_size
        timing_info["chunk_length"] = WHISPER_CHUNK_LENGTH if batch_size > 0 else 0
        timing_info["beam_size"] = WHISPER_BEAM_SIZE
        timing_info["repetition_penalty"] = WHISPER_REPETITION_PENALTY
        timing_info["audio_duration_sec"] = round(audio_duration, 2)
        timing_info["transcription_duration_sec"] = round(t_elapsed, 3)
        timing_info["transcription_rtf"] = rtf
        if cuda_fallback_reason:
            timing_info["cuda_fallback_reason"] = cuda_fallback_reason

    data: Transcript = {
        "language": final_language,
        "language_mode": language_mode,
        "detected_language": detected_lang,
        "language_probability": round(language_prob, 4),
        "transcription_model": WHISPER_MODEL,
        "segments": segments,
    }
    word_count = sum(len(segment["words"]) for segment in segments)
    if word_count == 0:
        print(
            "[transcribe] warning: no word timestamps were produced",
            flush=True,
        )
    _save_json(data, out_path)
    print(
        f"[transcribe] {word_count} word(s) in {len(segments)} segment(s) "
        f"on {actual_device} in {t_elapsed:.2f}s (RTF: {rtf:.3f}x)",
        flush=True,
    )
    return data


# ---------------------------------------------------------------------------
# Stage 3: diarize
# ---------------------------------------------------------------------------


def diarize(
    audio_path: Path,
    job_dir: Path,
    hf_token: str | None = None,
) -> list[Turn]:
    """Run pyannote speaker diarization and save raw speaker turns.

    Handles both pyannote 3.x (``Annotation``) and 4.x (``DiarizeOutput``)
    return shapes, and flushes the GPU as soon as the pipeline is released.

    Args:
        audio_path: 16 kHz mono WAV.
        job_dir: Job folder; output is ``diarization.json``.
        hf_token: Hugging Face token. Falls back to ``HF_TOKEN``.

    Returns:
        Turns sorted by start time, e.g.
        ``[{"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2}, ...]``.

    Raises:
        RuntimeError: No token was provided, pyannote is missing, or the
            model output is unexpected.
    """
    out_path = Path(job_dir) / "diarization.json"
    if out_path.exists():
        return _load_json(out_path)

    token = (hf_token or HF_TOKEN or "").strip()
    if not token:
        raise RuntimeError(
            "pyannote requires a Hugging Face token. Accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and pass "
            "hf_token=... or set HF_TOKEN."
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio and torch are required for diarization"
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:
        # pyannote.audio < 3.2 used `use_auth_token`.
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=token
        )

    try:
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        output = pipeline(str(audio_path))
        annotation = getattr(output, "speaker_diarization", output)
        if not hasattr(annotation, "itertracks"):
            raise RuntimeError(
                "Unexpected pyannote output; expected a speaker Annotation"
            )
        turns: list[Turn] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            start = max(0.0, float(segment.start))
            end = max(start, float(segment.end))
            turns.append(
                {
                    "speaker": str(speaker),
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
    finally:
        del pipeline
        _free_gpu()

    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    _save_json(turns, out_path)
    print(
        f"[diarize] {len(turns)} turn(s) from "
        f"{len({turn['speaker'] for turn in turns})} speaker(s)",
        flush=True,
    )
    return turns


# ---------------------------------------------------------------------------
# Stage 4: merge words + turns into messages
# ---------------------------------------------------------------------------


def _append_word(text: str, word: str) -> str:
    """Append a Whisper token while keeping sensible whitespace."""
    word = str(word)
    if not text:
        return word
    if word[:1] in ".,!?;:%)]}\"'”’":
        return text + word
    if word[:1].isspace() or text[-1:].isspace():
        return text + word
    return f"{text} {word}"


def _words_from_transcript(transcript: Any) -> list[tuple[str, float, float]]:
    """Flatten Whisper segments into ``(word, start, end)`` sorted by time.

    Segments without usable word timings fall back to their segment-level
    timestamps so no spoken text is silently dropped.
    """
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    words: list[tuple[str, float, float]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_text = str(segment.get("text") or "").strip()
        valid = 0
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(word.get("word") or "")
            if not text:
                continue
            start = max(0.0, start)
            end = max(start, end)
            words.append((text, start, end))
            valid += 1
        if valid == 0 and segment_text:
            try:
                start = max(0.0, float(segment["start"]))
                end = max(start, float(segment["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            words.append((segment_text, start, end))
    words.sort(key=lambda item: (item[1], item[2]))
    return words


def _sanitize_turns(raw: Any) -> list[Turn]:
    """Drop malformed diarization entries and sort the rest by time."""
    turns: list[Turn] = []
    if not isinstance(raw, list):
        return turns
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        try:
            start = float(turn["start"])
            end = float(turn["end"])
            speaker = str(turn["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = max(0.0, start)
        if end < start:
            start, end = end, start
            start = max(0.0, start)
        turns.append({"speaker": speaker, "start": start, "end": end})
    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    return turns


def _speaker_at(moment: float, turns: Sequence[Turn]) -> str | None:
    """Return the speaker whose turn contains *moment*, else the nearest one."""
    if not turns:
        return None
    for turn in turns:
        if turn["start"] <= moment <= turn["end"]:
            return turn["speaker"]
    nearest = min(
        turns,
        key=lambda turn: min(
            abs(moment - turn["start"]), abs(moment - turn["end"])
        ),
    )
    return nearest["speaker"]


def merge(job_dir: Path) -> list[Message]:
    """Combine Whisper words and diarization turns into speaker messages.

    Each word is assigned to the speaker whose turn contains the word midpoint.
    Consecutive words from the same speaker are grouped, and adjacent messages
    from the same speaker separated by less than ``MERGE_GAP_SEC`` are joined.
    Speakers are renamed ``S1``, ``S2``, ... by order of first appearance.

    Args:
        job_dir: Job folder containing ``transcript.json`` and
            ``diarization.json``.

    Returns:
        Sorted messages saved to ``messages.json``.

    Raises:
        FileNotFoundError: Transcript or diarization output is missing.
    """
    job_dir = Path(job_dir)
    out_path = job_dir / "messages.json"
    if out_path.exists():
        return _load_json(out_path)

    transcript = _load_json(job_dir / "transcript.json")
    raw_turns = _load_json(job_dir / "diarization.json")
    turns = _sanitize_turns(raw_turns)
    words = _words_from_transcript(transcript)

    if not words:
        print("[merge] warning: transcript contains no usable words", flush=True)
        _save_json([], out_path)
        return []

    labelled: list[tuple[str, str, float, float]] = []
    for text, start, end in words:
        midpoint = (start + end) / 2.0
        speaker = _speaker_at(midpoint, turns) or "UNKNOWN"
        labelled.append((speaker, text, start, end))

    rename: dict[str, str] = {}
    messages: list[Message] = []
    for speaker, text, start, end in labelled:
        if speaker not in rename:
            rename[speaker] = f"S{len(rename) + 1}"
        speaker_id = rename[speaker]

        if (
            messages
            and messages[-1]["speaker"] == speaker_id
            and start - messages[-1]["end"] <= MERGE_GAP_SEC
        ):
            messages[-1]["end"] = round(max(messages[-1]["end"], end), 3)
            messages[-1]["text"] = _append_word(messages[-1]["text"], text)
        else:
            messages.append(
                {
                    "speaker": speaker_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                }
            )

    cleaned: list[Message] = []
    for message in messages:
        text = re.sub(r"\s+", " ", message["text"]).strip()
        if not text:
            continue
        message["text"] = text
        cleaned.append(message)

    _save_json(cleaned, out_path)
    print(
        f"[merge] {len(cleaned)} message(s) for {len(rename)} speaker(s)",
        flush=True,
    )
    return cleaned


# ---------------------------------------------------------------------------
# Stage 5: face tracking with MediaPipe FaceMesh
# ---------------------------------------------------------------------------


@dataclass
class _FaceTrackBuilder:
    """Mutable face track used only while scanning the video."""

    track_id: int
    samples: list[TrackSample] = field(default_factory=list)
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_t: float = -1e12
    best_area: float = 0.0
    best_crop: Any = None


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection_w = ix1 - ix0
    intersection_h = iy1 - iy0
    if intersection_w <= 0 or intersection_h <= 0:
        return 0.0
    intersection = intersection_w * intersection_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _update_track(
    builder: _FaceTrackBuilder,
    detection: dict[str, Any],
    timestamp: float,
    frame: Any,
) -> None:
    """Attach one detection to a track and keep its largest crop."""
    box = detection["box"]
    builder.last_box = box
    builder.last_t = timestamp
    builder.samples.append(
        {"t": round(timestamp, 3), "mouth": round(float(detection["mouth"]), 5)}
    )

    x0, y0, x1, y1 = box
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= builder.best_area:
        return

    height, width = frame.shape[:2]
    face_height = max(y1 - y0, 1.0)
    padding = 0.25 * face_height
    cx0 = int(max(0, math.floor(x0 - padding)))
    cy0 = int(max(0, math.floor(y0 - padding)))
    cx1 = int(min(width, math.ceil(x1 + padding)))
    cy1 = int(min(height, math.ceil(y1 + padding)))
    crop = frame[cy0:cy1, cx0:cx1]
    if getattr(crop, "size", 0):
        builder.best_crop = crop.copy()
        builder.best_area = area


def _patch_mediapipe_windows() -> None:
    """Fix missing 'free' export in Windows libmediapipe.dll."""
    if sys.platform == "win32":
        try:
            import importlib

            c_bind = importlib.import_module(
                "mediapipe.tasks.python.core.mediapipe_c_bindings"
            )
            orig_load_raw = c_bind.load_raw_library

            def patched_load_raw(signatures=()):
                if c_bind._shared_lib is None:
                    lib_path = str(
                        c_bind.resources.files("mediapipe.tasks.c")
                        / "libmediapipe.dll"
                    )
                    c_bind._shared_lib = ctypes.CDLL(lib_path)
                    if not hasattr(c_bind._shared_lib, "free") and hasattr(
                        ctypes.cdll.msvcrt, "free"
                    ):
                        c_bind._shared_lib.free = ctypes.cdll.msvcrt.free
                return orig_load_raw(signatures)

            c_bind.load_raw_library = patched_load_raw
        except Exception:
            pass


class _FaceDetector:
    """Unified face detector using MediaPipe Tasks FaceLandmarker or legacy FaceMesh."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._mesh = None
        self._landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=8,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
        else:
            _patch_mediapipe_windows()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = Path(__file__).parent / "models" / "face_landmarker.task"
            if not model_path.is_file():
                alt = (
                    Path(__file__).resolve().parent.parent
                    / "pipeline"
                    / "models"
                    / "face_landmarker.task"
                )
                if alt.is_file():
                    model_path = alt
                else:
                    raise FileNotFoundError(
                        f"face_landmarker.task model file not found at {model_path}"
                    )

            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=8,
                min_face_detection_confidence=0.5,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(
                options
            )

    def detect(self, frame_bgr: Any) -> list[dict[str, Any]]:
        import cv2
        import mediapipe as mp

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections: list[dict[str, Any]] = []

        if self._mesh is not None:
            result = self._mesh.process(frame_rgb)
            for landmarks in result.multi_face_landmarks or []:
                points = landmarks.landmark
                xs = [point.x * width for point in points]
                ys = [point.y * height for point in points]
                x0 = max(0.0, min(xs))
                x1 = min(float(width), max(xs))
                y0 = max(0.0, min(ys))
                y1 = min(float(height), max(ys))
                face_height = y1 - y0
                if x1 - x0 < 8 or face_height < 8:
                    continue
                mouth = abs(points[13].y - points[14].y) * height / face_height
                detections.append(
                    {
                        "box": (x0, y0, x1, y1),
                        "mouth": min(max(float(mouth), 0.0), 1.0),
                    }
                )
        elif self._landmarker is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb
            )
            result = self._landmarker.detect(mp_image)
            for landmarks in result.face_landmarks or []:
                points = landmarks
                xs = [point.x * width for point in points]
                ys = [point.y * height for point in points]
                x0 = max(0.0, min(xs))
                x1 = min(float(width), max(xs))
                y0 = max(0.0, min(ys))
                y1 = min(float(height), max(ys))
                face_height = y1 - y0
                if x1 - x0 < 8 or face_height < 8:
                    continue
                mouth = abs(points[13].y - points[14].y) * height / face_height
                detections.append(
                    {
                        "box": (x0, y0, x1, y1),
                        "mouth": min(max(float(mouth), 0.0), 1.0),
                    }
                )

        return detections

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass


def _face_deps_available() -> bool:
    """True when OpenCV and MediaPipe (FaceMesh or FaceLandmarker task) are ready."""
    import importlib.util

    if (
        importlib.util.find_spec("cv2") is None
        or importlib.util.find_spec("mediapipe") is None
    ):
        return False

    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return True
        model_path = Path(__file__).parent / "models" / "face_landmarker.task"
        if not model_path.is_file():
            alt = (
                Path(__file__).resolve().parent.parent
                / "pipeline"
                / "models"
                / "face_landmarker.task"
            )
            if alt.is_file():
                model_path = alt
        return model_path.is_file()
    except Exception:
        return False


def track_faces(video_path: Path, job_dir: Path) -> list[FaceTrack]:
    """Detect, track and crop faces while recording mouth openness.

    Samples roughly ``FACE_FPS`` frames per second, runs MediaPipe FaceMesh on
    each sampled frame, links detections across frames by greedy IoU matching,
    saves the highest-resolution crop of every surviving track as
    ``faces/T{n}.jpg`` and records a normalized inner-lip distance per sample.

    Args:
        video_path: The job's ``video.mp4``.
        job_dir: Job folder; output is ``faces_tracks.json``.

    Returns:
        ``[{"track": 0, "samples": [{"t": 1.2, "mouth": 0.03}, ...],
        "crop": "faces/T0.jpg"}, ...]``.

    Notes:
        If OpenCV/MediaPipe are not installed the stage degrades gracefully:
        it logs a warning, writes an empty track list and every speaker is
        later marked off-camera. Delete ``faces_tracks.json`` to retry.
    """
    job_dir = Path(job_dir)
    out_path = job_dir / "faces_tracks.json"
    if out_path.exists():
        cached = _load_json(out_path)
        if isinstance(cached, list) and all(
            isinstance(track, dict)
            and isinstance(track.get("crop"), str)
            and (job_dir / track["crop"]).is_file()
            for track in cached
        ):
            return cached
        print("[faces] cached tracks are incomplete; recomputing", flush=True)

    if not ENABLE_FACE_TRACKING:
        print(
            "[faces] face tracking disabled by configuration; "
            "speaker visual presence will be unverified",
            flush=True,
        )
        _save_json([], out_path)
        return []

    if not _face_deps_available():
        print(
            "[faces] warning: face dependencies unavailable; "
            "speaker visual presence will be unverified",
            flush=True,
        )
        _save_json([], out_path)
        return []

    try:
        import cv2

        detector = _FaceDetector()
    except Exception as exc:
        print(
            f"[faces] warning: face detector failed to initialize ({exc}); "
            "speaker visual presence will be unverified",
            flush=True,
        )
        _save_json([], out_path)
        return []

    faces_dir = job_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f"Cannot open video for face tracking: {video_path}")
    try:
        # Respect phone rotation metadata when this OpenCV build supports it.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_step = max(int(round(fps / FACE_FPS)), 1)

    builders: list[_FaceTrackBuilder] = []
    next_track_id = 0
    frame_index = 0
    try:
        while True:
            # grab() decodes the frame without the colour conversion/retrieve
            # work needed for MediaPipe; sampled frames are read in full.
            if frame_index % frame_step != 0:
                if not capture.grab():
                    break
                frame_index += 1
                continue

            ok, frame = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps
            detections = detector.detect(frame)

            # Greedy one-to-one IoU association, strongest overlap first.
            candidates: list[tuple[float, int, _FaceTrackBuilder]] = []
            for detection_index, detection in enumerate(detections):
                for builder in builders:
                    if timestamp - builder.last_t > TRACK_MAX_GAP_SEC:
                        continue
                    score = _iou(detection["box"], builder.last_box)
                    if score >= TRACK_IOU_THRESHOLD:
                        candidates.append((score, detection_index, builder))
            candidates.sort(key=lambda item: item[0], reverse=True)

            used_detections: set[int] = set()
            used_tracks: set[int] = set()
            for _score, detection_index, builder in candidates:
                if (
                    detection_index in used_detections
                    or builder.track_id in used_tracks
                ):
                    continue
                used_detections.add(detection_index)
                used_tracks.add(builder.track_id)
                _update_track(
                    builder, detections[detection_index], timestamp, frame
                )

            for detection_index, detection in enumerate(detections):
                if detection_index in used_detections:
                    continue
                builder = _FaceTrackBuilder(track_id=next_track_id)
                next_track_id += 1
                builders.append(builder)
                _update_track(builder, detection, timestamp, frame)

            frame_index += 1
    finally:
        capture.release()
        try:
            detector.close()
        except Exception:
            pass

    tracks: list[FaceTrack] = []
    for builder in builders:
        if (
            len(builder.samples) < MIN_TRACK_SAMPLES
            or builder.best_crop is None
        ):
            continue
        relative = f"faces/T{builder.track_id}.jpg"
        destination = job_dir / relative
        written = cv2.imwrite(
            str(destination),
            builder.best_crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not written:
            print(f"[faces] warning: could not write {destination}", flush=True)
            continue
        tracks.append(
            {
                "track": builder.track_id,
                "samples": builder.samples,
                "crop": relative,
            }
        )

    _save_json(tracks, out_path)
    print(
        f"[faces] kept {len(tracks)} track(s) from "
        f"{len(builders)} candidate(s)",
        flush=True,
    )
    return tracks


# ---------------------------------------------------------------------------
# Stage 6: match speakers to face tracks
# ---------------------------------------------------------------------------


def _speech_vector(messages: Sequence[Message], speaker: str, times: Any) -> Any:
    """Binary speech-activity signal for one speaker over the time grid."""
    import numpy as np

    vector = np.zeros_like(times, dtype=float)
    for message in messages:
        if message["speaker"] == speaker:
            vector[
                (times >= message["start"]) & (times <= message["end"])
            ] = 1.0
    return vector


def _mouth_activity(track: FaceTrack, times: Any) -> tuple[Any, Any]:
    """Rolling standard deviation of mouth openness, plus a presence mask."""
    import numpy as np

    samples = track.get("samples") or []
    if not samples:
        return (
            np.zeros_like(times, dtype=float),
            np.zeros_like(times, dtype=bool),
        )

    sample_times = np.array(
        [float(sample["t"]) for sample in samples], dtype=float
    )
    mouth = np.array(
        [float(sample["mouth"]) for sample in samples], dtype=float
    )
    indices = np.clip(
        np.searchsorted(times, sample_times), 0, len(times) - 1
    )
    signal = np.zeros_like(times, dtype=float)
    present = np.zeros_like(times, dtype=bool)
    signal[indices] = mouth
    present[indices] = True

    window = max(int(round(0.6 * FACE_FPS)), 2)
    half = window // 2
    activity = np.zeros_like(times, dtype=float)
    for index in np.flatnonzero(present):
        lo = max(0, int(index) - half)
        hi = min(len(times), int(index) + half + 1)
        values = signal[lo:hi][present[lo:hi]]
        if values.size > 1:
            activity[index] = float(values.std())
    return activity, present


def _pearson(a: Any, b: Any, mask: Any) -> float:
    """Pearson correlation of two signals on the masked samples."""
    import numpy as np

    x = a[mask]
    y = b[mask]
    if x.size < 5:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(float((x * x).sum()) * float((y * y).sum())))
    if denominator < 1e-9:
        return 0.0
    return float((x * y).sum() / denominator)


def match_speakers_to_faces(
    messages: list[Message],
    tracks: list[FaceTrack],
    job_dir: Path,
) -> dict[str, SpeakerMeta]:
    """Assign at most one face track to each speaker by mouth correlation.

    A face whose mouth moves while a speaker talks belongs to that speaker.
    Each track's rolling mouth-movement signal is correlated with each
    speaker's binary speech-activity signal across the whole video. Assignments
    are made greedily, best score first, one face per speaker. Scores below
    ``ON_CAMERA_THRESHOLD`` leave the speaker off-camera with a null avatar.

    Args:
        messages: Merged speaker-labelled messages.
        tracks: Face tracks from :func:`track_faces`.
        job_dir: Job folder; winning crops are copied to ``faces/S{n}.jpg``
            and the assignment is cached in ``faces_match.json``.

    Returns:
        ``{"S1": {"on_camera": True, "avatar": "faces/S1.jpg",
        "score": 0.61, "track": 0}, ...}``.
    """
    job_dir = Path(job_dir)
    out_path = job_dir / "faces_match.json"
    speakers = _speakers_in_order(messages)

    if out_path.exists():
        cached = _load_json(out_path)
        if isinstance(cached, dict) and all(
            speaker in cached
            and (
                not cached[speaker].get("on_camera")
                or (
                    isinstance(cached[speaker].get("avatar"), str)
                    and (job_dir / cached[speaker]["avatar"]).is_file()
                )
            )
            for speaker in speakers
        ):
            return cached
        print("[match] cached face match is incomplete; recomputing", flush=True)

    deps_ok = _face_deps_available() and ENABLE_FACE_TRACKING
    default_status = "off_camera" if deps_ok else "unverified"
    metadata: dict[str, SpeakerMeta] = {
        speaker: {
            "on_camera": False,
            "avatar": None,
            "score": 0.0,
            "track": None,
            "visual_status": default_status,
        }
        for speaker in speakers
    }
    if not speakers or not tracks:
        _save_json(metadata, out_path)
        return metadata

    import numpy as np

    ends = [float(message["end"]) for message in messages]
    ends.extend(
        float(sample["t"])
        for track in tracks
        for sample in track.get("samples") or []
    )
    duration = max(ends) if ends else FACE_SAMPLE_SEC
    times = np.arange(0.0, duration + FACE_SAMPLE_SEC, FACE_SAMPLE_SEC)

    speech = {
        speaker: _speech_vector(messages, speaker, times)
        for speaker in speakers
    }
    scores: dict[tuple[str, int], float] = {}
    for track in tracks:
        activity, present = _mouth_activity(track, times)
        if int(present.sum()) < MIN_TRACK_SAMPLES:
            continue
        for speaker in speakers:
            scores[(speaker, int(track["track"]))] = _pearson(
                activity, speech[speaker], present
            )

    used_speakers: set[str] = set()
    used_tracks: set[int] = set()
    crops = {int(track["track"]): track["crop"] for track in tracks}

    for (speaker, track_id), score in sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    ):
        if score < ON_CAMERA_THRESHOLD:
            break
        if speaker in used_speakers or track_id in used_tracks:
            continue
        source = job_dir / crops[track_id]
        if not source.is_file():
            continue
        relative = f"faces/{speaker}.jpg"
        destination = job_dir / relative
        try:
            shutil.copyfile(source, destination)
        except OSError as exc:
            print(
                f"[match] warning: could not copy {source} -> {destination}: "
                f"{exc}",
                flush=True,
            )
            continue
        metadata[speaker] = {
            "on_camera": True,
            "avatar": relative,
            "score": round(float(score), 3),
            "track": track_id,
            "visual_status": "verified",
        }
        used_speakers.add(speaker)
        used_tracks.add(track_id)

    best_speaker_score: dict[str, float] = {speaker: 0.0 for speaker in speakers}
    for (speaker, track_id), score in scores.items():
        if score > best_speaker_score[speaker]:
            best_speaker_score[speaker] = float(score)

    for speaker in speakers:
        if speaker in used_speakers:
            continue
        best_sc = best_speaker_score.get(speaker, 0.0)
        # Uncertain association when some correlation was detected but below threshold
        visual_status = "unverified" if best_sc >= 0.15 else "off_camera"
        metadata[speaker] = {
            "on_camera": False,
            "avatar": None,
            "score": round(max(0.0, best_sc), 3),
            "track": None,
            "visual_status": visual_status,
        }

    _save_json(metadata, out_path)
    on_camera = sorted(used_speakers, key=_speaker_sort_key)
    print(
        f"[match] on camera: {', '.join(on_camera) if on_camera else 'nobody'}",
        flush=True,
    )
    return metadata


# ---------------------------------------------------------------------------
# Stage 7: LLM analysis
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You analyse transcripts of short videos. Reply with a single JSON object "
    "only: no markdown, no code fences, no commentary. Never invent facts that "
    "are not present in the transcript."
)

_LLM_PROMPT = """Transcript (format: [speaker id, start seconds] text):
{transcript}

Video duration: {duration:.1f} seconds.

Return exactly this JSON shape:
{{
  "topic": "2-3 sentences explaining what the video is about",
  "chapters": [
    {{"title": "short title", "start": 0.0, "end": 42.0, "summary": "1-2 sentences"}}
  ],
  "takeaways": ["short key takeaway", "..."]
}}

Rules:
- chapters: 3 to 8 chapters, sorted, covering 0 seconds to {duration:.1f}
  seconds with no gaps or overlaps. Split where the topic changes.
- takeaways: 5 to 7 short, concrete points.
- Use only information stated in the transcript.
"""


def _llm_configured() -> bool:
    """Whether enough credentials exist to attempt an LLM call."""
    if (
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_DEPLOYMENT
    ):
        return True
    if not LLM_MODEL:
        return False
    return bool(LLM_API_KEY or LLM_BASE_URL or os.getenv("OPENAI_API_KEY"))


def _call_llm(system: str, user: str) -> str:
    """Call an OpenAI-compatible endpoint (or Azure OpenAI) and return text."""
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for LLM analysis"
        ) from exc

    if (
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_DEPLOYMENT
    ):
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        model = AZURE_OPENAI_DEPLOYMENT
    else:
        options: dict[str, Any] = {
            "api_key": LLM_API_KEY
            or os.getenv("OPENAI_API_KEY")
            or "not-needed",
        }
        if LLM_BASE_URL:
            options["base_url"] = LLM_BASE_URL
        client = OpenAI(**options)
        model = LLM_MODEL

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise ValueError("LLM returned an empty response")
    return str(content)


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object even if it is fenced or surrounded by prose.

    Raises:
        ValueError: No JSON object can be decoded.
    """
    cleaned = re.sub(r"```(?:json)?", "", str(text)).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in LLM response")
    try:
        parsed, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        # A truncated object is common; try the last closing brace once.
        end = cleaned.rfind("}")
        if end <= start:
            raise ValueError(f"invalid JSON from LLM: {exc}") from exc
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as inner:
            raise ValueError(f"invalid JSON from LLM: {inner}") from inner
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


def _clean_chapters(raw: Any, duration: float) -> list[Chapter]:
    """Clamp, sort and make chapters contiguous across ``[0, duration]``."""
    duration = max(float(duration), 0.0)
    if not isinstance(raw, list):
        raw = []

    chapters: list[Chapter] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = min(max(start, 0.0), duration)
        end = min(max(end, start), duration)
        title = str(item.get("title") or "").strip() or (
            f"Chapter {len(chapters) + 1}"
        )
        summary = str(item.get("summary") or "").strip()
        chapters.append(
            {
                "title": title,
                "start": round(start, 2),
                "end": round(end, 2),
                "summary": summary,
            }
        )

    chapters.sort(key=lambda chapter: (chapter["start"], chapter["end"]))
    if not chapters or duration <= 0:
        return [
            {
                "title": "Full video",
                "start": 0.0,
                "end": round(duration, 2),
                "summary": "",
            }
        ]

    chapters[0]["start"] = 0.0
    for index, chapter in enumerate(chapters):
        boundary = (
            chapters[index + 1]["start"]
            if index + 1 < len(chapters)
            else duration
        )
        chapter["end"] = round(max(chapter["start"], boundary), 2)

    chapters = [
        chapter for chapter in chapters if chapter["end"] > chapter["start"]
    ]
    if not chapters:
        return [
            {
                "title": "Full video",
                "start": 0.0,
                "end": round(duration, 2),
                "summary": "",
            }
        ]
    chapters[-1]["end"] = round(duration, 2)
    return chapters


def _sanitize_llm(data: Any, duration: float) -> dict[str, Any]:
    """Coerce an LLM JSON object into the expected analysis shape."""
    if not isinstance(data, dict):
        raise ValueError("LLM JSON is not an object")
    topic = data.get("topic", "")
    topic = topic.strip() if isinstance(topic, str) else str(topic or "").strip()

    takeaways: list[str] = []
    raw_takeaways = data.get("takeaways", [])
    if isinstance(raw_takeaways, list):
        for item in raw_takeaways:
            if isinstance(item, str) and item.strip():
                takeaways.append(item.strip())

    return {
        "topic": topic,
        "chapters": _clean_chapters(data.get("chapters"), duration),
        "takeaways": takeaways,
    }


def _llm_fallback(duration: float) -> dict[str, Any]:
    """Analysis used when the LLM is unavailable or keeps returning garbage."""
    return {
        "topic": "",
        "chapters": _clean_chapters([], duration),
        "takeaways": [],
    }


def llm_analysis(messages: list[Message]) -> dict[str, Any]:
    """Ask the LLM for the topic, chapters and takeaways.

    The transcript is formatted as ``[S1 12.4s] text``. The model is asked for
    strict JSON; markdown fences and prose are stripped before parsing, and a
    second attempt is made if the first response is unusable. Any failure
    degrades to an empty topic, one full-video chapter and no takeaways, so the
    pipeline still produces a valid result.

    Args:
        messages: Merged speaker-labelled messages.

    Returns:
        ``{"topic": str, "chapters": [...], "takeaways": [...]}``.
    """
    duration = max(
        (float(message["end"]) for message in messages), default=0.0
    )
    fallback = _llm_fallback(duration)

    if not messages:
        print("[llm] no messages; skipping LLM analysis", flush=True)
        return fallback
    if not _llm_configured():
        print(
            "[llm] no LLM credentials configured; using fallback analysis",
            flush=True,
        )
        return fallback

    transcript = "\n".join(
        f"[{message['speaker']} {float(message['start']):.1f}s] "
        f"{message['text']}"
        for message in messages
    )
    base_prompt = _LLM_PROMPT.format(
        transcript=transcript, duration=duration
    )

    for attempt in range(1, 3):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                "\n\nIMPORTANT: your previous reply was not valid JSON. "
                "Return ONLY the JSON object, with no prose or fences."
            )
        try:
            raw = _call_llm(_LLM_SYSTEM, prompt)
            return _sanitize_llm(extract_json(raw), duration)
        except Exception as exc:
            print(f"[llm] attempt {attempt} failed: {exc}", flush=True)
            if attempt == 1:
                time.sleep(1.0)

    print(
        "[llm] using fallback analysis (topic/chapters/takeaways unavailable)",
        flush=True,
    )
    return fallback


# ---------------------------------------------------------------------------
# Stage 8: search index
# ---------------------------------------------------------------------------


def _make_chunks(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Group messages into roughly ``CHUNK_SECONDS``-long search chunks."""
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for message in messages:
        speech = max(
            0.0, float(message["end"]) - float(message["start"])
        )
        if current is None:
            current = {
                "start": float(message["start"]),
                "end": float(message["end"]),
                "texts": [message["text"]],
                "speech": {message["speaker"]: speech},
            }
        else:
            current["end"] = max(
                float(current["end"]), float(message["end"])
            )
            current["texts"].append(message["text"])
            current["speech"][message["speaker"]] = (
                current["speech"].get(message["speaker"], 0.0) + speech
            )
        if current["end"] - current["start"] >= CHUNK_SECONDS:
            chunks.append(current)
            current = None

    if current is not None:
        chunks.append(current)

    output: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        dominant = (
            max(chunk["speech"].items(), key=lambda item: item[1])[0]
            if chunk["speech"]
            else ""
        )
        output.append(
            {
                "id": index,
                "start": round(chunk["start"], 2),
                "end": round(chunk["end"], 2),
                "speaker": dominant,
                "text": " ".join(chunk["texts"]).strip(),
                "embedding": None,
            }
        )
    return output


_embedder_cache: Any = None
_embedder_failed: bool = False
_embedder_lock = threading.Lock()


def clear_embedder_cache() -> None:
    """Clear cached embedding model and reset failure status."""
    global _embedder_cache, _embedder_failed
    with _embedder_lock:
        _embedder_cache = None
        _embedder_failed = False
        _free_gpu()


def _embedder_unavailable_reason() -> str | None:
    """Cheaply check the embedding stack's declared dependencies.

    Importing ``sentence_transformers`` on a broken install can take several
    seconds before it raises (transformers validates its metadata on import),
    so a missing hard dependency is detected here first. Returns a short
    human-readable reason, or ``None`` when the stack looks importable.
    """
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        return "sentence-transformers is not installed"
    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed (required by sentence-transformers)"
    try:
        import importlib.metadata as metadata

        metadata.version("safetensors")
    except Exception:
        return (
            "the 'safetensors' package that transformers requires is "
            "missing or has no metadata"
        )
    return None


def _get_embedder(model_name: str) -> Any | None:
    """Lazily load and cache the SentenceTransformer model once per process."""
    global _embedder_cache, _embedder_failed

    if not ENABLE_EMBEDDINGS:
        return None

    if _embedder_cache is not None or _embedder_failed:
        return _embedder_cache

    with _embedder_lock:
        if _embedder_cache is not None or _embedder_failed:
            return _embedder_cache
        reason = _embedder_unavailable_reason()
        if reason is not None:
            _embedder_failed = True
            print(
                f"[index] warning: embeddings unavailable ({reason}); "
                "chunks saved without embeddings",
                flush=True,
            )
            return None
        try:
            from sentence_transformers import SentenceTransformer

            print(f"[index] loading embedding model {model_name}", flush=True)
            _embedder_cache = SentenceTransformer(model_name)
        except Exception as exc:
            _embedder_failed = True
            print(
                f"[index] warning: embedding failed ({exc}); chunks saved without embeddings",
                flush=True,
            )
        return _embedder_cache


def build_index(messages: list[Message], job_dir: Path) -> list[dict[str, Any]]:
    """Build ~45 s search chunks and embed them with sentence-transformers.

    Embeddings are optional: if disabled by configuration or if the model cannot
    be loaded (missing distribution, no internet, OOM), chunks are saved with
    ``embedding: null`` and the backend falls back to keyword search.

    Args:
        messages: Merged speaker-labelled messages.
        job_dir: Job folder; output is ``chunks.json``.

    Returns:
        The saved chunk list.
    """
    job_dir = Path(job_dir)
    out_path = job_dir / "chunks.json"
    if out_path.exists():
        cached = _load_json(out_path)
        if isinstance(cached, list) and (
            not ENABLE_EMBEDDINGS
            or any(chunk.get("embedding") is not None for chunk in cached)
            or not cached
        ):
            return cached

    chunks = _make_chunks(messages)
    if chunks:
        if not ENABLE_EMBEDDINGS:
            print(
                "[index] embeddings disabled by configuration; chunks saved without embeddings",
                flush=True,
            )
            for chunk in chunks:
                chunk["embedding"] = None
        else:
            model = _get_embedder(EMBED_MODEL)
            if model is not None:
                try:
                    vectors = model.encode(
                        [chunk["text"] for chunk in chunks],
                        normalize_embeddings=True,
                        batch_size=32,
                        show_progress_bar=False,
                    )
                    for chunk, vector in zip(chunks, vectors):
                        chunk["embedding"] = [
                            round(float(value), 5) for value in vector
                        ]
                except Exception as exc:
                    print(
                        f"[index] warning: encoding failed ({exc}); chunks saved without embeddings",
                        flush=True,
                    )
                    for chunk in chunks:
                        chunk["embedding"] = None
            else:
                for chunk in chunks:
                    chunk["embedding"] = None

    _save_json(chunks, out_path)
    return chunks


# ---------------------------------------------------------------------------
# Stage 9: assemble result.json
# ---------------------------------------------------------------------------


def _speakers_in_order(messages: Iterable[Message]) -> list[str]:
    """Speaker ids in order of first appearance."""
    seen: list[str] = []
    for message in messages:
        speaker = message["speaker"]
        if speaker not in seen:
            seen.append(speaker)
    return seen


def _speaker_sort_key(speaker: str) -> tuple[int, str]:
    """Natural ordering so S2 comes before S10."""
    match = _NATURAL_SPEAKER_RE.fullmatch(speaker)
    if match:
        return int(match.group(1)), speaker
    return 10**9, speaker


def _validate_result(result: Any) -> None:
    """Validate ``result.json`` against the frontend/backend contract.

    Raises:
        ValueError: A required key, type or ordering rule is violated.
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")

    required = ("topic", "duration", "speakers", "messages", "chapters", "takeaways")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"result is missing keys: {missing}")

    if not isinstance(result["topic"], str):
        raise ValueError("topic must be a string")
    duration = float(result["duration"])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be a non-negative number")

    speakers = result["speakers"]
    if not isinstance(speakers, list):
        raise ValueError("speakers must be a list")
    speaker_ids: set[str] = set()
    for speaker in speakers:
        if not isinstance(speaker, dict):
            raise ValueError("each speaker must be an object")
        for key in ("id", "name", "on_camera", "avatar", "talk_time_pct"):
            if key not in speaker:
                raise ValueError(f"speaker is missing '{key}'")
        if (
            not isinstance(speaker["id"], str)
            or not speaker["id"]
            or speaker["id"] in speaker_ids
        ):
            raise ValueError("speaker ids must be unique non-empty strings")
        speaker_ids.add(speaker["id"])
        if not isinstance(speaker["name"], str) or not speaker["name"]:
            raise ValueError(f"speaker {speaker['id']} has no name")
        if not isinstance(speaker["on_camera"], bool):
            raise ValueError(
                f"speaker {speaker['id']} on_camera must be a boolean"
            )
        if speaker["on_camera"]:
            if (
                not isinstance(speaker["avatar"], str)
                or not speaker["avatar"]
            ):
                raise ValueError(
                    f"on-camera speaker {speaker['id']} needs an avatar"
                )
        elif speaker["avatar"] is not None:
            raise ValueError(
                f"off-camera speaker {speaker['id']} must have avatar null"
            )
        percentage = float(speaker["talk_time_pct"])
        if not 0 <= percentage <= 100:
            raise ValueError(
                f"speaker {speaker['id']} talk_time_pct is out of range"
            )

    messages = result["messages"]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    previous_start = -math.inf
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        for key in ("speaker", "start", "end", "text"):
            if key not in message:
                raise ValueError(f"message is missing '{key}'")
        if message["speaker"] not in speaker_ids:
            raise ValueError(
                f"message speaker {message['speaker']} is not in speakers"
            )
        start = float(message["start"])
        end = float(message["end"])
        if (
            not (math.isfinite(start) and math.isfinite(end))
            or start < 0
            or start > end
        ):
            raise ValueError("message times are invalid")
        if start < previous_start - 1e-6:
            raise ValueError("messages must be sorted by start")
        previous_start = start
        if not isinstance(message["text"], str):
            raise ValueError("message text must be a string")

    chapters = result["chapters"]
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("chapters must be a non-empty list")
    previous_end = 0.0
    for index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            raise ValueError("each chapter must be an object")
        for key in ("title", "start", "end", "summary"):
            if key not in chapter:
                raise ValueError(f"chapter is missing '{key}'")
        start = float(chapter["start"])
        end = float(chapter["end"])
        if (
            not (math.isfinite(start) and math.isfinite(end))
            or start < -1e-6
            or start > end
        ):
            raise ValueError("chapter times are invalid")
        if index == 0 and abs(start) > 0.01:
            raise ValueError("the first chapter must start at 0")
        if start < previous_end - 0.01:
            raise ValueError("chapters must not overlap")
        previous_end = end
        if not isinstance(chapter["title"], str) or not chapter["title"]:
            raise ValueError("chapter title must be a non-empty string")
        if not isinstance(chapter["summary"], str):
            raise ValueError("chapter summary must be a string")
    if abs(previous_end - duration) > 0.05:
        raise ValueError("chapters must cover the full video duration")

    takeaways = result["takeaways"]
    if not isinstance(takeaways, list) or any(
        not isinstance(item, str) for item in takeaways
    ):
        raise ValueError("takeaways must be a list of strings")

    if "notes" in result and (
        not isinstance(result["notes"], list)
        or any(not isinstance(item, str) for item in result["notes"])
    ):
        raise ValueError("notes must be a list of strings")


def write_result(
    job_dir: Path,
    duration: float,
    speaker_metadata: dict[str, SpeakerMeta],
    messages: list[Message],
    llm_data: dict[str, Any],
    notes: Sequence[str] | None = None,
) -> Result:
    """Assemble, validate and save the final ``result.json``.

    Talk time is the sum of each speaker's message durations as a percentage
    of total speech, rounded to a whole percent. On-camera speakers keep their
    avatar only when the crop exists on disk; otherwise they are marked
    off-camera with ``avatar: null``.

    Args:
        job_dir: Job folder.
        duration: Exact video duration in seconds.
        speaker_metadata: Output of :func:`match_speakers_to_faces`.
        messages: Merged speaker-labelled messages.
        llm_data: Output of :func:`llm_analysis`.
        notes: Human-readable limitations of this run (e.g. diarization
            unavailable, no audio track). Stored under ``result["notes"]``.

    Returns:
        The saved result object.

    Raises:
        ValueError: The assembled object violates the contract.
    """
    job_dir = Path(job_dir)
    out_path = job_dir / "result.json"
    if out_path.exists():
        cached = _load_json(out_path)
        try:
            _validate_result(cached)
        except (ValueError, TypeError, KeyError) as exc:
            print(
                f"[result] cached result is invalid ({exc}); rebuilding",
                flush=True,
            )
        else:
            return cached

    duration = max(float(duration), 0.0)
    speaker_ids = _speakers_in_order(messages)
    talk: dict[str, float] = {speaker: 0.0 for speaker in speaker_ids}
    for message in messages:
        start = min(max(float(message["start"]), 0.0), duration)
        end = min(max(float(message["end"]), start), duration)
        talk[message["speaker"]] = (
            talk.get(message["speaker"], 0.0) + (end - start)
        )
    total_talk = sum(talk.values())

    metadata = speaker_metadata if isinstance(speaker_metadata, dict) else {}
    output_speakers: list[SpeakerOut] = []
    for speaker in sorted(speaker_ids, key=_speaker_sort_key):
        info = metadata.get(speaker, {})
        on_camera = bool(info.get("on_camera", False))
        avatar = info.get("avatar") if on_camera else None
        if on_camera and (
            not isinstance(avatar, str) or not (job_dir / avatar).is_file()
        ):
            print(
                f"[result] warning: avatar for {speaker} is missing; "
                "marking the speaker off-camera",
                flush=True,
            )
            on_camera, avatar = False, None
        visual_status = info.get("visual_status")
        if not visual_status:
            visual_status = (
                "verified"
                if on_camera
                else ("unverified" if not _face_deps_available() else "off_camera")
            )
        output_speakers.append(
            {
                "id": speaker,
                "name": f"Speaker {speaker}",
                "on_camera": on_camera,
                "avatar": avatar if on_camera else None,
                "talk_time_pct": (
                    round(100.0 * talk.get(speaker, 0.0) / total_talk)
                    if total_talk > 0
                    else 0
                ),
                "visual_status": visual_status,
            }
        )

    output_messages: list[Message] = []
    for message in sorted(
        messages, key=lambda item: (item["start"], item["end"])
    ):
        start = min(max(float(message["start"]), 0.0), duration)
        end = min(max(float(message["end"]), start), duration)
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        output_messages.append(
            {
                "speaker": message["speaker"],
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
            }
        )

    raw_topic = llm_data.get("topic", "") if isinstance(llm_data, dict) else ""
    topic = (
        raw_topic.strip()
        if isinstance(raw_topic, str)
        else str(raw_topic or "").strip()
    )
    chapters = _clean_chapters(
        llm_data.get("chapters", []) if isinstance(llm_data, dict) else [],
        duration,
    )
    raw_takeaways = (
        llm_data.get("takeaways", []) if isinstance(llm_data, dict) else []
    )
    takeaways = (
        [
            item.strip()
            for item in raw_takeaways
            if isinstance(item, str) and item.strip()
        ]
        if isinstance(raw_takeaways, list)
        else []
    )

    t_file = job_dir / "transcript.json"
    t_data: dict[str, Any] = {}
    if t_file.is_file():
        try:
            t_data = _load_json(t_file)
        except Exception:
            pass

    result: Result = {
        "topic": topic,
        "duration": round(duration, 2),
        "speakers": output_speakers,
        "messages": output_messages,
        "chapters": chapters,
        "takeaways": takeaways,
        "notes": [str(note) for note in notes or []],
    }
    if t_data.get("language"):
        result["language"] = t_data["language"]
    if t_data.get("language_mode"):
        result["language_mode"] = t_data["language_mode"]
    if t_data.get("detected_language"):
        result["detected_language"] = t_data["detected_language"]
    if t_data.get("language_probability") is not None:
        result["language_probability"] = t_data["language_probability"]
    if t_data.get("transcription_model"):
        result["transcription_model"] = t_data["transcription_model"]
    _validate_result(result)
    _save_json(result, out_path)
    return result


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------


def _update_job_stage(job_dir: Path, progress: int, message: str) -> None:
    """Persist stage progress to job_state.json so status endpoints report real progress."""
    try:
        p = Path(job_dir) / "job_state.json"
        state = {
            "job_id": Path(job_dir).name,
            "status": "processing",
            "progress": max(0, min(100, int(progress))),
            "message": str(message),
            "error": "",
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def process_video(
    video_path: str | Path,
    job_id: str,
    hf_token: str | None = None,
    jobs_dir: str | Path | None = None,
    hotwords: str | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
) -> Path:
    """Run the whole pipeline for one video into ``jobs_dir / job_id``.

    Each stage is skipped when its output file already exists, so the function
    is safe to call again after a crash. Heavy models are released and CUDA
    memory is flushed between stages.

    Args:
        video_path: Source video file.
        job_id: Folder name; letters, digits, ``_`` and ``-`` only.
        hf_token: Hugging Face token for pyannote. Falls back to ``HF_TOKEN``.
        jobs_dir: Root folder for jobs. Defaults to the module's ``JOBS_DIR``;
            callers with their own configured storage (e.g. the backend)
            should pass it explicitly instead of mutating environment
            variables.
        hotwords: Optional vocabulary hints / glossary for proper nouns.
        initial_prompt: Optional initial prompt to guide transcription context.

    Returns:
        The job folder containing ``result.json``.

    Raises:
        FileNotFoundError: The source video does not exist.
        ValueError: ``job_id`` is invalid or escapes ``jobs_dir``.
    """
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError(
            "job_id may contain only letters, digits, '_' and '-'"
        )

    root = Path(jobs_dir).expanduser() if jobs_dir is not None else JOBS_DIR
    jobs_root = root.resolve()
    job_dir = (root / job_id).resolve()
    if job_dir.parent != jobs_root:
        raise ValueError(f"job_id escapes jobs_dir: {job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)

    token = (hf_token or HF_TOKEN or "").strip() or None

    t_pipeline_start = time.perf_counter()
    video_details = _probe_media_details(source)

    print(f"\n=== Processing {source.name} -> {job_dir} ===", flush=True)

    notes: list[str] = []
    stage_timings: dict[str, dict[str, Any]] = {}
    timing_info: dict[str, Any] = {
        "device": "cpu",
        "audio_duration_sec": round(video_details["duration"], 2),
        "transcription_duration_sec": 0.0,
        "transcription_rtf": 0.0,
    }

    stage_progress: dict[str, tuple[int, str]] = {
        "ingest": (12, "Extracting audio track (16 kHz mono)..."),
        "transcribe": (35, "Transcribing speech with Whisper..."),
        "diarize": (55, "Diarizing speakers with pyannote..."),
        "merge": (68, "Aligning transcript & speaker turns..."),
        "track_faces": (78, "Tracking facial presence across frames..."),
        "match_speakers_to_faces": (86, "Correlating visible faces with speakers..."),
        "llm_analysis": (92, "Synthesizing case takeaways & chapters..."),
        "index": (96, "Generating semantic search embeddings..."),
        "result": (99, "Compiling final case evidence package..."),
    }

    @contextmanager
    def _time_stage(name: str, key: str) -> Iterator[None]:
        s = time.perf_counter()
        pct, desc = stage_progress.get(key, (0, f"{name} running..."))
        _update_job_stage(job_dir, pct, desc)
        print(f"[{name}] started", flush=True)
        try:
            yield
        except BaseException:
            dur = round(time.perf_counter() - s, 3)
            print(f"[{name}] failed after {dur:.1f}s", flush=True)
            stage_timings[key] = {
                "start": round(s - t_pipeline_start, 3),
                "end": round(time.perf_counter() - t_pipeline_start, 3),
                "elapsed_sec": dur,
                "status": "failed",
            }
            raise
        else:
            dur = round(time.perf_counter() - s, 3)
            print(f"[{name}] done in {dur:.1f}s", flush=True)
            stage_timings[key] = {
                "start": round(s - t_pipeline_start, 3),
                "end": round(time.perf_counter() - t_pipeline_start, 3),
                "elapsed_sec": dur,
                "status": "completed",
            }

    with _time_stage("1 ingest", "ingest"):
        info = ingest(source, job_dir)
    has_audio = bool(info.get("has_audio", True))

    if has_audio:
        with _time_stage("2 transcribe", "transcribe"):
            transcribe_kwargs: dict[str, Any] = {"timing_info": timing_info}
            if hotwords is not None:
                transcribe_kwargs["hotwords"] = hotwords
            if initial_prompt is not None:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            if language is not None:
                transcribe_kwargs["language"] = language
            transcribe(info["audio"], job_dir, **transcribe_kwargs)

        try:
            with _time_stage("3 diarize", "diarize"):
                diarize(info["audio"], job_dir, token)
        except Exception as exc:
            # The stage raised, but the pipeline continues with an explicit
            # single-speaker fallback; report that honestly instead of a bare
            # "failed" status for a stage whose output does exist.
            # Keep native loader diagnostics server-side, including chained
            # TorchCodec exceptions. Redact credentials defensively; HF tokens
            # must never be written into logs or result documents.
            safe_error = str(exc)
            if token and token in safe_error:
                safe_error = safe_error.replace(token, "[REDACTED]")
            if "diarize" in stage_timings:
                stage_timings["diarize"]["status"] = "fallback_single_speaker"
                stage_timings["diarize"]["reason"] = (
                    f"{type(exc).__name__}: {safe_error}"[:300]
                )
            safe_traceback = traceback.format_exc()
            if token and token in safe_traceback:
                safe_traceback = safe_traceback.replace(token, "[REDACTED]")
            logging.getLogger("vke.pipeline.diarize").error(
                "Speaker diarization failed; applying explicitly labelled fallback:\n%s",
                safe_traceback,
            )
            notes.append(
                "Speaker diarization was unavailable; all speech is "
                f"attributed to a single speaker. Reason: {safe_error[:300]}"
            )
            print(
                f"[diarize] unavailable ({safe_error[:300]}); "
                "using single-speaker fallback",
                flush=True,
            )
            _save_json(
                [
                    {
                        "speaker": "SPEAKER_00",
                        "start": 0.0,
                        "end": round(max(float(info["duration"]), 0.0), 3),
                    }
                ],
                job_dir / "diarization.json",
            )

        if timing_info.get("cuda_fallback_reason"):
            notes.append(
                "Transcription ran on the CPU because the GPU runtime failed: "
                f"{timing_info['cuda_fallback_reason']}"
            )

        with _time_stage("4 merge", "merge"):
            messages = merge(job_dir)
    else:
        notes.append(
            "The source video has no audio track, so no speech transcript "
            "could be produced."
        )
        print(
            "[pipeline] no audio track; skipping transcription and "
            "diarization",
            flush=True,
        )
        _save_json({"language": "", "segments": []}, job_dir / "transcript.json")
        _save_json([], job_dir / "diarization.json")
        messages = []
        _save_json(messages, job_dir / "messages.json")
        stage_timings["transcribe"] = {
            "start": round(time.perf_counter() - t_pipeline_start, 3),
            "end": round(time.perf_counter() - t_pipeline_start, 3),
            "elapsed_sec": 0.0,
            "status": "skipped_no_audio",
        }
        stage_timings["diarize"] = {
            "start": round(time.perf_counter() - t_pipeline_start, 3),
            "end": round(time.perf_counter() - t_pipeline_start, 3),
            "elapsed_sec": 0.0,
            "status": "skipped_no_audio",
        }
        stage_timings["merge"] = {
            "start": round(time.perf_counter() - t_pipeline_start, 3),
            "end": round(time.perf_counter() - t_pipeline_start, 3),
            "elapsed_sec": 0.0,
            "status": "skipped_no_audio",
        }

    if not ENABLE_FACE_TRACKING:
        notes.append(
            "Face tracking was disabled by configuration; speaker visual presence could not be verified."
        )
    elif not _face_deps_available():
        notes.append(
            "Face tracking was unavailable because face detection dependencies could not initialize; speaker visual presence is unverified."
        )

    with _time_stage("5 track faces", "track_faces"):
        tracks = track_faces(info["video"], job_dir)

    with _time_stage("6 match speakers to faces", "match_speakers_to_faces"):
        speaker_metadata = match_speakers_to_faces(messages, tracks, job_dir)

    llm_path = job_dir / "llm.json"
    if llm_path.exists():
        with _time_stage("7 llm analysis (cached)", "llm_analysis"):
            llm_data = _load_json(llm_path)
    else:
        with _time_stage("7 llm analysis", "llm_analysis"):
            llm_data = llm_analysis(messages)
            _save_json(llm_data, llm_path)

    with _time_stage("8 index", "index"):
        chunks = build_index(messages, job_dir)
    if chunks and all(chunk.get("embedding") is None for chunk in chunks):
        notes.append(
            "Semantic search embeddings were unavailable; search falls back "
            "to keyword matching."
        )

    with _time_stage("9 result", "result"):
        result = write_result(
            job_dir,
            info["duration"],
            speaker_metadata,
            messages,
            llm_data,
            notes=notes,
        )

    total_elapsed = round(time.perf_counter() - t_pipeline_start, 3)
    if has_audio:
        audio_dur = timing_info.get("audio_duration_sec") or round(
            info.get("duration", 0.0), 2
        )
    else:
        # No audio track: do not report the video duration as audio duration.
        audio_dur = 0.0
    trans_dur = timing_info.get("transcription_duration_sec", 0.0)
    rtf = round(trans_dur / max(audio_dur, 0.001), 3) if audio_dur > 0 else 0.0

    timings_report: dict[str, Any] = {
        "job_id": job_id,
        "video": {
            "filename": source.name,
            "file_size_bytes": video_details["file_size_bytes"],
            "file_size_mb": video_details["file_size_mb"],
            "duration_sec": round(video_details["duration"], 2),
            "resolution": video_details["resolution"],
            "width": video_details["width"],
            "height": video_details["height"],
            "fps": video_details["fps"],
            "video_codec": video_details["video_codec"],
            "audio_codec": video_details["audio_codec"],
        },
        "device": timing_info.get("device", "cpu"),
        "compute_type": timing_info.get("compute_type", "int8"),
        "whisper_model": timing_info.get("model", WHISPER_MODEL),
        "model_load_sec": timing_info.get("model_load_sec", 0.0),
        "model_cached": timing_info.get("model_cached", False),
        "batch_size": timing_info.get("batch_size", 0),
        "chunk_length": timing_info.get("chunk_length", 0),
        "has_audio": has_audio,
        "audio_duration_sec": round(audio_dur, 2),
        "transcription_duration_sec": round(trans_dur, 3),
        "transcription_rtf": rtf,
        "stages": stage_timings,
        "total_elapsed_sec": total_elapsed,
    }
    if timing_info.get("cuda_fallback_reason"):
        timings_report["cuda_fallback_reason"] = timing_info["cuda_fallback_reason"]
    _save_json(timings_report, job_dir / "timings.json")

    print("\n=== Timing Report ===", flush=True)
    print(
        f"[timings] Video: {source.name} ({video_details['duration']:.2f}s, "
        f"{video_details['resolution']} @ {video_details['fps']}fps, "
        f"{video_details['file_size_mb']}MB)",
        flush=True,
    )
    print(
        f"[timings] Device: {timings_report['device']} | "
        f"Audio: {timings_report['audio_duration_sec']}s | "
        f"Transcription: {timings_report['transcription_duration_sec']}s "
        f"(RTF: {rtf:.3f}x)",
        flush=True,
    )
    print(
        f"[timings] Whisper: {timings_report['whisper_model']} "
        f"({timings_report['compute_type']}, load "
        f"{timings_report['model_load_sec']}s, cached={timings_report['model_cached']})",
        flush=True,
    )
    if timings_report.get("cuda_fallback_reason"):
        print(
            f"[timings] CUDA fallback: {timings_report['cuda_fallback_reason']}",
            flush=True,
        )
    breakdown_parts = [f"{k}={v['elapsed_sec']:.2f}s" for k, v in stage_timings.items()]
    print(f"[timings] Breakdown: {', '.join(breakdown_parts)}", flush=True)
    print(f"[timings] Total elapsed: {total_elapsed:.2f}s", flush=True)
    print("=====================\n", flush=True)

    print("=== Done ===", flush=True)
    print(f"job folder : {job_dir}", flush=True)
    print(f"duration   : {result['duration']}s", flush=True)
    print(f"speakers   : {len(result['speakers'])}", flush=True)
    print(f"messages   : {len(result['messages'])}", flush=True)
    print(f"chapters   : {len(result['chapters'])}", flush=True)
    print(f"result     : {job_dir / 'result.json'}", flush=True)
    return job_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Turn a short video into a speaker-labelled transcript, "
            "chapters, takeaways and avatars."
        )
    )
    parser.add_argument("video", help="path to the source video")
    parser.add_argument("job_id", help="job folder name")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote (or set HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    try:
        process_video(args.video, args.job_id, args.hf_token)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
