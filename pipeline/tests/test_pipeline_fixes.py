"""Regression tests for the pipeline fixes (no ML models required).

Covers:
- ``process_video(jobs_dir=...)`` uses the explicitly passed job root.
- CUDA detection goes through CTranslate2, never through a torch import.
- Media probing distinguishes video-only files from video+audio files.
- Video-only files: transcription/diarization are skipped and the result
  records that no audio track existed (no fabricated transcript).
- Diarization unavailable: a single-speaker fallback is used and the result
  notes record that genuine speaker separation did not happen.

Run from the pipeline/ folder:  python -m pytest tests -q
FFmpeg-dependent tests are skipped automatically when ffmpeg/ffprobe are
missing.
"""
from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vke_pipeline as vp  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not installed"
)


def _make_video(path: Path, *, with_audio: bool, duration: int = 2) -> Path:
    """Generate a tiny H.264 MP4, optionally with an AAC tone track."""
    if with_audio:
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ]
    else:
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=red:duration={duration}",
            "-c:v", "libx264", str(path),
        ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _stub_stages(monkeypatch: pytest.MonkeyPatch, *, on_no_audio: bool) -> None:
    """Replace the model-heavy stages with deterministic stubs."""
    monkeypatch.setattr(vp, "track_faces", lambda *a, **k: [])
    monkeypatch.setattr(vp, "match_speakers_to_faces", lambda *a, **k: {})
    monkeypatch.setattr(
        vp, "build_index", lambda messages, job_dir: vp._save_json([], job_dir / "chunks.json") or []
    )
    if on_no_audio:
        def _forbidden(*_a, **_k):  # pragma: no cover - failure path
            raise AssertionError("stage must not run when there is no audio")
        monkeypatch.setattr(vp, "transcribe", _forbidden)
        monkeypatch.setattr(vp, "merge", _forbidden)


def _canned_transcribe(monkeypatch: pytest.MonkeyPatch, words: list[dict]) -> None:
    def fake_transcribe(
        audio_path: Path,
        job_dir: Path,
        timing_info: dict | None = None,
    ) -> dict:
        data = {"language": "en", "segments": [
            {"start": words[0]["start"], "end": words[-1]["end"],
             "text": " ".join(w["word"].strip() for w in words),
             "words": words},
        ]}
        vp._save_json(data, Path(job_dir) / "transcript.json")
        if timing_info is not None:
            timing_info.update({
                "device": "cpu",
                "audio_duration_sec": round(words[-1]["end"], 2),
                "transcription_duration_sec": 0.1,
                "transcription_rtf": 0.1,
            })
        return data
    monkeypatch.setattr(vp, "transcribe", fake_transcribe)


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------


def test_cuda_detection_never_imports_torch() -> None:
    import re

    for fn in (vp.transcribe, vp._cuda_device_count):
        assert re.search(r"\b(import torch|from torch\b)", inspect.getsource(fn)) is None


def test_cuda_detection_uses_ctranslate2(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.get_cuda_device_count = lambda: 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    # Any `import torch` inside these helpers would now raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert vp._cuda_device_count() == 1

    fake_ct2.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    assert vp._cuda_device_count() == 0

    # Simulate CTranslate2 being absent: `import ctranslate2` now raises.
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    assert vp._cuda_device_count() == 0


# ---------------------------------------------------------------------------
# Media probing
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_probe_media_reports_video_only_versus_av(tmp_path: Path) -> None:
    silent = _make_video(tmp_path / "silent.mp4", with_audio=False)
    loud = _make_video(tmp_path / "loud.mp4", with_audio=True)
    assert vp._probe_media(silent) == ("h264", None)
    assert vp._probe_media(loud) == ("h264", "aac")


# ---------------------------------------------------------------------------
# Video-only ingest honesty
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_video_only_ingest_generates_silent_track_and_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_video(tmp_path / "src.mp4", with_audio=False)
    job = tmp_path / "job"
    info = vp.ingest(source, job)
    assert info["has_audio"] is False
    assert Path(info["audio"]).is_file()
    assert Path(info["video"]).is_file()
    assert info["duration"] == pytest.approx(2.0, abs=0.3)
    # Sidecar persists the flag for later cached runs.
    meta = json.loads((job / "ingest.json").read_text(encoding="utf-8"))
    assert meta["has_audio"] is False
    # A cached run must not silently flip the flag back to True.
    again = vp.ingest(source, job)
    assert again["has_audio"] is False


@needs_ffmpeg
def test_process_video_video_only_skips_transcription_and_records_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, on_no_audio=True)
    source = _make_video(tmp_path / "src.mp4", with_audio=False)
    jobs_root = tmp_path / "jobs"
    job_dir = vp.process_video(source, "silent-job", jobs_dir=jobs_root)

    # Explicit jobs_dir is honored (no reliance on the module default).
    assert job_dir == (jobs_root / "silent-job").resolve()

    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert result["speakers"] == []
    assert result["messages"] == []
    assert any("no audio track" in note for note in result["notes"])
    assert result["duration"] == pytest.approx(2.0, abs=0.3)
    assert result["chapters"] and result["chapters"][0]["start"] == 0.0
    assert result["chapters"][-1]["end"] == pytest.approx(result["duration"], abs=0.05)

    transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["segments"] == []
    diarization = json.loads((job_dir / "diarization.json").read_text(encoding="utf-8"))
    assert diarization == []


# ---------------------------------------------------------------------------
# Diarization fallback
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_process_video_diarize_fallback_single_speaker_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, on_no_audio=False)
    words = [
        {"word": " hello", "start": 0.2, "end": 0.5},
        {"word": " world", "start": 0.6, "end": 0.9},
        {"word": " again", "start": 1.0, "end": 1.3},
    ]
    _canned_transcribe(monkeypatch, words)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("test: TorchCodec unavailable")

    monkeypatch.setattr(vp, "diarize", unavailable)
    source = _make_video(tmp_path / "src.mp4", with_audio=True)
    job_dir = vp.process_video(source, "fallback-job", jobs_dir=tmp_path / "jobs")

    # The forced stage failure must produce one explicitly labelled fallback,
    # never a false claim that pyannote succeeded.
    turns = json.loads((job_dir / "diarization.json").read_text(encoding="utf-8"))
    assert len(turns) == 1
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == pytest.approx(2.0, abs=0.3)

    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert any("diarization was unavailable" in note.lower() for note in result["notes"])
    assert [s["id"] for s in result["speakers"]] == ["S1"]
    assert result["speakers"][0]["on_camera"] is False
    assert result["speakers"][0]["talk_time_pct"] == 100
    assert len(result["messages"]) == 1
    assert "hello world again" in result["messages"][0]["text"]
    assert result["messages"][0]["start"] == pytest.approx(0.2)

    # Transcript and result live at the documented paths inside the job dir.
    transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["segments"][0]["words"] == words

    # The fallback is reported as a fallback, not as a genuine diarization run
    # and not as a bare failure.
    timings = json.loads((job_dir / "timings.json").read_text(encoding="utf-8"))
    assert timings["stages"]["diarize"]["status"] == "fallback_single_speaker"
    assert timings["stages"]["diarize"]["reason"]


# ---------------------------------------------------------------------------
# Result schema: notes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

STAGE_KEYS = (
    "ingest",
    "transcribe",
    "diarize",
    "merge",
    "track_faces",
    "match_speakers_to_faces",
    "llm_analysis",
    "index",
    "result",
)


@needs_ffmpeg
def test_timings_report_records_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, on_no_audio=False)
    _canned_transcribe(
        monkeypatch,
        [{"word": " hi", "start": 0.1, "end": 0.4}],
    )
    source = _make_video(tmp_path / "src.mp4", with_audio=True)
    job_dir = vp.process_video(source, "timed-job", jobs_dir=tmp_path / "jobs")
    timings = json.loads((job_dir / "timings.json").read_text(encoding="utf-8"))
    assert timings["job_id"] == "timed-job"
    assert timings["has_audio"] is True
    # The stubbed stage creates a 320x240 video (ffmpeg default when not
    # specifying dimensions, matching the pipeline's real-world behavior).
    assert timings["video"]["resolution"] == "320x240"
    assert timings["video"]["duration_sec"] == pytest.approx(2.0, abs=0.3)
    assert timings["video"]["file_size_bytes"] > 0
    assert timings["total_elapsed_sec"] >= 0
    assert timings["audio_duration_sec"] > 0
    assert timings["transcription_duration_sec"] == pytest.approx(0.1)
    assert timings["transcription_rtf"] > 0

    for key in STAGE_KEYS:
        assert key in timings["stages"], key
        stage = timings["stages"][key]
        assert {"start", "end", "elapsed_sec", "status"} <= set(stage)
        assert stage["end"] >= stage["start"]
        assert stage["elapsed_sec"] >= 0




@needs_ffmpeg
def test_timings_report_video_only_has_no_audio_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, on_no_audio=True)
    source = _make_video(tmp_path / "src.mp4", with_audio=False)
    job_dir = vp.process_video(source, "silent-timing", jobs_dir=tmp_path / "jobs")

    timings = json.loads((job_dir / "timings.json").read_text(encoding="utf-8"))
    assert timings["has_audio"] is False
    # Video length must not be reported as audio length.
    assert timings["audio_duration_sec"] == 0.0
    assert timings["transcription_duration_sec"] == 0.0
    assert timings["transcription_rtf"] == 0.0
    for key in ("transcribe", "diarize", "merge"):
        assert timings["stages"][key]["status"] == "skipped_no_audio"


# ---------------------------------------------------------------------------
# Batched decoding options
# ---------------------------------------------------------------------------


def test_transcribe_uses_batch_and_chunk_length_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakePipeline:
        def transcribe(self, path, **kwargs):  # noqa: ANN001, ANN003
            captured.update(kwargs)
            return iter([]), types.SimpleNamespace(duration=3.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cuda", "float16", 8)
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    timing: dict = {}
    vp.transcribe(audio, tmp_path, timing_info=timing)

    assert captured["batch_size"] == 8
    assert captured["chunk_length"] == vp.WHISPER_CHUNK_LENGTH
    assert 1 <= vp.WHISPER_CHUNK_LENGTH <= 30
    assert timing["batch_size"] == 8
    assert timing["chunk_length"] == vp.WHISPER_CHUNK_LENGTH


def test_transcribe_sequential_path_omits_batch_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):  # noqa: ANN001, ANN003
            captured.update(kwargs)
            return iter([]), types.SimpleNamespace(duration=2.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakeModel(), "cpu", "int8", 0)
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    timing: dict = {}
    vp.transcribe(audio, tmp_path, timing_info=timing)

    assert "batch_size" not in captured
    assert "chunk_length" not in captured
    assert timing["batch_size"] == 0
    assert timing["chunk_length"] == 0


def test_validate_result_accepts_and_typechecks_notes() -> None:
    base = {
        "topic": "t", "duration": 10.0,
        "speakers": [], "messages": [],
        "chapters": [{"title": "Full video", "start": 0.0, "end": 10.0, "summary": ""}],
        "takeaways": [],
    }
    vp._validate_result({**base, "notes": ["a note"]})
    with pytest.raises(ValueError, match="notes"):
        vp._validate_result({**base, "notes": "not-a-list"})
    with pytest.raises(ValueError, match="notes"):
        vp._validate_result({**base, "notes": [1]})
