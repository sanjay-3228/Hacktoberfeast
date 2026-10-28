"""Regression and accuracy tests for speech transcription.

Covers:
- Options propagation (beam_size, repetition_penalty, condition_on_previous_text, hotwords, initial_prompt).
- Timing instrumentation records beam_size and repetition_penalty.
- Decoder repetition loop prevention: repetitive/hesitant speech does not trigger 90+ duplicate loops.
- Authentic repetition preservation: natural repeats ("yeah yeah", "this this") are preserved.
- Degenerate zero-duration word collapse pruning.
- Word Error Rate (WER) computation utility and normalized evaluation.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vke_pipeline as vp


def test_transcribe_forwards_accuracy_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakePipeline:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([]), types.SimpleNamespace(duration=3.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cuda", "float16", 8)
    )
    monkeypatch.setattr(vp, "WHISPER_BEAM_SIZE", 5)
    monkeypatch.setattr(vp, "WHISPER_REPETITION_PENALTY", 1.1)
    monkeypatch.setattr(vp, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    timing: dict = {}
    vp.transcribe(
        audio,
        tmp_path,
        timing_info=timing,
        hotwords="David Goggins, Can't Hurt Me",
        initial_prompt="A podcast discussion",
    )

    assert captured["beam_size"] == 5
    assert captured["repetition_penalty"] == pytest.approx(1.1)
    assert captured["condition_on_previous_text"] is False
    assert captured["hotwords"] == "David Goggins, Can't Hurt Me"
    assert captured["initial_prompt"] == "A podcast discussion"
    assert timing["beam_size"] == 5
    assert timing["repetition_penalty"] == pytest.approx(1.1)


def test_transcribe_default_hotwords_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakePipeline:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([]), types.SimpleNamespace(duration=2.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cpu", "int8", 0)
    )
    monkeypatch.setattr(vp, "WHISPER_HOTWORDS", "Test Term, Another Term")

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    vp.transcribe(audio, tmp_path)

    assert captured.get("hotwords") == "Test Term, Another Term"


def test_zero_duration_degenerate_duplicates_are_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASR collapse artifacts where start==end for identical repeated words are dropped."""
    # Simulate a decoder collapse emitting 10 identical 'this' tokens with zero duration
    collapsed_words = [
        types.SimpleNamespace(word="heard", start=89.0, end=89.2),
        types.SimpleNamespace(word="this", start=89.2, end=89.5),
        types.SimpleNamespace(word="this", start=89.5, end=89.8),  # legitimate second spoken repeat
    ]
    # Appended zero-duration collapse loops
    for _ in range(10):
        collapsed_words.append(types.SimpleNamespace(word="this", start=89.8, end=89.8))

    fake_segment = types.SimpleNamespace(
        start=89.0,
        end=90.0,
        text="heard this this this this this this this this this this this",
        words=collapsed_words,
    )

    class FakePipeline:
        def transcribe(self, path, **kwargs):
            return iter([fake_segment]), types.SimpleNamespace(duration=90.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cpu", "int8", 0)
    )

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    doc = vp.transcribe(audio, tmp_path)

    words = doc["segments"][0]["words"]
    # Authentic words: 'heard', 'this' (89.2-89.5), 'this' (89.5-89.8)
    assert len(words) == 3
    assert [w["word"] for w in words] == ["heard", "this", "this"]
    assert words[1]["start"] == 89.2 and words[1]["end"] == 89.5
    assert words[2]["start"] == 89.5 and words[2]["end"] == 89.8


def test_natural_speech_repetition_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Natural repeated words with distinct timestamps must never be dropped."""
    repeated_words = [
        types.SimpleNamespace(word="yeah", start=1.0, end=1.3),
        types.SimpleNamespace(word="yeah", start=1.4, end=1.7),
        types.SimpleNamespace(word="yeah", start=1.8, end=2.1),
        types.SimpleNamespace(word="absolutely", start=2.2, end=2.8),
    ]
    fake_segment = types.SimpleNamespace(
        start=1.0, end=2.8, text="yeah yeah yeah absolutely", words=repeated_words
    )

    class FakePipeline:
        def transcribe(self, path, **kwargs):
            return iter([fake_segment]), types.SimpleNamespace(duration=3.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cpu", "int8", 0)
    )

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    doc = vp.transcribe(audio, tmp_path)

    words = doc["segments"][0]["words"]
    assert len(words) == 4
    assert [w["word"] for w in words] == ["yeah", "yeah", "yeah", "absolutely"]


def test_wer_computation_utility() -> None:
    """Test Word Error Rate normalized calculation logic."""
    import re

    def compute_wer(ref: str, hyp: str) -> tuple[float, int, int, int, int]:
        def norm(t: str) -> list[str]:
            return re.sub(r"[^\w\s]", "", t.lower()).split()

        r = norm(ref)
        h = norm(hyp)
        n = len(r)
        if n == 0:
            return (0.0 if len(h) == 0 else 1.0, 0, 0, 0, n)
        d = [[0] * (len(h) + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return (d[n][len(h)] / n, 0, 0, 0, n)

    # Exact match
    wer, _, _, _, n = compute_wer("Hello world", "hello world")
    assert wer == 0.0
    assert n == 2

    # Single substitution
    wer, _, _, _, _ = compute_wer("He had so many quotes", "He had so many courts")
    assert wer == pytest.approx(0.2)  # 1 out of 5

    # Autoregressive repetition explosion (like the baseline defect)
    wer_loop, _, _, _, _ = compute_wer("and I heard this this here", "and I heard " + "this " * 50 + "here")
    assert wer_loop > 5.0  # massive WER explosion due to 50 insertions


def test_real_audio_hesitation_no_repetition_loop(tmp_path: Path) -> None:
    """Run real faster-whisper transcription on the test recording if available.

    Verifies that the hesitation around 89-98s produces authentic speech without
    entering the 90+ duplicate word attractor loop.
    """
    audio_path = Path("backend/jobs/ac3d037fc573/audio.wav")
    if not audio_path.is_file():
        pytest.skip("Test recording backend/jobs/ac3d037fc573/audio.wav not found")

    vp.clear_whisper_cache()
    timing_info: dict = {}
    doc = vp.transcribe(audio_path, tmp_path, timing_info=timing_info)
    segments = doc.get("segments", [])

    # Filter segments in the hesitation window 80s - 105s
    window_segments = [
        s
        for s in segments
        if (80.0 <= s["start"] <= 105.0) or (80.0 <= s["end"] <= 105.0)
    ]
    window_text = " ".join(s["text"] for s in window_segments)
    this_count = sum(
        1 for w in window_text.lower().split() if w.strip(".,!?;:") == "this"
    )

    # In the defective baseline, 'this' was repeated 92 times.
    # In authentic speech, 'this' is spoken twice: 'and I heard this, this here...'
    assert this_count < 10, f"Decoder entered repetition loop: count={this_count}"
    assert this_count >= 1, "Spoken word 'this' should be captured"
    assert "quote" in window_text.lower()
    assert timing_info["repetition_penalty"] == pytest.approx(1.1)
    assert timing_info["beam_size"] >= 1


def test_transcribe_language_option_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that language parameter and WHISPER_LANGUAGE are passed to transcribe."""
    captured: dict = {}

    class FakePipeline:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([]), types.SimpleNamespace(duration=2.0, language="en")

    monkeypatch.setattr(
        vp, "_get_transcriber", lambda *a, **k: (FakePipeline(), "cpu", "int8", 0)
    )

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    # 1. Explicit parameter overrides env var
    monkeypatch.setattr(vp, "WHISPER_LANGUAGE", "fr")
    doc = vp.transcribe(audio, tmp_path / "job1", language="en")
    assert captured.get("language") == "en"
    assert doc["language"] == "en"

    # 2. Defaults to WHISPER_LANGUAGE env var when language parameter is None
    captured.clear()
    monkeypatch.setattr(vp, "WHISPER_LANGUAGE", "es")
    doc = vp.transcribe(audio, tmp_path / "job2", language=None)
    assert captured.get("language") == "es"
    assert doc["language"] == "es"

    # 3. Omits language option when WHISPER_LANGUAGE is None (auto-detect)
    captured.clear()
    monkeypatch.setattr(vp, "WHISPER_LANGUAGE", None)
    vp.transcribe(audio, tmp_path / "job3", language=None)
    assert "language" not in captured


def test_misidentified_english_heuristic() -> None:
    """Unit test: _is_misidentified_english distinguishes accented English from genuine Hindi."""
    # Defective English audio misclassified as 'hi' with mixed Urdu/Devanagari
    corrupted_segments = [
        {"text": "Learning for one hour every day. I think people underestimate its value"},
        {"text": "ایک ہے کہ لڑنگ آپ کو کھیں سے بھی مل سکتے ہیں but its secondary nuance is that you have to absorb it"},
        {"text": "So there is a theory which is called the cone of learning"},
    ]
    assert vp._is_misidentified_english("hi", corrupted_segments) is True

    # Genuine Hindi conversation (Devanagari script, negligible Latin text)
    authentic_hindi_segments = [
        {"text": "नमस्ते, आप कैसे हैं? आज हम इस विषय पर बात करेंगे।"},
        {"text": "यह बहुत महत्वपूर्ण है और हमें इसे समझना चाहिए।"},
    ]
    assert vp._is_misidentified_english("hi", authentic_hindi_segments) is False

    # English or Spanish detections should never trigger misidentification
    assert vp._is_misidentified_english("en", corrupted_segments) is False
    assert vp._is_misidentified_english("es", corrupted_segments) is False


def test_real_audio_english_no_urdu_or_devanagari_output(tmp_path: Path) -> None:
    """[Integration Test - Genuine Audio]: Job 2f17437f2404 audio.

    Verifies that the 10:30 English podcast recording:
    1. Transcribes cleanly with language='en'.
    2. Contains ZERO Urdu/Arabic script characters (\\u0600-\\u06FF).
    3. Contains ZERO Devanagari script characters (\\u0900-\\u097F).
    4. Does not trigger the pathological repetition loop at 09:33-09:47.
    """
    audio_path = Path("backend/jobs/2f17437f2404/audio.wav")
    if not audio_path.is_file():
        pytest.skip("Test recording backend/jobs/2f17437f2404/audio.wav not found")

    vp.clear_whisper_cache()
    timing_info: dict = {}
    doc = vp.transcribe(audio_path, tmp_path, timing_info=timing_info, language="en")

    assert doc["language"] == "en"
    segments = doc.get("segments", [])
    assert len(segments) > 0

    full_transcript = " ".join(s["text"] for s in segments)

    # Check that zero Urdu/Arabic characters exist in the transcript
    urdu_chars = [c for c in full_transcript if "\u0600" <= c <= "\u06ff"]
    assert len(urdu_chars) == 0, f"Found unexpected Urdu script characters: {urdu_chars[:10]}"

    # Check that zero Devanagari characters exist in the transcript
    devanagari_chars = [c for c in full_transcript if "\u0900" <= c <= "\u097f"]
    assert len(devanagari_chars) == 0, f"Found unexpected Devanagari script characters: {devanagari_chars[:10]}"

    # Verify 09:20-10:00 window (560s - 600s) has no repetition loop
    window_segments = [s for s in segments if 560.0 <= s["start"] <= 600.0 or 560.0 <= s["end"] <= 600.0]
    window_text = " ".join(s["text"] for s in window_segments)

    # In defective run, 'तो में' was repeated 58 times
    assert "तो में" not in window_text
    # Authentic discussion of podcast, succumbing to energy, and exposure therapy
    assert "podcast" in window_text.lower()
    assert "exposure therapy" in window_text.lower()
