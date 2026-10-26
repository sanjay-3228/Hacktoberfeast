"""CPU-only tests for the pure logic (no models needed).
Run from the pipeline/ folder:  python -m pytest tests -q   (or: python tests/test_logic.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from stages.index import make_chunks
from stages.llm import _clean_chapters, extract_json
from stages.match import match_speakers_to_faces
from stages.merge import merge
from stages.result import build_result, validate_result


def _words(text, start, step=0.4):
    return [{"word": " " + w, "start": start + i * step, "end": start + (i + 1) * step - 0.05}
            for i, w in enumerate(text.split())]


def test_merge_assigns_speakers_and_renames():
    w1 = _words("hello there everyone", 0.0)
    w2 = _words("thanks for having me", 5.0)
    transcript = {"segments": [{"words": w1}, {"words": w2}]}
    turns = [{"speaker": "SPEAKER_07", "start": 0, "end": 4},
             {"speaker": "SPEAKER_02", "start": 4.5, "end": 9}]
    msgs = merge(transcript, turns)
    assert [m["speaker"] for m in msgs] == ["S1", "S2"]
    assert msgs[0]["text"] == "hello there everyone"


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope it helps') == {"a": 2}


def test_chapters_are_contiguous():
    ch = _clean_chapters([{"title": "B", "start": 50}, {"title": "A", "start": 5}], 100)
    assert ch[0]["start"] == 0 and ch[0]["end"] == 50 and ch[1]["end"] == 100


def test_chunks():
    msgs = [{"speaker": "S1", "start": i * 10.0, "end": i * 10.0 + 9, "text": f"m{i}"}
            for i in range(10)]
    chunks = make_chunks(msgs)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == 99
    assert all(c["embedding"] is None for c in chunks)


def test_match_on_and_off_camera():
    fps = config.FACE_FPS
    duration = 60.0
    # S1 talks 0-30, S2 talks 30-60 (S2 is off camera: no matching face)
    msgs = [{"speaker": "S1", "start": 0, "end": 30, "text": "a"},
            {"speaker": "S2", "start": 30, "end": 60, "text": "b"}]
    rng = np.random.default_rng(0)
    samples = []
    for i in range(int(duration * fps)):
        t = i / fps
        mouth = 0.02 + (0.05 * abs(np.sin(t * 6)) if t < 30 else 0.0) + rng.normal(0, 0.0005)
        samples.append({"t": round(t, 2), "mouth": float(mouth)})
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        (job / "faces").mkdir()
        (job / "faces" / "track_0.jpg").write_bytes(b"x")
        tracks = [{"track": 0, "samples": samples, "crop": "faces/track_0.jpg"}]
        fm = match_speakers_to_faces(msgs, tracks, duration, job)
        assert fm["S1"]["on_camera"] and fm["S1"]["avatar"] == "faces/S1.jpg"
        assert not fm["S2"]["on_camera"] and fm["S2"]["avatar"] is None


def test_result_validates():
    msgs = [{"speaker": "S1", "start": 0, "end": 5, "text": "hi"},
            {"speaker": "S2", "start": 5, "end": 10, "text": "yo"}]
    fm = {"S1": {"on_camera": True, "avatar": "faces/S1.jpg"}}
    r = build_result("t", 10, msgs, fm, {"speaker_names": {"S1": "Priya"}, "chapters": []})
    validate_result(r)
    assert r["speakers"][0]["name"] == "Priya" and r["speakers"][1]["name"] == "Speaker 2"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
