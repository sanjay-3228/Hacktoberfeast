"""Tests for visual face tracking, speaker matching, and semantic transcript search.

Ensures compliance with CONTRACT.md, API.md, and:
- Rule 5: Visible persons are not attributed to speakers without temporal evidence (unverified/uncertain label).
- Rule 6: Speakers are not labeled off-camera merely because dependencies are missing.
- Semantic search: Embeddings are generated, cached, used for cosine similarity, and fall back to keywords.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure both pipeline and backend are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import vke_pipeline as vp
from backend.main import app, _search_chunks, _load_chunks, _semantic_scores, _keyword_scores


@pytest.fixture
def temp_job(tmp_path: Path) -> Path:
    job = tmp_path / "test_job"
    job.mkdir(parents=True)
    return job


def test_face_deps_available():
    """Verify face dependencies availability check."""
    assert vp._face_deps_available() is True


def test_face_detector_initialization_and_detection():
    """Verify _FaceDetector initializes and processes frames."""
    detector = vp._FaceDetector()
    try:
        # Create a blank RGB frame (height 200, width 200)
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = detector.detect(blank)
        assert isinstance(dets, list)
        assert len(dets) == 0
    finally:
        detector.close()


def test_match_speakers_to_faces_rule5_uncertain_association(temp_job: Path):
    """Rule 5: Weak/uncertain correlation (< threshold) is labeled unverified, not on-camera."""
    messages = [
        {"speaker": "S1", "start": 0.0, "end": 10.0, "text": "Hello world"},
    ]
    # Create a track with low/uncorrelated mouth movement
    samples = [{"t": float(t) * 0.2, "mouth": 0.01} for t in range(50)]
    (temp_job / "faces").mkdir()
    (temp_job / "faces" / "T0.jpg").write_bytes(b"dummy_crop")
    tracks = [{"track": 0, "samples": samples, "crop": "faces/T0.jpg"}]

    meta = vp.match_speakers_to_faces(messages, tracks, temp_job)
    assert meta["S1"]["on_camera"] is False
    assert meta["S1"]["avatar"] is None
    # Must be either unverified (if low correlation) or off_camera, never on_camera
    assert meta["S1"]["visual_status"] in ("unverified", "off_camera")


def test_match_speakers_to_faces_rule5_strong_correlation_verified(temp_job: Path):
    """Strong temporal correlation associates speaker and creates verified avatar."""
    # S1 speaks from 0 to 10s, S2 speaks from 10 to 20s
    messages = [
        {"speaker": "S1", "start": 0.0, "end": 10.0, "text": "Speaking here"},
        {"speaker": "S2", "start": 10.0, "end": 20.0, "text": "Another speaker"},
    ]
    fps = vp.FACE_FPS
    rng = np.random.default_rng(0)
    samples = []
    for step in range(int(20.0 * fps)):
        t = step / fps
        # Mouth moves when S1 speaks (0-10s), still when S2 speaks (10-20s)
        mouth = 0.02 + (0.05 * abs(np.sin(t * 6)) if t < 10.0 else 0.0) + float(rng.normal(0, 0.0005))
        samples.append({"t": round(t, 2), "mouth": float(mouth)})

    (temp_job / "faces").mkdir(parents=True, exist_ok=True)
    (temp_job / "faces" / "T0.jpg").write_bytes(b"image_bytes_s1")
    tracks = [{"track": 0, "samples": samples, "crop": "faces/T0.jpg"}]

    meta = vp.match_speakers_to_faces(messages, tracks, temp_job)
    assert meta["S1"]["on_camera"] is True
    assert meta["S1"]["avatar"] == "faces/S1.jpg"
    assert meta["S1"]["visual_status"] == "verified"
    assert (temp_job / "faces" / "S1.jpg").is_file()

    # S2 had no active face during their speech turn
    assert meta["S2"]["on_camera"] is False
    assert meta["S2"]["avatar"] is None
    assert meta["S2"]["visual_status"] == "off_camera"


def test_match_speakers_to_faces_rule6_missing_deps_labeled_unverified(temp_job: Path, monkeypatch):
    """Rule 6: When face dependencies are missing, speakers are labeled unverified, not off-camera."""
    monkeypatch.setattr(vp, "_face_deps_available", lambda: False)
    messages = [
        {"speaker": "S1", "start": 0.0, "end": 10.0, "text": "Hello world"},
    ]
    meta = vp.match_speakers_to_faces(messages, [], temp_job)
    assert meta["S1"]["on_camera"] is False
    assert meta["S1"]["avatar"] is None
    assert meta["S1"]["visual_status"] == "unverified"


def test_write_result_preserves_contract_and_includes_visual_status(temp_job: Path):
    """write_result strictly adheres to CONTRACT.md while preserving visual_status."""
    messages = [
        {"speaker": "S1", "start": 0.0, "end": 5.0, "text": "Speech one"},
        {"speaker": "S2", "start": 5.0, "end": 10.0, "text": "Speech two"},
    ]
    (temp_job / "faces").mkdir(parents=True, exist_ok=True)
    (temp_job / "faces" / "S1.jpg").write_bytes(b"crop")
    speaker_meta = {
        "S1": {"on_camera": True, "avatar": "faces/S1.jpg", "score": 0.8, "track": 0, "visual_status": "verified"},
        "S2": {"on_camera": False, "avatar": None, "score": 0.0, "track": None, "visual_status": "unverified"},
    }
    llm_data = {"topic": "Test video", "chapters": [], "takeaways": []}

    res = vp.write_result(temp_job, 10.0, speaker_meta, messages, llm_data)
    assert res["speakers"][0]["id"] == "S1"
    assert res["speakers"][0]["on_camera"] is True
    assert res["speakers"][0]["avatar"] == "faces/S1.jpg"
    assert res["speakers"][0]["visual_status"] == "verified"

    assert res["speakers"][1]["id"] == "S2"
    assert res["speakers"][1]["on_camera"] is False
    assert res["speakers"][1]["avatar"] is None
    assert res["speakers"][1]["visual_status"] == "unverified"


def test_build_index_embeds_and_caches_chunks(temp_job: Path):
    """build_index embeds chunks using sentence-transformers and caches result."""
    messages = [
        {"speaker": "S1", "start": 0.0, "end": 5.0, "text": "Artificial intelligence and deep learning models."},
    ]
    chunks = vp.build_index(messages, temp_job)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Artificial intelligence and deep learning models."
    assert isinstance(chunks[0]["embedding"], list)
    assert len(chunks[0]["embedding"]) == 384
    assert (temp_job / "chunks.json").is_file()

    # Second call should load from cache without re-encoding
    cached_chunks = vp.build_index(messages, temp_job)
    assert cached_chunks == chunks


def test_semantic_search_paraphrase_vs_keyword(temp_job: Path, monkeypatch):
    """Semantic search correctly matches conceptual paraphrases and falls back to keywords."""
    from backend.main import settings

    chunks_data = [
        {
            "id": 0,
            "start": 0.0,
            "end": 10.0,
            "speaker": "S1",
            "text": "The solar system contains eight planets orbiting the sun.",
        },
        {
            "id": 1,
            "start": 10.0,
            "end": 20.0,
            "speaker": "S2",
            "text": "Baking cookies requires flour, sugar, butter, and vanilla extract.",
        },
    ]
    # Compute embeddings for both chunks
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode([c["text"] for c in chunks_data], normalize_embeddings=True)
    for c, v in zip(chunks_data, vectors):
        c["embedding"] = v.tolist()

    (temp_job / "chunks.json").write_text(json.dumps(chunks_data), encoding="utf-8")

    # Query with no keyword overlap: 'astronomy celestial bodies galaxy'
    hits = _search_chunks(temp_job, "astronomy celestial bodies galaxy", k=2)
    assert len(hits) > 0
    # The solar system chunk must rank first
    assert hits[0]["speaker"] == "S1"
    assert "planets" in hits[0]["text"]
    assert hits[0]["score"] > 0.3

    # Now simulate model failure: fallback to keyword search
    monkeypatch.setattr("backend.main._get_embedder", lambda: None)
    # Query with exact keyword overlap
    kw_hits = _search_chunks(temp_job, "cookies flour sugar", k=2)
    assert len(kw_hits) > 0
    assert kw_hits[0]["speaker"] == "S2"
    assert "Baking cookies" in kw_hits[0]["text"]


def test_face_thumbnail_endpoint(temp_job: Path, monkeypatch):
    """Verify avatar serving endpoint via TestClient."""
    import backend.main as b_main
    from backend.main import Settings

    new_settings = Settings(
        jobs_dir=temp_job.parent,
        cors_origins=("*",),
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
        azure_openai_endpoint="",
        azure_openai_api_key="",
        azure_openai_deployment="",
        azure_openai_api_version="2024-10-21",
    )
    monkeypatch.setattr(b_main, "settings", new_settings)
    job_id = temp_job.name
    faces_dir = temp_job / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    # 1x1 valid JPEG bytes
    tiny_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
    )
    (faces_dir / "S1.jpg").write_bytes(tiny_jpeg)

    client = TestClient(app)
    # Test standard endpoint
    resp = client.get(f"/jobs/{job_id}/faces/S1.jpg")
    assert resp.status_code == 200
    assert "image/jpeg" in resp.headers["content-type"]
    assert resp.content == tiny_jpeg

    # Test api alias endpoint
    api_resp = client.get(f"/api/jobs/{job_id}/faces/S1.jpg")
    assert api_resp.status_code == 200
    assert api_resp.content == tiny_jpeg
