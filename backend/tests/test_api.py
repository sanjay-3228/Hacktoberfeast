"""API tests for the Video Knowledge Extractor backend.

Builds a throwaway ``JOBS_DIR`` per test (``tmp_path``), points the app's
``settings`` at it, and exercises every documented endpoint: happy paths,
404/400/416/500 errors, path-traversal defense, search and Q&A fallbacks,
semantic scoring with a stubbed embedder, and HTTP byte-range streaming.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import main
from main import Settings

# ---------------------------------------------------------------------------
# Fixtures: a temporary job tree wired into the app settings
# ---------------------------------------------------------------------------

RESULT_FULL: dict[str, Any] = {
    "job_id": "demo",
    "title": "Building a campus study-group app",
    "topic": "A short interview about a study-group app.",
    "duration": 120.0,
    "speakers": [
        {"id": "S1", "name": "Priya", "on_camera": True, "avatar": "faces/S1.jpg", "talk_time_pct": 54},
        {"id": "S3", "name": "Speaker 3", "on_camera": False, "avatar": None, "talk_time_pct": 22},
    ],
    "messages": [
        {"speaker": "S1", "start": 0.0, "end": 42.0, "text": "The core idea is to match students with overlapping free time."},
        {"speaker": "S3", "start": 42.0, "end": 85.0, "text": "We also rely on a reliability score for people who cancel late."},
        {"speaker": "S1", "start": 85.0, "end": 120.0, "text": "Timetables stay on the device until a match is confirmed."},
    ],
    "chapters": [{"title": "Introduction", "start": 0.0, "end": 120.0, "summary": "Overview."}],
    "takeaways": ["Match by shared course and free time."],
}

CHUNKS_FULL: list[dict[str, Any]] = [
    {"id": 0, "start": 0.0, "end": 42.0, "speaker": "S1", "text": "match students with overlapping free time", "embedding": None},
    {"id": 1, "start": 42.0, "end": 85.0, "speaker": "S3", "text": "reliability score for people who cancel late", "embedding": None},
    {"id": 2, "start": 85.0, "end": 120.0, "speaker": "S1", "text": "timetables stay on the device until confirmed", "embedding": None},
]

VIDEO_BYTES = b"\x00\x01\x02mp4data" * 300  # 2100 bytes
JPG_BYTES = b"\xff\xd8\xff\xe0-fake-jpeg" * 5


def _write_job(
    root: Any,
    job_id: str,
    *,
    result: Any = RESULT_FULL,
    chunks: Any = CHUNKS_FULL,
    video: bytes = VIDEO_BYTES,
    faces: dict[str, bytes] | None = None,
) -> None:
    """Create a job folder with the files the backend expects."""
    job = root / job_id
    (job / "faces").mkdir(parents=True)
    if result is not None:
        (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if chunks is not None:
        (job / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    if video is not None:
        (job / "video.mp4").write_bytes(video)
    if faces is None:
        faces = {"S1.jpg": JPG_BYTES, "S2.jpg": JPG_BYTES}
    for name, payload in faces.items():
        (job / "faces" / name).write_bytes(payload)


@pytest.fixture()
def jobs_dir(tmp_path: Any) -> Any:
    """A populated jobs directory: good, minimal, and corrupt jobs."""
    root = tmp_path / "jobs"
    root.mkdir()
    _write_job(root, "demo")
    _write_job(
        root,
        "minimal-result",
        result={"topic": "bare result without title or job_id", "duration": 30.0},
        chunks=[{"id": 0, "start": 0.0, "end": 30.0, "speaker": "S1", "text": "alpha beta gamma", "embedding": None}],
        faces={},
    )
    _write_job(root, "corrupt-result", result=None, chunks=None)
    (root / "corrupt-result" / "result.json").write_text("{ not json", encoding="utf-8")
    _write_job(root, "non-object-result", result=None, chunks=None)
    (root / "non-object-result" / "result.json").write_text("[1, 2, 3]", encoding="utf-8")
    _write_job(root, "mixed-chunks", chunks=[
        "not-a-dict",
        {"start": "nan", "end": 10.0, "speaker": "S1", "text": "valid text ignored"},
        {"start": 10.0, "end": 5.0, "speaker": "S1", "text": "end before start"},
        {"start": 10.0, "end": 20.0, "speaker": "S1", "text": "   "},
        {"start": 15.0, "end": 25.0, "speaker": "S2", "text": "real chunk about deadlines"},
        {"start": 30.0, "end": 40.0, "speaker": "S1", "text": "another chunk about reducers"},
        {"start": 300.0, "end": 310.0, "speaker": "S1", "text": "embedding with heterogeneous values", "embedding": [1.0, "x"]},
        {"start": 320.0, "end": 330.0, "speaker": "S1", "text": "chunk with numeric embedding", "embedding": [1.0, 0.0]},
    ])
    (root / "Sneaky-Name").mkdir()  # invalid job id pattern, no result.json
    return root


@pytest.fixture()
def client(jobs_dir: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient wired to the temporary jobs directory."""
    monkeypatch.setattr(
        main,
        "settings",
        Settings(
            jobs_dir=jobs_dir,
            cors_origins=("*",),
            embed_model="test-model",
            llm_base_url="",
            llm_api_key="",
            llm_model="",
            azure_openai_endpoint="",
            azure_openai_api_key="",
            azure_openai_deployment="",
            azure_openai_api_version="2024-10-21",
        ),
    )
    monkeypatch.setattr(main, "_embedder", None)
    monkeypatch.setattr(main, "_embedder_failed", False)
    with TestClient(main.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health, listing, status
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_list_jobs_skips_bad_entries(client: TestClient) -> None:
    response = client.get("/jobs")
    assert response.status_code == 200
    ids = [job["job_id"] for job in response.json()]
    assert ids == ["demo", "minimal-result", "mixed-chunks"]  # sorted, corrupt skipped


def test_list_jobs_minimal_title_is_blank(client: TestClient) -> None:
    summary = [job for job in client.get("/jobs").json() if job["job_id"] == "minimal-result"][0]
    assert summary["title"] == ""
    assert summary["duration"] == 30.0


def test_list_jobs_when_jobs_dir_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(
        main,
        "settings",
        Settings(jobs_dir=tmp_path / "does-not-exist", cors_origins=("*",), embed_model="m",
                 llm_base_url="", llm_api_key="", llm_model="", azure_openai_endpoint="",
                 azure_openai_api_key="", azure_openai_deployment="", azure_openai_api_version="2024-10-21"),
    )
    with TestClient(main.app) as test_client:
        response = test_client.get("/jobs")
    assert response.status_code == 200
    assert response.json() == []


def test_status_done_and_processing(client: TestClient, jobs_dir: Any) -> None:
    (jobs_dir / "no-result-yet").mkdir()
    assert client.get("/jobs/demo/status").json() == {"status": "done", "progress": 100}
    assert client.get("/jobs/no-result-yet/status").json() == {"status": "processing", "progress": 0}
    assert client.get("/jobs/nope/status").status_code == 404


def test_create_job_saves_upload_and_queues_pipeline(
    client: TestClient, jobs_dir: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: dict[str, Any] = {}

    class FakeExecutor:
        def submit(self, function: Any, *args: Any, **kwargs: Any) -> None:
            submitted["function"] = function
            submitted["args"] = args
            submitted["kwargs"] = kwargs

    monkeypatch.setattr(main, "_pipeline_executor", FakeExecutor())
    response = client.post(
        "/jobs",
        files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
        data={"title": "Uploaded clip", "use_faces": "false", "use_llm": "false", "embed": "false"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "processing"
    job_id = payload["job_id"]
    assert job_id.startswith("job-")
    assert (jobs_dir / job_id / "upload_title.txt").read_text(encoding="utf-8") == "Uploaded clip"
    assert submitted["args"][0] == job_id
    assert submitted["kwargs"] == {"use_faces": False, "use_llm": False, "embed": False, "num_speakers": None}


# ---------------------------------------------------------------------------
# result.json serving and path-safety
# ---------------------------------------------------------------------------


def test_result_returns_full_document(client: TestClient) -> None:
    data = client.get("/jobs/demo/result").json()
    assert data["title"] == RESULT_FULL["title"]
    assert data["speakers"][0]["name"] == "Priya"
    assert data["messages"][0]["start"] == 0.0


def test_result_fills_defaults_for_minimal_document(client: TestClient) -> None:
    data = client.get("/jobs/minimal-result/result").json()
    assert data["job_id"] == "minimal-result"
    assert data["title"] == ""
    assert data["chapters"] == []
    assert data["takeaways"] == []


def test_result_missing_job_is_404(client: TestClient) -> None:
    response = client.get("/jobs/ghost/result")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_result_rejects_invalid_job_ids(client: TestClient) -> None:
    for bad in ("bad$id", "..%2Fsecret", "a%2Fb", "sp%20ace"):
        response = client.get(f"/jobs/{bad}/result")
        # Encoded slashes decode before routing, yielding the router's generic
        # "Not Found"; other invalid ids reach the handler's "Job not found".
        assert response.status_code == 404, bad
        assert response.json()["detail"] in {"Job not found", "Not Found"}


def test_result_corrupt_json_is_500(client: TestClient) -> None:
    response = client.get("/jobs/corrupt-result/result")
    assert response.status_code == 500
    assert response.json()["detail"] == "Result file is unreadable"


def test_result_non_object_is_500(client: TestClient) -> None:
    response = client.get("/jobs/non-object-result/result")
    assert response.status_code == 500
    assert response.json()["detail"] == "Result file must contain a JSON object"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_keyword_fallback_orders_and_limits(client: TestClient) -> None:
    response = client.get("/jobs/demo/search", params={"q": "reliability score cancel", "k": 3})
    assert response.status_code == 200
    hits = response.json()["results"]
    assert hits and hits[0]["start"] == 42.0
    assert hits[0]["score"] >= hits[-1]["score"]
    for hit in hits:
        assert set(hit) == {"start", "end", "speaker", "text", "score"}


def test_search_default_k_is_three(client: TestClient) -> None:
    hits = client.get("/jobs/demo/search", params={"q": "students free time device"}).json()["results"]
    assert 0 < len(hits) <= 3


def test_search_ranks_semantically_with_stubbed_embedder(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def encode(self, texts: list[str], **_: Any) -> list[Any]:
            return [[0.9, 0.1] for _ in texts]

    monkeypatch.setattr(main, "_embedder", FakeModel())
    monkeypatch.setattr(main, "_embedder_failed", False)
    # All three demo chunks get keyword scores 0; semantic must decide the order.
    vectors = [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]
    chunks = [
        {"start": 0.0, "end": 10.0, "speaker": "S1", "text": "zzz unshared words", "embedding": list(vectors[0])},
        {"start": 10.0, "end": 20.0, "speaker": "S1", "text": "zzz unshared words", "embedding": list(vectors[1])},
        {"start": 20.0, "end": 30.0, "speaker": "S1", "text": "zzz unshared words", "embedding": list(vectors[2])},
    ]
    # Point search at demo but with embed-only chunks via a dedicated job.
    job = main.settings.jobs_dir / "embed-job"
    (job / "faces").mkdir(parents=True)
    (job / "result.json").write_text(json.dumps({"topic": "t", "duration": 30.0}), encoding="utf-8")
    (job / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (job / "video.mp4").write_bytes(b"x")

    hits = client.get("/jobs/embed-job/search", params={"q": "probe"}).json()["results"]
    assert [hit["start"] for hit in hits] == [0.0, 20.0, 10.0]  # cosine order (1,0) > (0.707,0.707) > (0,1)


def test_search_inconsistent_embeddings_falls_back_to_keywords(client: TestClient) -> None:
    response = client.get("/jobs/mixed-chunks/search", params={"q": "chunk deadlines reducers", "k": 5})
    assert response.status_code == 200
    texts = [hit["text"] for hit in response.json()["results"]]
    assert "real chunk about deadlines" in texts
    assert "okay this is filler" not in [text for text in texts]  # invalid chunks skipped


def test_search_mixed_chunks_skip_invalid_entries(client: TestClient) -> None:
    hits = client.get("/jobs/mixed-chunks/search", params={"q": "filler", "k": 5}).json()["results"]
    assert hits == []


def test_search_missing_chunks_is_empty_not_error(client: TestClient, jobs_dir: Any) -> None:
    _write_job(jobs_dir, "nochunks", chunks=None, faces={})
    response = client.get("/jobs/nochunks/search", params={"q": "students free time"})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_search_no_hits_returns_empty_list(client: TestClient) -> None:
    response = client.get("/jobs/demo/search", params={"q": "quantumxyx zzyyx"})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_search_rejects_empty_or_unwordy_queries(client: TestClient) -> None:
    assert client.get("/jobs/demo/search", params={"q": "   "}).status_code == 400
    assert client.get("/jobs/demo/search", params={"q": "### !!!"}).status_code == 400


def test_search_rejects_out_of_range_k(client: TestClient) -> None:
    assert client.get("/jobs/demo/search", params={"q": "students", "k": 0}).status_code == 400
    assert client.get("/jobs/demo/search", params={"q": "students", "k": 999}).status_code == 400
    assert client.get("/jobs/demo/search", params={"q": "students", "k": "many"}).status_code == 400


def test_search_validation_missing_q_returns_400_not_422(client: TestClient) -> None:
    response = client.get("/jobs/demo/search")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request"


def test_search_unknown_job_404(client: TestClient) -> None:
    assert client.get("/jobs/ghost/search", params={"q": "text"}).status_code == 404


# ---------------------------------------------------------------------------
# Ask (Q&A with grounded excerpts)
# ---------------------------------------------------------------------------


def test_ask_without_llm_returns_labeled_excerpt(client: TestClient) -> None:
    # Keyword search matches whole words (no stemming), so ask with exact words
    # from the transcript chunks.
    response = client.post("/jobs/demo/ask", json={"question": "reliability score cancel late"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("(Showing the most relevant transcript excerpt.)")
    assert payload["sources"][0]["speaker"] == "Speaker 3"  # display names resolved
    assert payload["sources"][0]["start"] == pytest.approx(42.0)
    assert set(payload["sources"][0]) == {"start", "end", "speaker", "text", "score"}


def test_ask_no_relevant_chunks_returns_clear_negative(client: TestClient) -> None:
    response = client.post("/jobs/demo/ask", json={"question": "quantumxyx zzyyx"})
    assert response.status_code == 200
    assert response.json() == {"answer": "No relevant transcript excerpts were found for this question.", "sources": []}


def test_ask_empty_question_is_400(client: TestClient) -> None:
    assert client.post("/jobs/demo/ask", json={"question": "   "}).status_code == 400
    assert client.post("/jobs/demo/ask", json={"question": ""}).status_code == 400
    assert client.post("/jobs/demo/ask", json={"wrong": "x"}).status_code == 400


def test_ask_with_llm_returns_grounding_sources(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="They use a reliability score [1]."))]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(main, "_llm_client", lambda: (FakeClient(), "fake-model"))
    response = client.post("/jobs/demo/ask", json={"question": "what about people who cancel?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "They use a reliability score [1]."
    assert payload["sources"]
    user_prompt = captured["messages"][1]["content"]
    assert "Question: what about people who cancel?" in user_prompt
    assert "reliability score" in user_prompt


def test_ask_llm_failure_falls_back_to_excerpt(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenCompletions:
        def create(self, **_: Any) -> Any:
            raise RuntimeError("network down")

    class BrokenClient:
        chat = SimpleNamespace(completions=BrokenCompletions())

    monkeypatch.setattr(main, "_llm_client", lambda: (BrokenClient(), "fake-model"))
    response = client.post("/jobs/demo/ask", json={"question": "students free time"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("(Showing the most relevant transcript excerpt.)")


def test_ask_unknown_job_404(client: TestClient) -> None:
    assert client.post("/jobs/ghost/ask", json={"question": "text"}).status_code == 404


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_contains_core_sections(client: TestClient) -> None:
    response = client.get("/jobs/demo/report")
    assert response.status_code == 200
    body = response.text
    assert "# Building a campus study-group app" in body
    assert "## Topic" in body
    assert "## Speakers" in body
    assert "Priya" in body
    assert "## Chapters" in body
    assert "## Key takeaways" in body
    assert "## Transcript" in body
    assert "text/markdown" in response.headers["content-type"]


def test_report_missing_job_404(client: TestClient) -> None:
    assert client.get("/jobs/ghost/report").status_code == 404


def test_report_corrupt_result_500(client: TestClient) -> None:
    response = client.get("/jobs/corrupt-result/report")
    assert response.status_code == 500
    assert response.json()["detail"] == "Result file is unreadable"


# ---------------------------------------------------------------------------
# Media streaming and byte ranges
# ---------------------------------------------------------------------------


def test_video_full_response_headers(client: TestClient) -> None:
    response = client.get("/jobs/demo/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == str(len(VIDEO_BYTES))
    assert response.content == VIDEO_BYTES


def test_video_byte_range_returns_206_with_correct_slice(client: TestClient) -> None:
    response = client.get("/jobs/demo/video", headers={"Range": "bytes=100-199"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 100-199/{len(VIDEO_BYTES)}"
    assert response.headers["content-length"] == "100"
    assert response.content == VIDEO_BYTES[100:200]


def test_video_open_ended_range_to_eof(client: TestClient) -> None:
    response = client.get("/jobs/demo/video", headers={"Range": "bytes=2000-"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 2000-{len(VIDEO_BYTES) - 1}/{len(VIDEO_BYTES)}"
    assert response.content == VIDEO_BYTES[2000:]


def test_video_suffix_range_returns_tail(client: TestClient) -> None:
    response = client.get("/jobs/demo/video", headers={"Range": "bytes=-100"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes {len(VIDEO_BYTES) - 100}-{len(VIDEO_BYTES) - 1}/{len(VIDEO_BYTES)}"
    assert response.content == VIDEO_BYTES[-100:]


def test_video_range_end_clamped_to_size(client: TestClient) -> None:
    response = client.get("/jobs/demo/video", headers={"Range": "bytes=0-99999"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-{len(VIDEO_BYTES) - 1}/{len(VIDEO_BYTES)}"
    assert response.content == VIDEO_BYTES


def test_video_unsatisfiable_range_returns_416(client: TestClient) -> None:
    response = client.get("/jobs/demo/video", headers={"Range": f"bytes={len(VIDEO_BYTES)}-"})
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(VIDEO_BYTES)}"


def test_video_unsupported_range_unit_is_ignored(client: TestClient) -> None:
    # RFC 9110: an unsupported range unit is ignored, not a 416.
    response = client.get("/jobs/demo/video", headers={"Range": "chunks=0-1"})
    assert response.status_code == 200
    assert response.content == VIDEO_BYTES


def test_video_malformed_and_multi_ranges_rejected(client: TestClient) -> None:
    for bad in ("bytes=abc-def", "bytes=-", "bytes=0-1,5-9", "bytes=100-50"):
        response = client.get("/jobs/demo/video", headers={"Range": bad})
        assert response.status_code == 416, bad


def test_video_missing_file_404(client: TestClient, jobs_dir: Any) -> None:
    (jobs_dir / "novideo" / "faces").mkdir(parents=True)
    (jobs_dir / "novideo" / "result.json").write_text(json.dumps({"topic": "t"}), encoding="utf-8")
    assert client.get("/jobs/novideo/video").status_code == 404


def test_avatar_served_with_image_type_and_range(client: TestClient) -> None:
    response = client.get("/jobs/demo/faces/S1.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    ranged = client.get("/jobs/demo/faces/S1.jpg", headers={"Range": "bytes=0-5"})
    assert ranged.status_code == 206
    assert ranged.content == JPG_BYTES[:6]


def test_avatar_rejects_traversal_and_bad_names(client: TestClient) -> None:
    assert client.get("/jobs/demo/faces/..%2Fvideo.mp4").status_code == 404
    assert client.get("/jobs/demo/faces/sub%2Fsecret.jpg").status_code == 404
    assert client.get("/jobs/demo/faces/s1.png.exe").status_code == 404
    assert client.get("/jobs/demo/faces/nothere.jpg").status_code == 404


def test_avatar_rejects_missing_face_file(client: TestClient) -> None:
    assert client.get("/jobs/demo/faces/S9.jpg").status_code == 404


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_allows_configured_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://example.com"})
    assert response.headers.get("access-control-allow-origin") == "*"  # wildcard mode
    assert "access-control-allow-credentials" not in response.headers  # wildcard w/o credentials
