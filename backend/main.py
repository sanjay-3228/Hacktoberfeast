"""Production FastAPI backend for the Video Knowledge Extractor.

The service does no ML work. It serves completed job folders produced by the
upstream pipeline::

    JOBS_DIR/
        {job_id}/
            video.mp4
            audio.wav
            transcript.json
            diarization.json
            result.json
            chunks.json
            faces/S1.jpg, S2.jpg, ...

Endpoints
---------
GET  /health
GET  /jobs
GET  /jobs/{job_id}/status
GET  /jobs/{job_id}/result
GET  /jobs/{job_id}/search?q=...&k=3
POST /jobs/{job_id}/ask
GET  /jobs/{job_id}/report
GET  /jobs/{job_id}/video           (HTTP Range / seeking)
GET  /jobs/{job_id}/faces/{file}    (avatars)

Run locally::

    uvicorn main:app --reload

Configuration is read from the environment (and the nearest ``.env``).
"""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Sequence

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load the nearest .env without overriding real environment variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = (
        Path.cwd() / ".env",
        _BASE_DIR / ".env",
        _BASE_DIR.parent / ".env",
    )
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _env(name: str, default: str = "") -> str:
    """Read and trim an environment variable."""
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    """Read a bounded integer environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _split_csv(raw: str, default: tuple[str, ...] = ("*",)) -> tuple[str, ...]:
    """Split a comma-separated setting, dropping empty entries."""
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration."""

    jobs_dir: Path
    cors_origins: tuple[str, ...]
    embed_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_api_version: str
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"


def load_settings() -> Settings:
    """Build settings from the environment.

    Relative ``JOBS_DIR`` values are resolved against this file's directory so
    ``JOBS_DIR=./jobs`` means ``backend/jobs`` regardless of the process cwd.
    """
    _load_env_file()

    raw_jobs_dir = Path(
        _env("JOBS_DIR") or str(_BASE_DIR / "jobs")
    ).expanduser()
    if not raw_jobs_dir.is_absolute():
        raw_jobs_dir = _BASE_DIR / raw_jobs_dir
    jobs_dir = raw_jobs_dir.resolve()

    return Settings(
        jobs_dir=jobs_dir,
        cors_origins=_split_csv(_env("CORS_ORIGINS", "*")),
        embed_model=_env("EMBED_MODEL", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2",
        llm_base_url=_env("LLM_BASE_URL"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_model=_env("LLM_MODEL"),
        azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=_env("AZURE_OPENAI_API_KEY"),
        azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_openai_api_version=_env(
            "AZURE_OPENAI_API_VERSION", "2024-10-21"
        )
        or "2024-10-21",
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b")
        or "openai/gpt-oss-120b",
    )


settings = load_settings()

_LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper() or "INFO"
if _LOG_LEVEL not in {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
}:
    _LOG_LEVEL = "INFO"
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vke.backend")

# ---------------------------------------------------------------------------
# Constants and regular expressions
# ---------------------------------------------------------------------------

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_FACE_FILE_RE = re.compile(
    r"^[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE
)
_BYTE_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

_STREAM_CHUNK_SIZE = 64 * 1024
_ASK_CONTEXT_CHUNKS = 4
_LLM_TIMEOUT_SEC = 30.0

# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness response."""

    ok: bool = True


class JobSummary(BaseModel):
    """One entry in the job listing."""

    job_id: str
    title: str = ""
    topic: str = ""
    duration: float = 0.0


class Source(BaseModel):
    """One timestamped transcript excerpt."""

    start: float
    end: float
    speaker: str = ""
    text: str
    score: float = 0.0


class SearchResponse(BaseModel):
    """Search results envelope."""

    results: list[Source]


class AskRequest(BaseModel):
    """Question payload for the Q&A endpoint."""

    question: str = Field(..., min_length=1, max_length=4000)


class AskResponse(BaseModel):
    """Answer with the excerpts it was grounded in."""

    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Job-folder access (path-safe)
# ---------------------------------------------------------------------------


def _job_dir(job_id: str) -> Path:
    """Resolve a job folder, rejecting traversal and unknown jobs.

    Raises:
        HTTPException: 404 when the id is invalid or the folder is missing.
    """
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    candidate = (settings.jobs_dir / job_id).resolve()
    if candidate.parent != settings.jobs_dir or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    return candidate


def _safe_file(job_dir: Path, *parts: str) -> Path:
    """Resolve a file inside *job_dir*, rejecting escapes and missing files."""
    target = job_dir.joinpath(*parts).resolve()
    if target == job_dir or job_dir not in target.parents:
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _face_file(job_dir: Path, filename: str) -> Path:
    """Resolve an avatar filename, allowing only flat image names."""
    if not _FACE_FILE_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="File not found")
    return _safe_file(job_dir, "faces", filename)


def _read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_result(job_dir: Path) -> dict[str, Any]:
    """Load ``result.json`` and fill harmless defaults for the frontend.

    Raises:
        HTTPException: 404 when the file is missing, 500 when it is unreadable.
    """
    path = job_dir / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        data = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        raise HTTPException(
            status_code=500, detail="Result file is unreadable"
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500, detail="Result file must contain a JSON object"
        )

    data.setdefault("job_id", job_dir.name)
    data.setdefault("title", "")
    data.setdefault("topic", "")
    data.setdefault("duration", 0.0)
    data.setdefault("speakers", [])
    data.setdefault("messages", [])
    data.setdefault("chapters", [])
    data.setdefault("takeaways", [])
    return data


def _load_chunks(job_dir: Path) -> list[dict[str, Any]]:
    """Load and sanitize ``chunks.json``.

    Malformed individual chunks are skipped. A missing file means the job has
    no search index yet; a corrupt file is a server-side data error.
    """
    path = job_dir / "chunks.json"
    if not path.is_file():
        return []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        raise HTTPException(
            status_code=500, detail="Search index is unreadable"
        ) from exc
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=500, detail="Search index must contain a JSON array"
        )

    chunks: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        embedding = item.get("embedding")
        valid_embedding = (
            isinstance(embedding, list)
            and bool(embedding)
            and all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in embedding
            )
        )
        chunks.append(
            {
                "start": start,
                "end": end,
                "speaker": str(item.get("speaker") or ""),
                "text": text,
                "embedding": embedding if valid_embedding else None,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_embedder: Any | None = None
_embedder_failed = False
_embedder_lock = threading.Lock()
_encode_lock = threading.Lock()


def _get_embedder() -> Any | None:
    """Lazily load the optional sentence-transformers model once per process."""
    global _embedder, _embedder_failed

    if _embedder is not None or _embedder_failed:
        return _embedder

    with _embedder_lock:
        if _embedder is not None or _embedder_failed:
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(settings.embed_model)
            logger.info("Loaded embedding model %s", settings.embed_model)
        except Exception as exc:  # optional dependency or no model cache
            _embedder_failed = True
            logger.warning(
                "Semantic search disabled (%s); using keyword search", exc
            )
        return _embedder


def _tokens(text: str) -> set[str]:
    """Unicode-aware lowercase word tokens."""
    return set(_WORD_RE.findall(text.lower()))


def _keyword_scores(
    query: str, chunks: Sequence[dict[str, Any]]
) -> list[float] | None:
    """Score chunks by the fraction of query words they contain."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return None

    scores: list[float] = []
    for chunk in chunks:
        overlap = len(query_tokens & _tokens(chunk["text"]))
        scores.append(overlap / len(query_tokens))
    return scores


def _semantic_scores(
    query: str, chunks: Sequence[dict[str, Any]]
) -> list[float] | None:
    """Cosine similarity between the query and chunk embeddings.

    Returns ``None`` when embeddings or the model are unavailable, so callers
    can fall back to keyword search.
    """
    vectors: list[list[float]] = []
    dimension: int | None = None
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            return None
        if dimension is None:
            dimension = len(embedding)
        elif len(embedding) != dimension:
            logger.warning("Chunk embeddings have inconsistent dimensions")
            return None
        vectors.append(embedding)

    model = _get_embedder()
    if model is None:
        return None

    try:
        import numpy as np

        with _encode_lock:
            query_vector = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

        matrix = np.asarray(vectors, dtype=np.float32)
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if (
            matrix.ndim != 2
            or query_vector.ndim != 1
            or matrix.shape[1] != query_vector.shape[0]
        ):
            logger.warning("Embedding dimensions are incompatible with chunks")
            return None

        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(row_norms, 1e-12)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm < 1e-12:
            return None
        return (matrix @ (query_vector / query_norm)).astype(float).tolist()
    except Exception as exc:
        logger.warning("Semantic scoring failed (%s); using keywords", exc)
        return None


def _clean_query(raw: str) -> str:
    """Validate a query string for search and Q&A."""
    query = raw.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if not _tokens(query):
        raise HTTPException(
            status_code=400, detail="Query must contain at least one word"
        )
    return query


def _search_chunks(
    job_dir: Path, query: str, k: int
) -> list[dict[str, Any]]:
    """Return the top-*k* positive-scoring chunks for a query."""
    chunks = _load_chunks(job_dir)
    if not chunks:
        return []

    scores = _semantic_scores(query, chunks)
    if scores is None:
        scores = _keyword_scores(query, chunks)
    if scores is None:
        raise HTTPException(
            status_code=400, detail="Query must contain at least one word"
        )

    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index]["start"]),
    )
    results: list[dict[str, Any]] = []
    for index in order:
        score = float(scores[index])
        if score <= 0.0:
            continue
        chunk = chunks[index]
        results.append(
            {
                "start": round(chunk["start"], 2),
                "end": round(chunk["end"], 2),
                "speaker": chunk["speaker"],
                "text": chunk["text"],
                "score": round(score, 4),
            }
        )
        if len(results) >= k:
            break
    return results


# ---------------------------------------------------------------------------
# Optional LLM Q&A
# ---------------------------------------------------------------------------


def _llm_client() -> tuple[Any | None, str]:
    """Build a Groq, OpenAI, or Azure client, or ``(None, "")`` when unconfigured."""
    try:
        from groq import Groq
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        logger.warning("groq/openai is not installed; LLM answers are disabled")
        return None, ""

    if settings.groq_api_key:
        return Groq(api_key=settings.groq_api_key), settings.groq_model

    if (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=_LLM_TIMEOUT_SEC,
            max_retries=1,
        )
        return client, settings.azure_openai_deployment

    if settings.llm_model and (
        settings.llm_api_key or settings.llm_base_url
    ):
        options: dict[str, Any] = {
            "api_key": settings.llm_api_key or "not-needed",
            "timeout": _LLM_TIMEOUT_SEC,
            "max_retries": 1,
        }
        if settings.llm_base_url:
            options["base_url"] = settings.llm_base_url
        return OpenAI(**options), settings.llm_model

    return None, ""


def _speaker_names(result: dict[str, Any]) -> dict[str, str]:
    """Map speaker ids to display names from ``result.json``."""
    names: dict[str, str] = {}
    speakers = result.get("speakers")
    if not isinstance(speakers, list):
        return names
    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        speaker_id = speaker.get("id")
        if isinstance(speaker_id, str) and speaker_id:
            names[speaker_id] = str(speaker.get("name") or speaker_id)
    return names


def _mmss(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _llm_answer(question: str, sources: Sequence[Source]) -> str | None:
    """Ask the configured LLM to answer from the supplied excerpts.

    Returns ``None`` when no LLM is configured or the call fails, allowing the
    endpoint to fall back to the best transcript excerpt.
    """
    client, model = _llm_client()
    if client is None:
        return None

    excerpts = "\n".join(
        f"[{index + 1}] ({_mmss(source.start)}-{_mmss(source.end)}, "
        f"{source.speaker}): {source.text}"
        for index, source in enumerate(sources)
    )
    system = (
        "You answer questions about a video using ONLY the numbered transcript "
        "excerpts provided. Cite excerpts like [1] or [2]. If the answer is not "
        "in the excerpts, say you could not find it in the video. Keep the "
        "answer short and clear."
    )
    user = f"Excerpts:\n{excerpts}\n\nQuestion: {question}"

    try:
        options: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if settings.groq_api_key:
            options.update(
                temperature=1,
                max_completion_tokens=400,
                top_p=1,
                reasoning_effort="medium",
            )
        else:
            options.update(temperature=0.2, max_tokens=400)
        response = client.chat.completions.create(**options)
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
        logger.warning("LLM returned an empty answer")
    except Exception as exc:
        logger.warning("LLM request failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a JSON value to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _build_report(job_id: str, result: dict[str, Any]) -> str:
    """Render a Markdown report from ``result.json``."""
    title = str(result.get("title") or f"Video report: {job_id}")
    duration = _safe_float(result.get("duration"))
    topic = str(result.get("topic") or "").strip()
    names = _speaker_names(result)

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"**Job:** `{job_id}`  ")
    lines.append(f"**Duration:** {_mmss(duration)}")
    lines.append("")

    if topic:
        lines.extend(["## Topic", "", topic, ""])

    speakers = result.get("speakers")
    if isinstance(speakers, list) and speakers:
        lines.extend(["## Speakers", ""])
        for speaker in speakers:
            if not isinstance(speaker, dict):
                continue
            speaker_id = str(speaker.get("id") or "")
            name = str(speaker.get("name") or speaker_id or "Unknown")
            location = (
                "on camera" if speaker.get("on_camera") else "off camera"
            )
            percentage = _safe_float(speaker.get("talk_time_pct"))
            lines.append(
                f"- **{name}** ({location}) — {percentage:g}% of talk time"
            )
        lines.append("")

    chapters = result.get("chapters")
    if isinstance(chapters, list) and chapters:
        lines.extend(["## Chapters", ""])
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            start = _safe_float(chapter.get("start"))
            chapter_title = str(chapter.get("title") or "Chapter")
            summary = str(chapter.get("summary") or "").strip()
            entry = f"- **{_mmss(start)} — {chapter_title}**"
            if summary:
                entry += f"  \n  {summary}"
            lines.append(entry)
        lines.append("")

    takeaways = result.get("takeaways")
    if isinstance(takeaways, list) and takeaways:
        lines.extend(["## Key takeaways", ""])
        for takeaway in takeaways:
            text = str(takeaway).strip()
            if text:
                lines.append(f"- {text}")
        lines.append("")

    messages = result.get("messages")
    if isinstance(messages, list) and messages:
        lines.extend(["## Transcript", ""])
        for message in messages:
            if not isinstance(message, dict):
                continue
            start = _safe_float(message.get("start"))
            speaker_id = str(message.get("speaker") or "")
            speaker_name = names.get(speaker_id, speaker_id or "Unknown")
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            lines.append(
                f"**[{_mmss(start)}] {speaker_name}:** {text}"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Range-enabled file streaming
# ---------------------------------------------------------------------------


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    """Yield *length* bytes of *path*, beginning at byte *start*."""
    remaining = length
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            while remaining > 0:
                block = handle.read(min(_STREAM_CHUNK_SIZE, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
    except OSError as exc:
        # Headers are already sent at this point; log instead of raising.
        logger.warning("Streaming interrupted for %s: %s", path, exc)


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single byte range; raise 416 when it is unsatisfiable."""
    unsatisfiable = HTTPException(
        status_code=416,
        detail="Requested range is not satisfiable",
        headers={"Content-Range": f"bytes */{size}"},
    )
    match = _BYTE_RANGE_RE.fullmatch(header.strip())
    if not match:
        raise unsatisfiable

    start_raw, end_raw = match.groups()
    if not start_raw and not end_raw:
        raise unsatisfiable

    if not start_raw:  # suffix form: bytes=-500
        try:
            suffix = int(end_raw)
        except ValueError as exc:
            raise unsatisfiable from exc
        if suffix <= 0:
            raise unsatisfiable
        start = max(size - suffix, 0)
        end = size - 1
    else:
        try:
            start = int(start_raw)
        except ValueError as exc:
            raise unsatisfiable from exc
        if start < 0 or start >= size:
            raise unsatisfiable
        if end_raw:
            try:
                end = int(end_raw)
            except ValueError as exc:
                raise unsatisfiable from exc
            end = min(end, size - 1)
            if end < start:
                raise unsatisfiable
        else:
            end = size - 1

    return start, end


def _stream_file(request: Request, path: Path) -> Response:
    """Serve a file with ``Accept-Ranges`` and HTTP 206 partial responses."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc

    media_type = (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }

    if size == 0:
        headers["Content-Length"] = "0"
        return Response(
            status_code=200, media_type=media_type, headers=headers
        )

    range_header = request.headers.get("range")
    if not range_header or not range_header.strip().lower().startswith(
        "bytes="
    ):
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _iter_file(path, 0, size),
            media_type=media_type,
            headers=headers,
        )

    start, end = _parse_range(range_header, size)
    length = end - start + 1
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(
        _iter_file(path, start, length),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Application lifecycle and middleware
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log the configured job root at startup."""
    logger.info("Serving job folders from %s", settings.jobs_dir)
    if not settings.jobs_dir.is_dir():
        logger.warning(
            "Jobs directory does not exist yet: %s", settings.jobs_dir
        )
    yield


app = FastAPI(
    title="Video Knowledge Extractor API",
    version="1.0.0",
    description=(
        "Reads completed video job folders and serves transcripts, search, "
        "grounded Q&A, reports and range-enabled media."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Turn FastAPI's 422 validation response into a clear 400."""
    try:
        errors = jsonable_encoder(exc.errors())
    except (TypeError, ValueError):
        errors = [{"msg": "One or more fields are invalid"}]
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request", "errors": errors},
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a JSON 500 without leaking internals to the client."""
    logger.exception(
        "Unhandled error on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness probe",
)
def health() -> HealthResponse:
    """Return ``{"ok": true}`` when the service is running."""
    return HealthResponse(ok=True)


@app.get(
    "/jobs",
    response_model=list[JobSummary],
    tags=["jobs"],
    summary="List available jobs",
)
def list_jobs() -> list[JobSummary]:
    """Scan ``JOBS_DIR`` and list jobs that have a readable ``result.json``."""
    root = settings.jobs_dir
    if not root.is_dir():
        return []

    try:
        entries = sorted(
            (entry for entry in root.iterdir() if entry.is_dir()),
            key=lambda entry: entry.name.lower(),
        )
    except OSError as exc:
        logger.exception("Could not scan jobs directory %s", root)
        raise HTTPException(
            status_code=500, detail="Could not scan jobs directory"
        ) from exc

    jobs: list[JobSummary] = []
    for entry in entries:
        if not _JOB_ID_RE.fullmatch(entry.name):
            continue
        # Keep the listing consistent with _job_dir(), which rejects a job
        # folder that resolves outside JOBS_DIR (for example via symlink).
        try:
            if entry.resolve().parent != root:
                logger.warning("Skipping out-of-root job entry: %s", entry)
                continue
        except OSError:
            continue
        result_path = entry / "result.json"
        if not result_path.is_file():
            continue
        try:
            data = _read_json(result_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable job %s: %s", entry.name, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping job with non-object result: %s", entry.name)
            continue
        jobs.append(
            JobSummary(
                job_id=entry.name,
                title=str(data.get("title") or ""),
                topic=str(data.get("topic") or ""),
                duration=_safe_float(data.get("duration")),
            )
        )
    return jobs


@app.get(
    "/jobs/{job_id}/status",
    tags=["jobs"],
    summary="Job readiness",
)
def get_status(job_id: str) -> dict[str, Any]:
    """Report whether the job's final result is ready to serve."""
    job_dir = _job_dir(job_id)
    if not (job_dir / "result.json").is_file():
        return {"status": "processing", "progress": 0}
    return {"status": "done", "progress": 100}


@app.get(
    "/jobs/{job_id}/result",
    tags=["jobs"],
    summary="Full result.json",
)
def get_result(job_id: str) -> dict[str, Any]:
    """Return the complete ``result.json`` for a completed job."""
    return _load_result(_job_dir(job_id))


@app.get(
    "/jobs/{job_id}/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic or keyword transcript search",
)
def search(
    job_id: str,
    q: str = Query(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language or keyword query",
    ),
    k: int = Query(
        3, ge=1, le=50, description="Maximum number of excerpts to return"
    ),
) -> SearchResponse:
    """Search ``chunks.json`` and return the best matches with timestamps.

    Uses cosine similarity when sentence-transformers and chunk embeddings are
    available; otherwise falls back to keyword overlap.
    """
    job_dir = _job_dir(job_id)
    query = _clean_query(q)
    hits = _search_chunks(job_dir, query, k)
    return SearchResponse(results=[Source(**hit) for hit in hits])


@app.post(
    "/jobs/{job_id}/ask",
    response_model=AskResponse,
    tags=["search"],
    summary="Ask a question grounded in the transcript",
)
def ask(job_id: str, body: AskRequest) -> AskResponse:
    """Answer a question from the video transcript.

    The best matching chunks are used as context for a configured LLM. When no
    LLM is configured (or the request fails), the most relevant excerpt is
    returned instead, so the endpoint always degrades gracefully.
    """
    job_dir = _job_dir(job_id)
    question = _clean_query(body.question)

    hits = _search_chunks(job_dir, question, _ASK_CONTEXT_CHUNKS)
    if not hits:
        return AskResponse(
            answer=(
                "No relevant transcript excerpts were found for this question."
            ),
            sources=[],
        )

    try:
        names = _speaker_names(_load_result(job_dir))
    except HTTPException:
        names = {}

    sources = [
        Source(
            start=hit["start"],
            end=hit["end"],
            speaker=names.get(hit["speaker"], hit["speaker"]),
            text=hit["text"],
            score=hit["score"],
        )
        for hit in hits
    ]

    answer = _llm_answer(question, sources)
    if answer is None:
        best = sources[0]
        answer = (
            "(Showing the most relevant transcript excerpt.) "
            f"[{_mmss(best.start)}] {best.speaker}: {best.text}"
        )
    return AskResponse(answer=answer, sources=sources)


@app.get(
    "/jobs/{job_id}/report",
    tags=["reports"],
    summary="Markdown report",
)
def get_report(job_id: str) -> Response:
    """Return a human-readable Markdown report built from ``result.json``."""
    job_dir = _job_dir(job_id)
    result = _load_result(job_dir)
    report = _build_report(job_dir.name, result)
    return Response(
        content=report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'inline; filename="{job_dir.name}-report.md"'
        },
    )


@app.get(
    "/jobs/{job_id}/video",
    tags=["media"],
    summary="Stream the video with byte-range support",
)
def get_video(job_id: str, request: Request) -> Response:
    """Serve ``video.mp4``, honoring single HTTP byte ranges for seeking."""
    job_dir = _job_dir(job_id)
    path = _safe_file(job_dir, "video.mp4")
    return _stream_file(request, path)


@app.get(
    "/jobs/{job_id}/faces/{filename}",
    tags=["media"],
    summary="Serve an avatar crop",
)
def get_face(job_id: str, filename: str, request: Request) -> Response:
    """Serve a flat avatar image from ``faces/`` (traversal-safe)."""
    job_dir = _job_dir(job_id)
    path = _face_file(job_dir, filename)
    return _stream_file(request, path)
