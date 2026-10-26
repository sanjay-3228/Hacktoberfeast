"""Semantic search over chunks.json, with a keyword fallback."""
import re

import numpy as np

import config
from services.jobs import load_chunks

_embedder = None
_embedder_failed = False


def _get_embedder():
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    except Exception:
        _embedder_failed = True
    return _embedder


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_scores(query: str, chunks: list[dict]) -> list[float]:
    q = _tokens(query)
    if not q:
        return [0.0] * len(chunks)
    return [len(q & _tokens(c["text"])) / len(q) for c in chunks]


def _semantic_scores(query: str, chunks: list[dict]):
    if not chunks or any(not c.get("embedding") for c in chunks):
        return None
    model = _get_embedder()
    if model is None:
        return None
    q = model.encode([query], normalize_embeddings=True)[0]
    m = np.array([c["embedding"] for c in chunks], dtype="float32")
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return (m @ q).tolist()


def search(job_id: str, query: str, k: int = 5) -> list[dict]:
    chunks = load_chunks(job_id)
    if not chunks:
        return []
    scores = _semantic_scores(query, chunks)
    if scores is None:
        scores = _keyword_scores(query, chunks)
    order = np.argsort(scores)[::-1][:k]
    out = []
    for i in order:
        c = chunks[int(i)]
        out.append({
            "start": c["start"],
            "end": c["end"],
            "speaker": c.get("speaker", ""),
            "text": c["text"],
            "score": round(float(scores[int(i)]), 4),
        })
    return out
