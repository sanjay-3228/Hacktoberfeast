"""Read job folders from disk."""
import json
import re
from pathlib import Path

from config import JOBS_DIR

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def job_path(job_id: str) -> Path:
    if not _ID_RE.match(job_id):
        raise FileNotFoundError(job_id)
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise FileNotFoundError(job_id)
    return path


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_result(job_id: str) -> dict:
    data = _read_json(job_path(job_id) / "result.json")
    data.setdefault("job_id", job_id)
    return data


def load_chunks(job_id: str) -> list[dict]:
    path = job_path(job_id) / "chunks.json"
    return _read_json(path) if path.exists() else []


def list_jobs() -> list[dict]:
    jobs = []
    if not JOBS_DIR.is_dir():
        return jobs
    for p in sorted(JOBS_DIR.iterdir()):
        if p.is_dir() and (p / "result.json").exists():
            try:
                r = _read_json(p / "result.json")
            except Exception:
                continue
            jobs.append({
                "job_id": p.name,
                "title": r.get("title", ""),
                "topic": r.get("topic", ""),
                "duration": r.get("duration", 0),
            })
    return jobs


def file_in_job(job_id: str, relative: str) -> Path:
    """Safe path inside a job folder (blocks ../ tricks)."""
    base = job_path(job_id).resolve()
    target = (base / relative).resolve()
    if base not in target.parents and target != base:
        raise FileNotFoundError(relative)
    if not target.is_file():
        raise FileNotFoundError(relative)
    return target
