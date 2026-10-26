"""FastAPI server: serves finished job folders + live search/Q&A.

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

import config
from models import (AskRequest, AskResponse, JobSummary, Result,
                    SearchResponse)
from services import jobs, qa, report, search as search_service

app = FastAPI(title="Video Knowledge Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _not_found(_: FileNotFoundError):
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs():
    return jobs.list_jobs()


@app.get("/jobs/{job_id}/status")
def status(job_id: str):
    try:
        jobs.job_path(job_id)
    except FileNotFoundError as e:
        _not_found(e)
    # Plan A: jobs are pre-processed, so they are always done.
    return {"status": "done", "progress": 100}


@app.get("/jobs/{job_id}/result", response_model=Result)
def result(job_id: str):
    try:
        return jobs.load_result(job_id)
    except FileNotFoundError as e:
        _not_found(e)


@app.get("/jobs/{job_id}/video")
def video(job_id: str):
    try:
        return FileResponse(jobs.file_in_job(job_id, "video.mp4"), media_type="video/mp4")
    except FileNotFoundError as e:
        _not_found(e)


@app.get("/jobs/{job_id}/faces/{filename}")
def face(job_id: str, filename: str):
    try:
        return FileResponse(jobs.file_in_job(job_id, f"faces/{filename}"))
    except FileNotFoundError as e:
        _not_found(e)


@app.get("/jobs/{job_id}/search", response_model=SearchResponse)
def search(job_id: str, q: str = Query(..., min_length=1), k: int = 5):
    try:
        jobs.job_path(job_id)
    except FileNotFoundError as e:
        _not_found(e)
    return {"results": search_service.search(job_id, q, k)}


@app.post("/jobs/{job_id}/ask", response_model=AskResponse)
def ask(job_id: str, body: AskRequest):
    try:
        jobs.job_path(job_id)
    except FileNotFoundError as e:
        _not_found(e)
    return qa.ask(job_id, body.question)


@app.get("/jobs/{job_id}/report", response_class=PlainTextResponse)
def get_report(job_id: str):
    try:
        return report.build_report(job_id)
    except FileNotFoundError as e:
        _not_found(e)
