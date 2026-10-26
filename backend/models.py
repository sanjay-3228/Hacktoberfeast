from typing import Optional
from pydantic import BaseModel


class Speaker(BaseModel):
    id: str
    name: str
    on_camera: bool
    avatar: Optional[str] = None
    talk_time_pct: float = 0


class Message(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class Chapter(BaseModel):
    title: str
    start: float
    end: float
    summary: str = ""


class Result(BaseModel):
    job_id: str = ""
    title: str = ""
    topic: str = ""
    duration: float = 0
    speakers: list[Speaker] = []
    messages: list[Message] = []
    chapters: list[Chapter] = []
    takeaways: list[str] = []


class JobSummary(BaseModel):
    job_id: str
    title: str = ""
    topic: str = ""
    duration: float = 0


class Source(BaseModel):
    start: float
    end: float
    speaker: str = ""
    text: str
    score: float = 0


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchResponse(BaseModel):
    results: list[Source]
