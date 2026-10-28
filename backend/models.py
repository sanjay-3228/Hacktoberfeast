from typing import Optional
from pydantic import BaseModel, Field


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
    language: Optional[str] = None
    language_mode: Optional[str] = None
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    transcription_model: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    title: str = ""
    topic: str = ""
    duration: float = 0


class Source(BaseModel):
    segment_id: str = ""
    start: float
    end: float
    speaker_id: str = ""
    speaker: str = ""
    text: str
    score: float = 0
    evidence_id: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    new_conversation: bool = False
    speaker_id: Optional[str] = Field(default=None, max_length=100)
    start_time: Optional[float] = Field(default=None, ge=0)
    end_time: Optional[float] = Field(default=None, ge=0)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    conversation_id: str = ""
    answer_type: str = "insufficient"
    model: Optional[str] = None
    status: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[Source]
