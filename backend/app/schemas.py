from typing import Literal

from pydantic import BaseModel, Field

SourceMode = Literal['default', 'uploaded', 'combined']
AnswerMode = Literal['strict', 'hybrid']
PresentationMode = Literal['standard', 'exam']
ExamProfile = Literal[
    'auto',
    'two_mark',
    'five_mark',
    'six_mark',
    'seven_mark',
    'ten_mark',
    'comparison',
    'viva',
]


class SessionCreateResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    file_names: list[str]
    pages: int
    chunks: int


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    kb_label: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1)
    source_mode: SourceMode = 'default'
    answer_mode: AnswerMode = 'strict'
    presentation_mode: PresentationMode = 'exam'
    exam_profile: ExamProfile = 'auto'


class ChatResponse(BaseModel):
    answer: str
    resolved_profile: str
    sources: list[SourceChunk]
    file_names: list[str]
    source_mode: SourceMode
    answer_mode: AnswerMode
    presentation_mode: PresentationMode


class ResetResponse(BaseModel):
    session_id: str
    cleared_chat: bool
    cleared_documents: bool


class ConfigResponse(BaseModel):
    source_modes: list[str]
    answer_modes: list[str]
    presentation_modes: list[str]
    exam_profiles: list[str]
    default_knowledge_base_ready: bool
