from pydantic import BaseModel, Field
from typing import Any

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    path: str
    error: str | None = None
    temperature: float | None = None
    classes: list[str] = Field(default_factory=list)
    tta: bool | None = None
    device: str | None = None

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str | None = None
    page: int | str | None = None
    element_id: str | None = None
    kind: str = "document"
    region_type: str | None = None
    bbox: list[float] | None = None
    text_source: str | None = None
    layout_confidence: float | None = None
    reading_order: int | None = None
    score: float | None = None

class RetrievedChunk(Chunk):
    rank: int

class AnalysisResponse(BaseModel):
    case_id: str
    file: str
    question: str
    answer: str
    model_status: list[ModelStatus]
    document: dict[str, Any]
    retrieved: list[RetrievedChunk]
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    case_id: str
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    ok: bool
    models: list[ModelStatus]
