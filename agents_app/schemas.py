"""Request/response schemas (pydantic v2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)


class SourceRef(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    outcome: str
    verified: bool
    sources: list[SourceRef] = Field(default_factory=list)
    needs_confirmation: bool = False
    debug: Optional[dict] = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)
    rating: int
    comment: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("rating")
    @classmethod
    def _rating_is_thumbs(cls, value: int) -> int:
        if value not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return value
