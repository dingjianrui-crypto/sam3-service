from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class VideoCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    mime_type: str = Field(min_length=1, max_length=100)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("filename must not contain a path")
        return value


class PromptCreate(BaseModel):
    text: str = Field(min_length=1, max_length=80)

    @field_validator("text")
    @classmethod
    def normalized(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("prompt must not be blank")
        return text


class JobSettings(BaseModel):
    working_max_dimension: int = Field(default=1280, ge=320, le=1920)
    include_boxes: bool = True
    score_threshold: float = Field(default=0.5, ge=0, le=1)


class JobCreate(BaseModel):
    video_id: str
    prompts: list[PromptCreate] = Field(min_length=1)
    settings: JobSettings = Field(default_factory=JobSettings)


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody

