from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    metadata: dict[str, Any]
    logs: list[dict[str, Any]]
    can_undo: bool
    can_redo: bool


class UploadResponse(BaseModel):
    session_id: str
    metadata: dict[str, Any]


class UndoRedoResponse(BaseModel):
    success: bool
    metadata: dict[str, Any]
    can_undo: bool
    can_redo: bool
