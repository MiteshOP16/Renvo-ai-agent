from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class RowDiff(BaseModel):
    """Before/after snapshot for a single changed row."""
    row_index: int
    column: str
    before: Optional[Any] = None
    after: Optional[Any] = None


class ToolAction(BaseModel):
    tool_name: str
    args: dict[str, Any]
    result: str
    success: bool
    is_analysis: bool
    rows_affected: Optional[int] = None
    before_shape: Optional[tuple] = None
    after_shape: Optional[tuple] = None
    changed_columns: list[str] = []
    row_diffs: list[RowDiff] = []


class ChatResponse(BaseModel):
    reply: str
    metadata: dict[str, Any]
    before_metadata: Optional[dict[str, Any]] = None
    logs: list[dict[str, Any]]
    can_undo: bool
    can_redo: bool
    tool_actions: list[ToolAction] = []
    version_index: int = 0
    total_versions: int = 1


class UploadResponse(BaseModel):
    session_id: str
    metadata: dict[str, Any]


class UndoRedoResponse(BaseModel):
    success: bool
    metadata: dict[str, Any]
    can_undo: bool
    can_redo: bool
