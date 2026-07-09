"""
In-memory session store.

For each session (= one uploaded dataset) we keep a *stack of DataFrame
versions*. Every successful tool call appends a new version instead of
mutating in place, so undo/redo is just moving a pointer:

    versions:        [v0, v1, v2, v3]
    current_index:              ^

    undo()  -> current_index -= 1
    redo()  -> current_index += 1

If the user undoes and then applies a new tool call, the "future" versions
are discarded (standard undo/redo stack behaviour).

NOTE: This is a simple in-process dict. It is fine for a single-server MVP.
For production/multi-worker deployment, swap this for Redis or a DB-backed
store keyed by session_id, and persist DataFrames as parquet blobs.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    # ---- lifecycle -------------------------------------------------
    def create_session(self, df: pd.DataFrame, filename: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "filename": filename,
            "versions": [df.copy()],
            "current_index": 0,
            "logs": [],
            "chat_history": [],
        }
        self._log(session_id, "SYSTEM", f"Dataset '{filename}' uploaded. Shape: {df.shape}")
        return session_id

    def get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError("Session not found")
        return self._sessions[session_id]

    def get_current_df(self, session_id: str) -> pd.DataFrame:
        s = self.get_session(session_id)
        return s["versions"][s["current_index"]]

    # ---- versioning / undo-redo -------------------------------------
    def apply_new_version(self, session_id: str, new_df: pd.DataFrame, description: str):
        s = self.get_session(session_id)
        # discard any redo-able future if we branch off after an undo
        s["versions"] = s["versions"][: s["current_index"] + 1]
        s["versions"].append(new_df)
        s["current_index"] += 1
        self._log(session_id, "TOOL", description)

    def undo(self, session_id: str) -> bool:
        s = self.get_session(session_id)
        if s["current_index"] > 0:
            s["current_index"] -= 1
            self._log(session_id, "SYSTEM", "Undo performed")
            return True
        return False

    def redo(self, session_id: str) -> bool:
        s = self.get_session(session_id)
        if s["current_index"] < len(s["versions"]) - 1:
            s["current_index"] += 1
            self._log(session_id, "SYSTEM", "Redo performed")
            return True
        return False

    def can_undo(self, session_id: str) -> bool:
        s = self.get_session(session_id)
        return s["current_index"] > 0

    def can_redo(self, session_id: str) -> bool:
        s = self.get_session(session_id)
        return s["current_index"] < len(s["versions"]) - 1

    # ---- chat memory --------------------------------------------------
    def add_chat(self, session_id: str, role: str, content: str):
        s = self.get_session(session_id)
        s["chat_history"].append({"role": role, "content": content})

    def get_chat_history(self, session_id: str, limit: int = 8) -> list[dict]:
        s = self.get_session(session_id)
        return s["chat_history"][-limit:]

    # ---- logs (for the UX log panel) -----------------------------------
    def _log(self, session_id: str, level: str, message: str):
        s = self._sessions[session_id]
        s["logs"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message,
            }
        )

    def log_public(self, session_id: str, level: str, message: str):
        """Allow the agent graph to push arbitrary log lines (e.g. errors)."""
        self._log(session_id, level, message)

    def get_logs(self, session_id: str) -> list[dict]:
        return self.get_session(session_id)["logs"]


# Single shared instance used across the app (simple process-local memory)
session_manager = SessionManager()
