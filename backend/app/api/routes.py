import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph
from app.agent.llm import get_plain_llm
from app.core.config import settings
from app.core.context_manager import summarize_old_turns
from app.core.metadata_extractor import extract_metadata
from app.core.session_manager import session_manager
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    UndoRedoResponse,
    UploadResponse,
)

router = APIRouter()


@router.post("/dataset/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(400, "Only .csv or .xlsx files are supported.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {e}")

    session_id = session_manager.create_session(df, file.filename)
    metadata = extract_metadata(df)
    return UploadResponse(session_id=session_id, metadata=metadata)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        session_manager.get_session(req.session_id)
    except KeyError:
        raise HTTPException(404, "Session not found. Please upload a dataset first.")

    session_manager.add_chat(req.session_id, "user", req.message)

    # Hierarchical summarization: once stored raw history grows past the
    # trigger, fold the oldest turns into the running summary and prune
    # them from storage. This is what keeps token usage roughly constant
    # across a long conversation instead of growing every turn.
    full_history = session_manager.get_full_chat_history(req.session_id)
    if len(full_history) > settings.SUMMARY_TRIGGER_TURNS:
        keep = settings.SUMMARY_KEEP_RECENT
        old_turns, _recent = full_history[:-keep], full_history[-keep:]
        try:
            new_summary = summarize_old_turns(
                get_plain_llm(), old_turns, session_manager.get_summary(req.session_id)
            )
            session_manager.set_summary(req.session_id, new_summary)
            session_manager.prune_history(req.session_id, keep)
        except Exception as e:
            # Summarization is a nice-to-have for token efficiency, never a
            # reason to break the chat -- just skip pruning this turn.
            session_manager.log_public(req.session_id, "ERROR", f"Summarization skipped: {e}")

    history = session_manager.get_chat_history(req.session_id, limit=settings.MAX_CHAT_HISTORY)

    # Rebuild trimmed LangChain message history (context optimisation: only
    # last N turns, and only role/content -- no metadata duplicated per turn)
    lc_messages = []
    for turn in history[:-1]:
        if turn["role"] == "user":
            lc_messages.append(HumanMessage(content=turn["content"]))
        else:
            lc_messages.append(AIMessage(content=turn["content"]))
    lc_messages.append(HumanMessage(content=req.message))

    initial_state = {
        "session_id": req.session_id,
        "messages": lc_messages,
        "metadata": {},
        "summary": "",
        "tool_call_count": 0,
        "executed_calls": [],
    }

    try:
        result = agent_graph.invoke(initial_state)
        final_reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_reply = msg.content
                break
        if not final_reply:
            if result.get("tool_call_count", 0) >= settings.MAX_TOOL_CALLS_PER_TURN:
                final_reply = (
                    "I've made several changes and reached this turn's action limit. "
                    "Let me know if you'd like me to continue."
                )
            else:
                final_reply = "Done."
    except Exception as e:
        # Graph-level failure (not a single tool failure -- those are already
        # caught inside the graph). Recover without losing conversation
        # state: the user's message and dataset version are untouched.
        session_manager.log_public(req.session_id, "ERROR", f"Agent error: {e}")
        final_reply = (
            "Something went wrong on my end handling that request. Your dataset wasn't "
            "changed -- could you try rephrasing, or try again in a moment?"
        )

    session_manager.add_chat(req.session_id, "assistant", final_reply)

    metadata = extract_metadata(session_manager.get_current_df(req.session_id))
    logs = session_manager.get_logs(req.session_id)[-15:]

    return ChatResponse(
        reply=final_reply,
        metadata=metadata,
        logs=logs,
        can_undo=session_manager.can_undo(req.session_id),
        can_redo=session_manager.can_redo(req.session_id),
    )


@router.post("/dataset/undo", response_model=UndoRedoResponse)
async def undo(session_id: str):
    try:
        success = session_manager.undo(session_id)
        metadata = extract_metadata(session_manager.get_current_df(session_id))
    except KeyError:
        raise HTTPException(404, "Session not found.")
    return UndoRedoResponse(
        success=success,
        metadata=metadata,
        can_undo=session_manager.can_undo(session_id),
        can_redo=session_manager.can_redo(session_id),
    )


@router.post("/dataset/redo", response_model=UndoRedoResponse)
async def redo(session_id: str):
    try:
        success = session_manager.redo(session_id)
        metadata = extract_metadata(session_manager.get_current_df(session_id))
    except KeyError:
        raise HTTPException(404, "Session not found.")
    return UndoRedoResponse(
        success=success,
        metadata=metadata,
        can_undo=session_manager.can_undo(session_id),
        can_redo=session_manager.can_redo(session_id),
    )


@router.get("/dataset/logs/{session_id}")
async def get_logs(session_id: str):
    try:
        return session_manager.get_logs(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.")


@router.get("/dataset/download/{session_id}")
async def download(session_id: str):
    try:
        df = session_manager.get_current_df(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.")
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cleaned_dataset.csv"
    return response