import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph
from app.core.config import settings
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
        "tool_call_count": 0,
    }

    try:
        result = agent_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")

    final_reply = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_reply = msg.content
            break
    if not final_reply:
        final_reply = "Done."

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
