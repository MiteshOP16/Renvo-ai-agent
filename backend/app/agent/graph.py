"""
The core decision graph.

    build_context --> agent_decide --(tool_calls?)--> execute_tool --+
                            ^                                        |
                            |                                        |
                            +----------------------------------------+
                            |
                          (no tool_calls, incl. clarifying questions)
                            |
                            v
                           END

- build_context: pulls the CURRENT dataframe's metadata (never raw rows) and
  the running conversation summary into state.
- agent_decide: routes this turn's tools via semantic retrieval + rerank
  (see tool_router.py). The system prompt is rebuilt EVERY time with the
  exact tool names actually being bound this call (see prompts.py) -- this
  is deliberate: a prompt that names a tool NOT in the current request's
  bound schema causes Groq to reject the whole call with "tool call
  validation failed: ... not in request.tools" once the model tries to use
  it. This must stay true across every attempt, including the narrower
  fallback retry below, which binds a different (smaller) tool set and so
  needs its own rebuilt prompt, not the original one reused.
- execute_tool: validates the proposed call against the real function
  signature, blocks duplicate calls within the same turn, dispatches to the
  mutating or analysis registry, and records both a UI log line and a
  structured tool_history entry.
"""

import json

import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import get_llm_with_tools, get_plain_llm
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.agent.tool_router import accumulate_turn_text, route_tools
from app.core.config import settings
from app.core.metadata_extractor import extract_metadata
from app.core.reliability import ToolValidationError, call_with_retry, validate_tool_call
from app.core.session_manager import session_manager
from app.tools.analysis_implementations import ANALYSIS_TOOL_EXECUTORS
from app.tools.column_resolver import ColumnNotFoundError
from app.tools.definitions import ALL_TOOLS
from app.tools.implementations import TOOL_EXECUTORS

_ALL_EXECUTORS_FOR_VALIDATION = {**TOOL_EXECUTORS, **ANALYSIS_TOOL_EXECUTORS}
_TOOL_SCHEMA_ERROR_MARKERS = ("tool call validation failed", "not in request.tools", "tool_use_failed")


def _looks_like_tool_schema_error(e: Exception) -> bool:
    text = str(e).lower()
    return any(marker in text for marker in _TOOL_SCHEMA_ERROR_MARKERS)


def build_context_node(state: AgentState) -> dict:
    df = session_manager.get_current_df(state["session_id"])
    metadata = extract_metadata(df)
    summary = session_manager.get_summary(state["session_id"])
    return {"metadata": metadata, "summary": summary}


def agent_decide_node(state: AgentState) -> dict:
    metadata_json = json.dumps(state["metadata"], indent=2)
    summary = state.get("summary", "")

    context_text = accumulate_turn_text(state["messages"])
    try:
        routing = route_tools(context_text)
    except Exception as e:
        session_manager.log_public(
            state["session_id"], "ERROR", f"Tool retrieval unavailable, falling back to full tool set: {e}"
        )
        routing = {"tools": ALL_TOOLS, "confidence": 1.0, "needs_clarification": False}

    # --- too vague to confidently route: ask, don't guess ---
    if routing["needs_clarification"]:
        session_manager.log_public(
            state["session_id"], "SYSTEM",
            f"Routing confidence too low ({routing['confidence']:.2f}) -- asking for clarification instead of guessing a tool.",
        )
        clarify_system = SystemMessage(
            content=build_system_prompt(metadata_json, summary, tool_names=[])
            + "\n\nThe user's last message doesn't clearly match a specific supported action "
              "(nothing in the tool catalog matched with confidence). Do NOT guess which action "
              "they want and do NOT call a tool. Ask one short, specific clarifying question -- "
              "e.g. which column is involved, or what kind of change they want -- referencing the "
              "actual column names from the metadata above where relevant."
        )
        try:
            response = call_with_retry(get_plain_llm().invoke, [clarify_system] + state["messages"])
        except Exception as e:
            session_manager.log_public(state["session_id"], "ERROR", f"Clarification call failed: {e}")
            response = AIMessage(content="Could you tell me a bit more about what you'd like me to do, and which column it involves?")
        return {"messages": [response]}

    routed_tools = routing["tools"]
    routed_names = [t.name for t in routed_tools]
    system_msg = SystemMessage(content=build_system_prompt(metadata_json, summary, tool_names=routed_names))
    messages = [system_msg] + state["messages"]

    try:
        llm = get_llm_with_tools(tools=routed_tools)
        response = call_with_retry(llm.invoke, messages)
    except Exception as e:
        if _looks_like_tool_schema_error(e):
            session_manager.log_public(
                state["session_id"], "ERROR", f"Tool schema error, retrying with a narrower tool set: {e}"
            )
            try:
                narrow_tools = routed_tools[:3]
                narrow_names = [t.name for t in narrow_tools]
                # MUST rebuild the system prompt here too -- it has to name
                # exactly narrow_tools, not the original (larger) routed set,
                # or this retry hits the exact same "not in request.tools"
                # failure the fallback is supposed to fix.
                narrow_system = SystemMessage(content=build_system_prompt(metadata_json, summary, tool_names=narrow_names))
                narrow_messages = [narrow_system] + state["messages"]
                narrow_llm = get_llm_with_tools(tools=narrow_tools)
                response = call_with_retry(narrow_llm.invoke, narrow_messages, max_attempts=1)
            except Exception as e2:
                session_manager.log_public(state["session_id"], "ERROR", f"Narrow retry also failed: {e2}")
                response = AIMessage(
                    content=(
                        "Sorry, I had trouble matching that request to one of my actions. "
                        "Could you rephrase it, or tell me exactly which column and action you mean? "
                        "Your dataset hasn't been changed."
                    )
                )
        else:
            session_manager.log_public(state["session_id"], "ERROR", f"LLM call failed after retries: {e}")
            response = AIMessage(
                content=(
                    "Sorry, I'm having trouble reaching the model right now. "
                    "Your dataset hasn't been changed -- please try again in a moment."
                )
            )
    return {"messages": [response]}


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("tool_call_count", 0) < settings.MAX_TOOL_CALLS_PER_TURN:
        return "execute_tool"
    return END


def _call_signature(name: str, args: dict) -> str:
    return f"{name}::{json.dumps(args, sort_keys=True, default=str)}"


def execute_tool_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    last = state["messages"][-1]
    df = session_manager.get_current_df(session_id)

    already_called = set(state.get("executed_calls", []))
    new_signatures = []
    tool_messages = []

    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        call_id = call["id"]
        signature = _call_signature(name, args)

        if signature in already_called:
            result_text = f"Skipped: '{name}' with the same arguments was already applied this turn -- avoiding a duplicate change."
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
            continue

        is_analysis = name in ANALYSIS_TOOL_EXECUTORS

        try:
            validate_tool_call(name, args, _ALL_EXECUTORS_FOR_VALIDATION)

            if is_analysis:
                executor = ANALYSIS_TOOL_EXECUTORS[name]
                report = executor(df, **args)
                if not isinstance(report, str) or not report.strip():
                    raise ToolValidationError(f"'{name}' didn't return a usable result.")
                session_manager.record_tool_call(session_id, name, args, report, success=True)
                session_manager.log_public(session_id, "ANALYSIS", f"{name}: {report.splitlines()[0]}")
                result_text = f"Result: {report}"
            else:
                executor = TOOL_EXECUTORS[name]
                new_df, description = executor(df, **args)

                if not isinstance(new_df, pd.DataFrame):
                    raise ToolValidationError(f"'{name}' returned something that wasn't a dataset.")
                if not description:
                    raise ToolValidationError(f"'{name}' didn't report what it changed.")

                session_manager.apply_new_version(session_id, new_df, description)
                session_manager.record_tool_call(session_id, name, args, description, success=True)
                df = new_df
                result_text = f"Success: {description}"

        except (ToolValidationError, ColumnNotFoundError) as e:
            result_text = f"Error: {e}"
            session_manager.log_public(session_id, "ERROR", result_text)
            session_manager.record_tool_call(session_id, name, args, str(e), success=False)
        except Exception as e:
            result_text = f"Error running {name}: {e}"
            session_manager.log_public(session_id, "ERROR", result_text)
            session_manager.record_tool_call(session_id, name, args, str(e), success=False)

        new_signatures.append(signature)
        tool_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))

    return {
        "messages": tool_messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "executed_calls": new_signatures,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("build_context", build_context_node)
    graph.add_node("agent_decide", agent_decide_node)
    graph.add_node("execute_tool", execute_tool_node)

    graph.set_entry_point("build_context")
    graph.add_edge("build_context", "agent_decide")
    graph.add_conditional_edges("agent_decide", route_after_agent, {"execute_tool": "execute_tool", END: END})
    graph.add_edge("execute_tool", "build_context")

    return graph.compile()


agent_graph = build_graph()