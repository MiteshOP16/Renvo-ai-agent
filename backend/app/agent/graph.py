"""
The core decision graph.

    build_context --> agent_decide --(tool_calls?)--> execute_tool --+
                            ^                                        |
                            |                                        |
                            +----------------------------------------+
                            |
                          (no tool_calls)
                            |
                            v
                           END

- build_context: pulls the CURRENT dataframe's metadata (never raw rows) and
  the running conversation summary into state. Runs at the start of a turn
  and again after every tool call, so the model always reasons/summarizes
  against fresh numbers.
- agent_decide: LLM (Groq, tool-bound to both mutating and analysis tools)
  reads the domain system prompt + metadata + summary + recent chat history
  and either calls a tool or replies in plain language. Wrapped in retry
  logic for transient failures.
- execute_tool: validates the proposed call against the real function
  signature (catches hallucinated/malformed calls before they touch data),
  blocks duplicate calls within the same turn (loop guard), dispatches to
  the mutating registry (creates a new dataset version) or the analysis
  registry (read-only, no version created), and records both a UI-facing
  log line and a structured tool_history entry.

Tool results are prefixed "Success:" / "Result:" / "Error:" / "Skipped:" on
purpose -- the system prompt requires the model to stay grounded in that
literal prefix rather than assuming a change happened. This directly fixes
a bug where the model would claim it dropped a column that a tool call had
actually failed to find.

A tool_call_count guard prevents runaway loops.
"""

import json

import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import get_llm_with_tools
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.core.config import settings
from app.core.metadata_extractor import extract_metadata
from app.core.reliability import ToolValidationError, call_with_retry, validate_tool_call
from app.core.session_manager import session_manager
from app.tools.analysis_implementations import ANALYSIS_TOOL_EXECUTORS
from app.tools.column_resolver import ColumnNotFoundError
from app.tools.implementations import TOOL_EXECUTORS

# Combined only for argument-signature validation -- execution still
# dispatches to the correct registry based on which one the name is in.
_ALL_EXECUTORS_FOR_VALIDATION = {**TOOL_EXECUTORS, **ANALYSIS_TOOL_EXECUTORS}


def build_context_node(state: AgentState) -> dict:
    df = session_manager.get_current_df(state["session_id"])
    metadata = extract_metadata(df)
    summary = session_manager.get_summary(state["session_id"])
    return {"metadata": metadata, "summary": summary}


def agent_decide_node(state: AgentState) -> dict:
    llm = get_llm_with_tools()
    metadata_json = json.dumps(state["metadata"], indent=2)
    system_msg = SystemMessage(content=build_system_prompt(metadata_json, state.get("summary", "")))

    messages = [system_msg] + state["messages"]

    try:
        response = call_with_retry(llm.invoke, messages)
    except Exception as e:
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

        # --- loop / duplicate-step guard ---
        if signature in already_called:
            result_text = (
                f"Skipped: '{name}' with the same arguments was already applied this turn "
                "-- avoiding a duplicate change."
            )
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
            continue

        is_analysis = name in ANALYSIS_TOOL_EXECUTORS

        try:
            # --- validate before executing: catches unknown/hallucinated
            # tools, missing required args, and unexpected args up front ---
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
            # Both are "the request couldn't be carried out as specified" --
            # friendly, already contains enough detail (e.g. real column
            # list + suggestions) for the model to self-correct or ask.
            result_text = f"Error: {e}"
            session_manager.log_public(session_id, "ERROR", result_text)
            session_manager.record_tool_call(session_id, name, args, str(e), success=False)
        except Exception as e:  # other real execution errors
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
    graph.add_conditional_edges(
        "agent_decide", route_after_agent, {"execute_tool": "execute_tool", END: END}
    )
    graph.add_edge("execute_tool", "build_context")

    return graph.compile()


agent_graph = build_graph()