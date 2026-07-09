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

- build_context: pulls the CURRENT dataframe's metadata (never raw rows) into
  state. Runs both at the start of a turn and again after every tool call, so
  the LLM always summarizes against fresh numbers.
- agent_decide: LLM (Groq, tool-bound) reads the domain system prompt +
  metadata + trimmed chat history and either calls a tool or replies in
  plain language.
- execute_tool: looks up the real pandas implementation for the tool the LLM
  picked, runs it against the session's current DataFrame, and commits a new
  version to session memory (this is what undo/redo walks back through).

A tool_call_count guard prevents runaway loops; the system prompt also asks
the model to use at most one tool per turn for now.
"""

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import get_llm_with_tools
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.core.config import settings
from app.core.metadata_extractor import extract_metadata
from app.core.session_manager import session_manager
from app.tools.implementations import TOOL_EXECUTORS


def build_context_node(state: AgentState) -> dict:
    df = session_manager.get_current_df(state["session_id"])
    return {"metadata": extract_metadata(df)}


def agent_decide_node(state: AgentState) -> dict:
    llm = get_llm_with_tools()
    metadata_json = json.dumps(state["metadata"], indent=2)
    system_msg = SystemMessage(content=build_system_prompt(metadata_json))

    # Context optimisation: system prompt (fresh metadata) + only the
    # messages accumulated so far this turn (chat history is trimmed by the
    # API layer before the graph is invoked).
    messages = [system_msg] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("tool_call_count", 0) < settings.MAX_TOOL_CALLS_PER_TURN:
        return "execute_tool"
    return END


def execute_tool_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    last = state["messages"][-1]
    df = session_manager.get_current_df(session_id)

    tool_messages = []
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        executor = TOOL_EXECUTORS.get(name)

        if executor is None:
            result_text = f"Error: unknown tool '{name}'."
            session_manager.log_public(session_id, "ERROR", result_text)
        else:
            try:
                new_df, description = executor(df, **args)
                session_manager.apply_new_version(session_id, new_df, description)
                df = new_df
                result_text = f"Success: {description}"
            except Exception as e:
                result_text = f"Error running {name} with args {args}: {e}"
                session_manager.log_public(session_id, "ERROR", result_text)

        tool_messages.append(ToolMessage(content=result_text, tool_call_id=call["id"]))

    return {
        "messages": tool_messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
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
    # refresh metadata after every tool call before the LLM summarizes / decides again
    graph.add_edge("execute_tool", "build_context")

    return graph.compile()


agent_graph = build_graph()
