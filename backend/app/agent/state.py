import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    # messages accumulate across nodes within a single turn (reducer = append)
    messages: Annotated[list[BaseMessage], operator.add]
    metadata: dict
    summary: str
    tool_call_count: int
    # signatures of tool calls already executed THIS turn (name + args hash),
    # used to detect and block duplicate/looping tool calls
    executed_calls: Annotated[list[str], operator.add]