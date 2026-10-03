# DataCleanAI Project Guide

## What the project does

DataCleanAI is a FastAPI service with a small browser client. A user uploads a CSV or Excel file, asks for a cleaning or analysis operation in plain English, and receives an updated metadata view and activity log. The LLM never receives raw rows. It receives dataset metadata, compact conversation context, and a small set of tool schemas.

## Request workflow

```text
Browser
  -> POST /api/dataset/upload
  -> POST /api/chat
       -> store user message and optionally summarize old history
       -> build_context: current DataFrame -> metadata
       -> agent_decide: retrieve -> rerank -> bind relevant tools -> call Groq
       -> execute_tool: validate and run the selected pandas executor
       -> build_context again if a tool was called
  -> return reply, metadata, logs, undo/redo state
```

For a normal chat request:

1. `routes.py` verifies the session, stores the user message, and keeps recent chat turns plus a running summary.
2. `graph.py` extracts fresh metadata and asks `tool_router.py` which tools are relevant.
3. The router embeds the request, searches Qdrant for 15 candidates, and reranks them. The top 7 are normally bound to the LLM.
4. The model either answers directly or proposes a tool call. A vague request binds no tools and produces a clarification question.
5. The graph validates the tool name and arguments, resolves columns, and calls the matching implementation. Mutations create a new DataFrame version; analysis tools only return a report.
6. The graph refreshes metadata after a tool call and can repeat until the configured per-turn action limit is reached. The final response is stored in chat history.

## Token usage: routed tools vs. binding all tools

This project is designed to use **fewer LLM input tokens** than binding every tool on every request.

| Situation | Tool schemas sent to Groq | Token effect |
|---|---:|---|
| Normal successful routing | 7 by default | Lower prompt size |
| Ambiguous request | 0 | Lowest tool-schema cost; asks for clarification |
| Retrieval/reranker failure | 25 | Same schema cost as binding all tools |
| `get_llm_with_tools()` called without an argument | 25 | Full-tool fallback/default |

The exact token count depends on the provider's serialization of each JSON schema and on the prompt. The savings are approximately proportional to the schema size removed: normal calls send 7 of 25 schemas, or about 28% of the tool-schema catalog. This does not mean the whole request costs only 28%, because metadata, system instructions, chat history, tool results, and model output still consume tokens.

There is a tradeoff. Every routed request performs local embedding work and a cross-encoder rerank. These use CPU/RAM and add latency, but they do not count as Groq prompt tokens. The router also embeds the request even when the final answer will be conceptual. Startup loads the embedding and reranker models and builds the Qdrant index once per process.

Token usage is also controlled by two other mechanisms:

- `metadata_extractor.py` sends aggregates such as shape, dtypes, null rates, and small statistics instead of raw cells.
- `context_manager.py` summarizes old turns after 12 stored turns and keeps only the configured recent window (`MAX_CHAT_HISTORY=6`).

To measure real cost, inspect token usage fields in the provider response for the model being used. The repository currently does not collect a per-request token dashboard, so this guide describes expected behavior rather than measured totals.

## File-by-file map

### Root and frontend

- `README.md`: quick-start instructions, architecture summary, examples, and limitations.
- `frontend/index.html`: no-build browser UI. Uploads files, sends chat messages, renders metadata and logs, and calls undo, redo, and download endpoints.

### Backend entry points

- `backend/app/main.py`: creates the FastAPI app, enables CORS, registers `/api`, and performs startup health checks for models and Qdrant.
- `backend/app/api/routes.py`: HTTP endpoints for upload, chat, undo, redo, logs, and CSV download.
- `backend/app/models/schemas.py`: Pydantic request and response models used by the API.

### Agent layer

- `backend/app/agent/state.py`: LangGraph state passed between nodes, including messages, metadata, summary, action count, and duplicate-call signatures.
- `backend/app/agent/graph.py`: decision graph and execution boundary. Builds context, calls the LLM, validates tool calls, dispatches executors, records results, and enforces the action limit.
- `backend/app/agent/llm.py`: creates ChatGroq clients with a supplied routed tool list, or with no tools for clarification and summarization calls.
- `backend/app/agent/tool_router.py`: retrieve-then-rerank routing, confidence thresholding, and Qdrant index construction.
- `backend/app/agent/embeddings.py`: lazy, cached BGE-M3 embedding model.
- `backend/app/agent/reranker.py`: lazy, cached BGE cross-encoder used to score retrieved candidates.
- `backend/app/agent/vector_store.py`: Qdrant collection setup, indexing, and similarity search for tool descriptions.
- `backend/app/agent/prompts.py`: builds the system prompt using only the tool names actually bound for the current request.

### Core services

- `backend/app/core/config.py`: environment-backed settings for Groq, Qdrant, context windows, routing sizes, retries, and action limits.
- `backend/app/core/metadata_extractor.py`: converts a DataFrame to the safe metadata object sent to the LLM.
- `backend/app/core/context_manager.py`: folds old chat turns into a compact running summary.
- `backend/app/core/session_manager.py`: in-memory sessions, DataFrame version stack, chat history, summaries, tool history, and public logs.
- `backend/app/core/reliability.py`: tool-call validation and retry behavior for LLM calls.

### Tools

- `backend/app/tools/definitions.py`: LangChain tool schemas and descriptions. These declarations tell the LLM names and argument shapes; their placeholder bodies do not modify data.
- `backend/app/tools/implementations.py`: pandas implementations for the 20 mutating tools and the `TOOL_EXECUTORS` dispatch map.
- `backend/app/tools/analysis_implementations.py`: five read-only analysis implementations and the `ANALYSIS_TOOL_EXECUTORS` dispatch map.
- `backend/app/tools/column_resolver.py`: validates and resolves model column references against the current DataFrame.

### Supporting files and data

- `backend/requirements.txt`: Python dependencies.
- `backend/scripts/build_tool_index.py`: manually rebuilds the Qdrant tool description index after tool descriptions change.
- `backend/qdrant_data/`: local embedded Qdrant data. It is generated/runtime state, not application logic.

## Important defaults

The defaults in `backend/app/core/config.py` are:

- `RETRIEVAL_TOP_K=15`: candidates searched before reranking.
- `RERANK_TOP_K=7`: schemas bound to the normal LLM call.
- `MAX_CHAT_HISTORY=6`: recent stored turns included in the request.
- `SUMMARY_TRIGGER_TURNS=12`, `SUMMARY_KEEP_RECENT=6`: history compression.
- `MAX_TOOL_CALLS_PER_TURN=5`: graph loop guard.
- `CLARIFICATION_CONFIDENCE_THRESHOLD=0.35`: below this, ask instead of guess.

## Limitations

Sessions and DataFrames live only in process memory, so a restart loses them. The frontend is a static HTML page, CORS is open for local development, and the project currently supports CSV/XLS/XLSX upload and CSV download. Production use would need persistent storage, authentication, controlled CORS, and token usage/latency monitoring.
