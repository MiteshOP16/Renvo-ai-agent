# DataCleanAI — Conversational Data Cleaning Agent (MVP)

A tool-calling agent that lets non-coder data analysts clean a dataset by chatting
in plain English. Built with **FastAPI + LangGraph + LangChain + Groq**.

This is phase 1: **one tool call per turn, human-in-the-loop.** Phase 2 (fully
autonomous multi-step cleaning) is a straightforward extension of this same graph
— see "Scaling to phase 2" below.

## Architecture

```
frontend/index.html        Simple chat UI (upload, chat, undo/redo, live log panel)

backend/app/
├── main.py                FastAPI app entrypoint
├── api/routes.py          /dataset/upload, /chat, /dataset/undo, /dataset/redo,
│                          /dataset/logs, /dataset/download
├── core/
│   ├── session_manager.py In-memory dataset "memory": versioned DataFrame stack
│   │                      per session -> undo/redo is just moving a pointer
│   ├── metadata_extractor.py  DataFrame -> small JSON summary. This is the ONLY
│   │                      thing the LLM ever sees (never raw rows/cells)
│   └── config.py          Settings (Groq key/model, history window, loop cap)
├── tools/
│   ├── definitions.py      LangChain @tool schemas -> bound to the LLM via
│   │                      `llm.bind_tools([...])` (tool_bind). The LLM only
│   │                      ever proposes {name, args} — it never touches data.
│   └── implementations.py  The actual pandas logic for each tool. Only these
│                      5 fixed, reviewed functions can mutate the dataset.
└── agent/
    ├── state.py            LangGraph state schema
    ├── prompts.py          Domain-specific system prompt (rules, tone, guardrails)
    ├── llm.py              ChatGroq client, tool-bound
    └── graph.py            The decision graph (see below)
```

### The graph

```
build_context --> agent_decide --(tool_calls?)--> execute_tool --+
      ^                  |                                        |
      |                  | (no tool_calls)                        |
      |                  v                                        |
      +------------------ END <--------------------------------- (loops back)
```

- **build_context** — pulls fresh metadata for the session's *current* DataFrame
  version. Runs at the start of a turn and again after every tool call, so the
  LLM always reasons/summarizes against up-to-date numbers.
- **agent_decide** — Groq LLM (tool-bound) reads the domain system prompt +
  metadata JSON + a trimmed window of chat history (context optimisation — no
  raw data, no full history, just the last `MAX_CHAT_HISTORY` turns) and either
  calls a tool or replies directly.
- **execute_tool** — maps the LLM's chosen tool name to the real pandas function,
  runs it against the session's current DataFrame, and commits a **new version**
  to session memory (never mutates in place). A `tool_call_count` guard (default
  3) plus the system prompt's "one tool per turn" instruction prevent runaway loops.

### Why metadata-only context?

The LLM never receives raw dataset rows — only shape, dtypes, null %, duplicate
counts, and light stats (min/max/mean for numbers, top-5 categories for text).
This keeps prompts small and cheap, avoids leaking potentially sensitive raw
data to a third-party API, and forces the model to reason about *data quality*
rather than pattern-match on specific values.

### Why version-stack memory instead of in-place mutation?

Every tool call appends a new DataFrame version instead of editing the current
one. Undo/redo is then just moving a pointer:

```
versions:        [v0, v1, v2, v3]
current_index:               ^
undo() -> index -= 1
redo() -> index += 1
```

Applying a new tool call after an undo discards the redo-able future, exactly
like a normal editor's undo stack.

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                   # then paste your Groq key in .env
uvicorn app.main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it points at
`http://localhost:8000/api`). No build step needed.

Get a free Groq API key at https://console.groq.com — any tool-calling-capable
Groq model works (`llama-3.3-70b-versatile` is a good default; `mixtral` and
`llama-3.1-8b-instant` also support tool calling if you want cheaper/faster).

## The 5 tools (phase 1)

| Tool | What it does |
|---|---|
| `drop_duplicates` | Remove duplicate rows, optional subset of columns, `keep` first/last/none |
| `handle_missing_values` | Impute a column via mean / median / mode / constant, or drop rows with nulls |
| `drop_column` | Remove one or more columns |
| `rename_column` | Rename a column |
| `convert_dtype` | Cast a column to int / float / str / datetime / category / bool |

Try prompts like:
- "drop duplicate rows"
- "fill missing values in age with the median"
- "the salary column should be a number, not text"
- "get rid of the customer_id column"

If a request is ambiguous (wrong column name, unclear strategy) the agent will
ask a clarifying question instead of guessing — this is enforced in the system
prompt in `agent/prompts.py`.

## Scaling to phase 2 (autonomous multi-step cleaning)

The graph is already shaped for this — you mostly need to relax constraints,
not rebuild anything:

1. **Raise `MAX_TOOL_CALLS_PER_TURN`** and remove the "one tool per turn" line
   from the system prompt, so the agent can chain several tool calls before
   handing control back to the user.
2. **Add a planning node** before `agent_decide` that asks the LLM to propose
   a short ordered plan (e.g. "1. drop duplicates, 2. impute nulls in age,
   3. fix dtype of price") from the metadata alone, then walks that plan.
3. **Add a confirmation node** for an autonomous mode: after each tool call,
   surface a one-line diff-style log entry (already captured in
   `session_manager.logs`) and let the user approve/pause instead of blindly
   continuing — a good middle ground between full autonomy and full manual.
4. **Swap in-memory session storage for Redis or Postgres** (store DataFrames
   as parquet blobs keyed by `session_id:version_index`) once you need more
   than one server process.
5. **Add more tools** — the pattern (schema in `definitions.py`, real logic in
   `implementations.py`, one entry in `TOOL_EXECUTORS`) scales to as many as
   you need without touching the graph itself.

## Notes / limitations of this MVP

- Sessions are stored **in-process memory** — restarting the server loses all
  sessions. Fine for local development/demo; swap for Redis/DB before deploying.
- Only CSV/XLSX upload is wired up.
- No auth — add an API key or session-cookie layer before exposing this
  publicly.
