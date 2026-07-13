"""
Hierarchical conversation summarization.

Without this, every turn resends the FULL chat history to the LLM, so
token usage (and cost/latency) grows linearly forever. Instead:

  - Only the last `SUMMARY_KEEP_RECENT` raw turns are ever sent verbatim.
  - Once stored history passes `SUMMARY_TRIGGER_TURNS`, the oldest turns
    are folded into a compact running summary (via a cheap, tool-free LLM
    call) and then pruned from storage entirely.
  - The summary and the recent window are concatenated fresh each turn in
    the system prompt (see agent/prompts.py), so the model always has
    continuity without ever re-reading the whole transcript.

This keeps memory pools separate on purpose:
  - conversation_summary  -> compressed facts/decisions from old turns
  - chat_history (recent)  -> raw last few turns, for natural flow/tone
  - tool_history           -> structured record of every tool call & result,
                              independent of the human-readable summary
"""

from langchain_core.messages import HumanMessage

SUMMARIZE_PROMPT = """Summarize the conversation turns below into compact bullet points \
worth remembering for future turns of this data-cleaning session: the user's goals, \
decisions made, notable tool actions and their outcomes, and any stated preferences \
(e.g. "always cap outliers instead of removing them"). Be terse -- this is for a machine \
to reuse as context, not for a human to read as a narrative. Merge with the existing \
summary rather than repeating it verbatim.

EXISTING SUMMARY:
{existing_summary}

NEW TURNS TO FOLD IN:
{transcript}

Updated summary (bullet points only, no preamble):"""


def summarize_old_turns(llm, old_turns: list[dict], existing_summary: str) -> str:
    """Collapse old turns into an updated running summary. Falls back to a
    terse note (never a crash) if the summarizer call itself fails, so a
    flaky summarization step never breaks the chat."""
    if not old_turns:
        return existing_summary

    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in old_turns)
    prompt = SUMMARIZE_PROMPT.format(
        existing_summary=existing_summary.strip() or "(none yet)",
        transcript=transcript,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = (response.content or "").strip()
        return summary or existing_summary
    except Exception:
        fallback_note = f"- ({len(old_turns)} earlier turn(s) occurred but could not be summarized)"
        return f"{existing_summary}\n{fallback_note}".strip()