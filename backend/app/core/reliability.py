"""
Reliability helpers used by the agent graph:

- call_with_retry(): retries transient failures (timeouts, rate limits,
  connection errors) with exponential backoff. Non-transient errors
  (bad input, programming errors) are re-raised immediately -- retrying
  those would just waste time and tokens.

- validate_tool_call(): checks a tool call the LLM proposed against the
  REAL function signature before it ever touches the dataset. Catches
  unknown tool names, missing required arguments, and unexpected
  arguments (all symptoms of a hallucinated or malformed tool call) and
  turns them into a friendly, user-facing message instead of a stack trace.
"""

import inspect
import time
from typing import Callable

from app.core.config import settings


class ToolValidationError(Exception):
    """A malformed/hallucinated tool call. Always non-retryable and always
    safe to surface to the user as plain text."""


class RetryableError(Exception):
    """Wrap a transient failure explicitly if a caller already knows it's
    worth retrying (e.g. a timeout it caught itself)."""


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "429",
    "502",
    "503",
    "504",
    "connection",
    "temporarily unavailable",
)


def _looks_transient(e: Exception) -> bool:
    if isinstance(e, RetryableError):
        return True
    text = str(e).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def call_with_retry(fn: Callable, *args, max_attempts: int | None = None, **kwargs):
    """Call fn(*args, **kwargs), retrying on transient-looking failures with
    exponential backoff. Re-raises the final error if every attempt fails,
    or immediately if the error doesn't look transient."""
    max_attempts = max_attempts or settings.MAX_LLM_RETRIES
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - intentionally broad, classified below
            last_err = e
            if not _looks_transient(e) or attempt == max_attempts:
                raise
            time.sleep(settings.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    raise last_err  # pragma: no cover - unreachable, satisfies type checkers


def validate_tool_call(name: str, args: dict, executors: dict) -> None:
    """Raise ToolValidationError with a friendly message if this tool call
    can't actually be executed as proposed. Does not run the tool."""
    if name not in executors:
        raise ToolValidationError(
            f"I tried to use an action called '{name}', but it isn't available in this "
            "system. Let me reconsider and try a supported action instead."
        )

    fn = executors[name]
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[1:]  # first param is always `df`

    required = [p.name for p in params if p.default is inspect.Parameter.empty]
    missing = [p for p in required if p not in (args or {})]
    if missing:
        raise ToolValidationError(
            f"That action needs {', '.join(missing)} to run, but it wasn't provided. "
            "Could you clarify?"
        )

    allowed = {p.name for p in params}
    unexpected = set((args or {}).keys()) - allowed
    if unexpected:
        raise ToolValidationError(
            f"That action received parameters it doesn't recognize: {', '.join(sorted(unexpected))}."
        )