"""
Column name resolution + friendly, self-correcting error messages.

WHY THIS EXISTS:
In testing, the agent would sometimes attempt to act on a column name that
didn't actually match the dataset (wrong case, stray whitespace, or a
genuinely wrong guess) -- the tool call would correctly fail, but the
model's follow-up reply still claimed success. That's two separate bugs
working together:

  1. Harmless mismatches (case/whitespace) were failing when they didn't
     need to.
  2. Genuine mismatches produced an error message that didn't give the
     model enough to work with, and nothing forced the model to stay
     honest about what the tool result actually said.

This module fixes (1) directly: every tool that takes a `column`/`columns`
argument now resolves it through here instead of doing a raw membership
check, so cosmetic differences (case, leading/trailing spaces) resolve
automatically instead of erroring. Genuine mismatches still raise -- but
with the full column list and close-match suggestions attached, so the
model can self-correct instead of guessing blindly or claiming success.
Fix (2) -- forcing the model to stay grounded in the actual tool result --
is handled in agent/prompts.py.
"""

import difflib

import pandas as pd


class ColumnNotFoundError(ValueError):
    """Raised when a column genuinely doesn't exist, even after
    case/whitespace-insensitive matching. Carries a message that already
    includes the real column list + suggestions, so callers can surface it
    to the model/user as-is."""


def resolve_column(df: pd.DataFrame, name: str) -> str:
    """Return the real column name in `df` that `name` refers to.

    Resolution order:
      1. Exact match -- returned as-is.
      2. Case-insensitive / whitespace-trimmed match -- safe to auto-resolve,
         since it's unambiguously the same logical column.
      3. Otherwise: raise ColumnNotFoundError with the full column list and
         up to 3 close-match suggestions (via difflib), so the caller can
         self-correct instead of guessing again.
    """
    if name in df.columns:
        return name

    normalized = {str(c).strip().lower(): c for c in df.columns}
    key = str(name).strip().lower()
    if key in normalized:
        return normalized[key]

    available = [str(c) for c in df.columns]
    suggestions = difflib.get_close_matches(str(name), available, n=3, cutoff=0.5)
    suggestion_text = f" Close matches: {suggestions}." if suggestions else ""
    raise ColumnNotFoundError(
        f"Column '{name}' does not exist in this dataset.{suggestion_text} "
        f"Available columns: {available}."
    )


def resolve_columns(df: pd.DataFrame, names: list[str]) -> list[str]:
    """Resolve a list of column names in order; raises on the first
    unresolvable one, with the same friendly message."""
    return [resolve_column(df, n) for n in names]