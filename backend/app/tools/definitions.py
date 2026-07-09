"""
LangChain tool *schemas* only.

These are what get passed to `llm.bind_tools([...])` so the LLM knows the
tool names, descriptions and argument shapes. The function bodies are never
actually called by LangChain -- the graph intercepts `AIMessage.tool_calls`
and runs the matching function in `implementations.py` against the real
DataFrame. This keeps the LLM completely decoupled from raw data: it only
ever proposes a `{name, args}` pair.
"""

from typing import List, Optional

from langchain_core.tools import tool


@tool
def drop_duplicates(subset: Optional[List[str]] = None, keep: str = "first") -> str:
    """Remove duplicate rows from the dataset.

    Args:
        subset: Column names to consider when identifying duplicates.
            If omitted, all columns are used.
        keep: Which occurrence to keep: 'first', 'last', or 'none'
            (drop every row that has a duplicate, keeping none of them).
    """
    return "handled_by_executor"


@tool
def handle_missing_values(
    column: str,
    strategy: str = "mean",
    constant_value: Optional[str] = None,
) -> str:
    """Handle missing (null/NaN) values in a specific column.

    Args:
        column: The column to impute.
        strategy: One of 'mean', 'median', 'mode', 'constant', 'drop_rows'.
            'drop_rows' removes rows where this column is null instead of
            filling them.
        constant_value: The fill value to use when strategy == 'constant'.
    """
    return "handled_by_executor"


@tool
def drop_column(columns: List[str]) -> str:
    """Remove one or more columns entirely from the dataset.

    Args:
        columns: List of column names to drop.
    """
    return "handled_by_executor"


@tool
def rename_column(old_name: str, new_name: str) -> str:
    """Rename a single column.

    Args:
        old_name: The column's current name.
        new_name: The new name to give it.
    """
    return "handled_by_executor"


@tool
def convert_dtype(column: str, target_type: str) -> str:
    """Convert a column to a different data type.

    Args:
        column: The column to convert.
        target_type: One of 'int', 'float', 'str', 'datetime', 'category', 'bool'.
    """
    return "handled_by_executor"


@tool
def handle_outliers(
    column: str,
    method: str = "cap",
    iqr_multiplier: float = 1.5,
) -> str:
    """Detect and handle statistical outliers in a numeric column using the IQR method.

    Args:
        column: The numeric column to check for outliers.
        method: 'cap' to clip outlier values to the nearest acceptable bound
            (Winsorize-style, keeps all rows), or 'remove' to drop rows whose
            value in this column falls outside the bounds.
        iqr_multiplier: How many IQRs beyond Q1/Q3 counts as an outlier.
            1.5 is the standard threshold; use a higher value (e.g. 3.0) to
            only flag extreme outliers.
    """
    return "handled_by_executor"


@tool
def standardize_text(
    column: str,
    trim_whitespace: bool = True,
    case: Optional[str] = None,
) -> str:
    """Clean up a text column: trim leading/trailing whitespace and/or
    standardize letter case, so values like 'Mumbai', ' mumbai ', 'MUMBAI'
    become consistent.

    Args:
        column: The text column to standardize.
        trim_whitespace: Whether to strip leading/trailing whitespace.
        case: One of 'lower', 'upper', 'title', or None to leave case as-is.
    """
    return "handled_by_executor"


@tool
def find_and_replace(
    column: str,
    old_value: str,
    new_value: str,
    match_case: bool = True,
) -> str:
    """Replace a specific bad or inconsistent value in a column with a
    corrected value. Useful for fixing typos, inconsistent category labels
    (e.g. 'NY' vs 'New York'), or a stray non-numeric entry in a numeric column.

    Args:
        column: The column to fix.
        old_value: The exact value to find and replace (as it currently
            appears in the data).
        new_value: The value to replace it with.
        match_case: Whether the match should be case-sensitive.
    """
    return "handled_by_executor"


@tool
def split_column(
    column: str,
    delimiter: str,
    new_column_names: List[str],
    drop_original: bool = True,
) -> str:
    """Split a single column into multiple new columns based on a delimiter,
    e.g. splitting 'full_name' by ' ' into 'first_name' and 'last_name'.

    Args:
        column: The column to split.
        delimiter: The character/string to split on.
        new_column_names: Names for the resulting columns, in order. The
            number of names should match the expected number of parts.
        drop_original: Whether to remove the original column after splitting.
    """
    return "handled_by_executor"


@tool
def filter_rows(
    column: str,
    condition: str,
    value: Optional[str] = None,
) -> str:
    """Remove rows that match (or fail) a condition on a column -- e.g.
    dropping rows where age > 150 or where a column is blank.

    Args:
        column: The column to evaluate.
        condition: One of 'greater_than', 'less_than', 'equals', 'not_equals',
            'contains', 'is_null', 'not_null'. Rows matching the condition are
            REMOVED (e.g. condition='greater_than', value='150' on 'age'
            removes every row where age > 150).
        value: The comparison value (not needed for 'is_null'/'not_null').
    """
    return "handled_by_executor"


ALL_TOOLS = [
    drop_duplicates,
    handle_missing_values,
    drop_column,
    rename_column,
    convert_dtype,
    handle_outliers,
    standardize_text,
    find_and_replace,
    split_column,
    filter_rows,
]
