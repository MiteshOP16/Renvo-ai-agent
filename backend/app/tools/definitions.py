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


ALL_TOOLS = [
    drop_duplicates,
    handle_missing_values,
    drop_column,
    rename_column,
    convert_dtype,
]
