"""
Actual pandas logic for each tool.

Every function takes the current DataFrame plus the LLM-supplied args and
returns (new_dataframe, human_readable_description). Nothing here ever
executes based on LLM-authored code -- only these five, fixed, reviewed
functions ever touch the dataset. This is the safety boundary that keeps
the "tool calling" pattern reliable: the LLM can choose *which* function
and *which arguments*, never *what code runs*.
"""

import pandas as pd


def drop_duplicates_impl(df: pd.DataFrame, subset=None, keep: str = "first"):
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
    before = len(df)
    keep_arg = False if keep == "none" else keep
    new_df = df.drop_duplicates(subset=subset, keep=keep_arg)
    removed = before - len(new_df)
    return new_df, f"Dropped {removed} duplicate row(s) (subset={subset or 'all columns'}, keep={keep})."


def handle_missing_values_impl(df: pd.DataFrame, column: str, strategy: str = "mean", constant_value=None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    new_df = df.copy()
    null_before = int(new_df[column].isnull().sum())

    if strategy == "mean":
        new_df[column] = new_df[column].fillna(new_df[column].mean())
    elif strategy == "median":
        new_df[column] = new_df[column].fillna(new_df[column].median())
    elif strategy == "mode":
        mode_vals = new_df[column].mode(dropna=True)
        if len(mode_vals):
            new_df[column] = new_df[column].fillna(mode_vals.iloc[0])
    elif strategy == "constant":
        if constant_value is None:
            raise ValueError("constant_value is required when strategy='constant'.")
        new_df[column] = new_df[column].fillna(constant_value)
    elif strategy == "drop_rows":
        new_df = new_df.dropna(subset=[column])
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use mean, median, mode, constant, or drop_rows.")

    return new_df, f"Handled {null_before} missing value(s) in '{column}' using strategy='{strategy}'."


def drop_column_impl(df: pd.DataFrame, columns: list[str]):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    new_df = df.drop(columns=columns)
    return new_df, f"Dropped column(s): {columns}."


def rename_column_impl(df: pd.DataFrame, old_name: str, new_name: str):
    if old_name not in df.columns:
        raise ValueError(f"Column '{old_name}' not found.")
    if new_name in df.columns:
        raise ValueError(f"A column named '{new_name}' already exists.")
    new_df = df.rename(columns={old_name: new_name})
    return new_df, f"Renamed column '{old_name}' to '{new_name}'."


def convert_dtype_impl(df: pd.DataFrame, column: str, target_type: str):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    new_df = df.copy()
    try:
        if target_type == "int":
            new_df[column] = pd.to_numeric(new_df[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            new_df[column] = pd.to_numeric(new_df[column], errors="coerce").astype(float)
        elif target_type == "str":
            new_df[column] = new_df[column].astype(str)
        elif target_type == "datetime":
            new_df[column] = pd.to_datetime(new_df[column], errors="coerce")
        elif target_type == "category":
            new_df[column] = new_df[column].astype("category")
        elif target_type == "bool":
            new_df[column] = new_df[column].astype(bool)
        else:
            raise ValueError(f"Unsupported target_type '{target_type}'.")
    except Exception as e:
        raise ValueError(f"Failed to convert '{column}' to {target_type}: {e}")
    return new_df, f"Converted column '{column}' to {target_type}."


# name -> callable(df, **args) -> (new_df, description)
TOOL_EXECUTORS = {
    "drop_duplicates": drop_duplicates_impl,
    "handle_missing_values": handle_missing_values_impl,
    "drop_column": drop_column_impl,
    "rename_column": rename_column_impl,
    "convert_dtype": convert_dtype_impl,
}
