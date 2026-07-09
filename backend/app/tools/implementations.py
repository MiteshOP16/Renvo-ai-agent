"""
Actual pandas logic for each tool.

Every function takes the current DataFrame plus the LLM-supplied args and
returns (new_dataframe, human_readable_description). Nothing here ever
executes based on LLM-authored code -- only these fixed, reviewed functions
ever touch the dataset. This is the safety boundary that keeps the "tool
calling" pattern reliable: the LLM can choose *which* function and *which
arguments*, never *what code runs*.
"""

import numpy as np
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


def handle_outliers_impl(df: pd.DataFrame, column: str, method: str = "cap", iqr_multiplier: float = 1.5):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric; convert its dtype first.")

    new_df = df.copy()
    q1 = new_df[column].quantile(0.25)
    q3 = new_df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    outlier_mask = (new_df[column] < lower) | (new_df[column] > upper)
    n_outliers = int(outlier_mask.sum())

    if method == "cap":
        new_df[column] = new_df[column].clip(lower=lower, upper=upper)
        desc = (
            f"Capped {n_outliers} outlier value(s) in '{column}' to the range "
            f"[{round(lower, 2)}, {round(upper, 2)}] (IQR x{iqr_multiplier})."
        )
    elif method == "remove":
        new_df = new_df[~outlier_mask]
        desc = (
            f"Removed {n_outliers} row(s) where '{column}' fell outside "
            f"[{round(lower, 2)}, {round(upper, 2)}] (IQR x{iqr_multiplier})."
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'cap' or 'remove'.")

    return new_df, desc


def standardize_text_impl(df: pd.DataFrame, column: str, trim_whitespace: bool = True, case: str | None = None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    new_df = df.copy()
    series = new_df[column].astype("string")

    if trim_whitespace:
        series = series.str.strip()
    if case == "lower":
        series = series.str.lower()
    elif case == "upper":
        series = series.str.upper()
    elif case == "title":
        series = series.str.title()
    elif case is not None:
        raise ValueError(f"Unknown case '{case}'. Use 'lower', 'upper', 'title', or omit it.")

    new_df[column] = series
    actions = []
    if trim_whitespace:
        actions.append("trimmed whitespace")
    if case:
        actions.append(f"standardized case to {case}")
    return new_df, f"Standardized '{column}': {', '.join(actions) or 'no changes'}."


def find_and_replace_impl(df: pd.DataFrame, column: str, old_value: str, new_value: str, match_case: bool = True):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    new_df = df.copy()

    if match_case:
        mask = new_df[column].astype(str) == str(old_value)
    else:
        mask = new_df[column].astype(str).str.lower() == str(old_value).lower()

    n_matches = int(mask.sum())
    new_df.loc[mask, column] = new_value
    return new_df, f"Replaced {n_matches} occurrence(s) of '{old_value}' with '{new_value}' in '{column}'."


def split_column_impl(
    df: pd.DataFrame,
    column: str,
    delimiter: str,
    new_column_names: list[str],
    drop_original: bool = True,
):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not new_column_names:
        raise ValueError("new_column_names must contain at least one name.")
    for name in new_column_names:
        if name in df.columns and name != column:
            raise ValueError(f"A column named '{name}' already exists.")

    new_df = df.copy()
    split_data = new_df[column].astype(str).str.split(delimiter, n=len(new_column_names) - 1, expand=True)

    for i, name in enumerate(new_column_names):
        new_df[name] = split_data[i] if i in split_data.columns else None

    if drop_original and column not in new_column_names:
        new_df = new_df.drop(columns=[column])

    return new_df, f"Split '{column}' by '{delimiter}' into columns: {new_column_names}."


def filter_rows_impl(df: pd.DataFrame, column: str, condition: str, value: str | None = None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    new_df = df.copy()
    series = new_df[column]
    before = len(new_df)

    if condition == "is_null":
        keep_mask = series.notnull()
    elif condition == "not_null":
        keep_mask = series.isnull()
    elif condition in ("greater_than", "less_than", "equals", "not_equals"):
        if value is None:
            raise ValueError(f"'value' is required for condition '{condition}'.")
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            numeric_value = float(value)
            is_numeric_compare = True
        except ValueError:
            is_numeric_compare = False

        if condition == "greater_than" and is_numeric_compare:
            keep_mask = ~(numeric_series > numeric_value)
        elif condition == "less_than" and is_numeric_compare:
            keep_mask = ~(numeric_series < numeric_value)
        elif condition == "equals":
            keep_mask = series.astype(str) != str(value)
        elif condition == "not_equals":
            keep_mask = series.astype(str) == str(value)
        else:
            raise ValueError(f"Condition '{condition}' requires a numeric column and numeric value.")
    elif condition == "contains":
        if value is None:
            raise ValueError("'value' is required for condition 'contains'.")
        keep_mask = ~series.astype(str).str.contains(str(value), na=False)
    else:
        raise ValueError(
            f"Unknown condition '{condition}'. Use greater_than, less_than, equals, "
            "not_equals, contains, is_null, or not_null."
        )

    new_df = new_df[keep_mask]
    removed = before - len(new_df)
    return new_df, f"Removed {removed} row(s) where '{column}' {condition} {value if value is not None else ''}.".strip()


# name -> callable(df, **args) -> (new_df, description)
TOOL_EXECUTORS = {
    "drop_duplicates": drop_duplicates_impl,
    "handle_missing_values": handle_missing_values_impl,
    "drop_column": drop_column_impl,
    "rename_column": rename_column_impl,
    "convert_dtype": convert_dtype_impl,
    "handle_outliers": handle_outliers_impl,
    "standardize_text": standardize_text_impl,
    "find_and_replace": find_and_replace_impl,
    "split_column": split_column_impl,
    "filter_rows": filter_rows_impl,
}
