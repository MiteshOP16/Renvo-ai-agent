"""
Actual pandas logic for each MUTATING tool (i.e. tools that change the
dataset and create a new undo/redo version). Read-only analysis tools live
separately in analysis_implementations.py, since they never touch the
DataFrame version stack.

Every function takes the current DataFrame plus the LLM-supplied args and
returns (new_dataframe, human_readable_description). Nothing here ever
executes based on LLM-authored code -- only these fixed, reviewed functions
ever touch the dataset.

Every function that takes a `column`/`columns` argument resolves it through
column_resolver.resolve_column(s) first. This means harmless mismatches
(case, stray whitespace) succeed automatically, and genuine mismatches raise
a ColumnNotFoundError that already lists the real columns and close-match
suggestions -- giving the agent what it needs to self-correct or ask the
user, instead of guessing or claiming a change happened when it didn't.
"""

import numpy as np
import pandas as pd

from app.tools.column_resolver import resolve_column, resolve_columns


def drop_duplicates_impl(df: pd.DataFrame, subset=None, keep: str = "first"):
    resolved_subset = resolve_columns(df, subset) if subset else None
    before = len(df)
    keep_arg = False if keep == "none" else keep
    new_df = df.drop_duplicates(subset=resolved_subset, keep=keep_arg)
    removed = before - len(new_df)
    return new_df, f"Dropped {removed} duplicate row(s) (subset={resolved_subset or 'all columns'}, keep={keep})."


def handle_missing_values_impl(df: pd.DataFrame, column: str, strategy: str = "mean", constant_value=None):
    column = resolve_column(df, column)
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
    resolved = resolve_columns(df, columns)
    new_df = df.drop(columns=resolved)
    return new_df, f"Dropped column(s): {resolved}."


def rename_column_impl(df: pd.DataFrame, old_name: str, new_name: str):
    old_name = resolve_column(df, old_name)
    if new_name in df.columns:
        raise ValueError(f"A column named '{new_name}' already exists.")
    new_df = df.rename(columns={old_name: new_name})
    return new_df, f"Renamed column '{old_name}' to '{new_name}'."


def convert_dtype_impl(df: pd.DataFrame, column: str, target_type: str):
    column = resolve_column(df, column)
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
    column = resolve_column(df, column)
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
    column = resolve_column(df, column)
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
    column = resolve_column(df, column)
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
    column = resolve_column(df, column)
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
    column = resolve_column(df, column)
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


def merge_columns_impl(
    df: pd.DataFrame,
    columns: list[str],
    new_column_name: str,
    separator: str = " ",
    drop_original: bool = True,
):
    resolved = resolve_columns(df, columns)
    if len(resolved) < 2:
        raise ValueError("merge_columns needs at least 2 columns.")

    new_df = df.copy()
    new_df[new_column_name] = new_df[resolved].astype(str).agg(separator.join, axis=1)
    if drop_original:
        new_df = new_df.drop(columns=[c for c in resolved if c != new_column_name])

    return new_df, f"Merged columns {resolved} into '{new_column_name}' using separator '{separator}'."


_DT_PART_EXTRACTORS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "weekday": lambda s: s.dt.day_name(),
    "hour": lambda s: s.dt.hour,
    "minute": lambda s: s.dt.minute,
    "quarter": lambda s: s.dt.quarter,
}


def extract_datetime_parts_impl(df: pd.DataFrame, column: str, parts: list[str], drop_original: bool = False):
    column = resolve_column(df, column)
    unknown = [p for p in parts if p not in _DT_PART_EXTRACTORS]
    if unknown:
        raise ValueError(f"Unsupported datetime part(s): {unknown}. Use {list(_DT_PART_EXTRACTORS)}.")

    new_df = df.copy()
    dt_series = pd.to_datetime(new_df[column], errors="coerce")
    for part in parts:
        new_df[f"{column}_{part}"] = _DT_PART_EXTRACTORS[part](dt_series)
    if drop_original:
        new_df = new_df.drop(columns=[column])

    return new_df, f"Extracted {parts} from '{column}' into new column(s)."


def remove_type_anomalies_impl(df: pd.DataFrame, column: str, expected_type: str):
    column = resolve_column(df, column)
    new_df = df.copy()
    if expected_type == "numeric":
        coerced = pd.to_numeric(new_df[column], errors="coerce")
    elif expected_type == "datetime":
        coerced = pd.to_datetime(new_df[column], errors="coerce")
    else:
        raise ValueError("expected_type must be 'numeric' or 'datetime'.")

    was_present = new_df[column].notnull()
    now_null = coerced.isnull()
    anomaly_mask = was_present & now_null
    n_anomalies = int(anomaly_mask.sum())

    new_df.loc[anomaly_mask, column] = None
    return new_df, f"Flagged {n_anomalies} value(s) in '{column}' that don't look like {expected_type} and set them to null."


def clip_numeric_range_impl(df: pd.DataFrame, column: str, min_value: float | None = None, max_value: float | None = None):
    column = resolve_column(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric; convert its dtype first.")
    if min_value is None and max_value is None:
        raise ValueError("Provide at least one of min_value or max_value.")

    new_df = df.copy()
    before = new_df[column].copy()
    new_df[column] = new_df[column].clip(lower=min_value, upper=max_value)
    n_changed = int((before != new_df[column]).sum())

    return new_df, f"Clamped {n_changed} value(s) in '{column}' to range [{min_value}, {max_value}]."


def sort_dataset_impl(df: pd.DataFrame, column: str, ascending: bool = True):
    column = resolve_column(df, column)
    new_df = df.sort_values(by=column, ascending=ascending).reset_index(drop=True)
    return new_df, f"Sorted dataset by '{column}' ({'ascending' if ascending else 'descending'})."


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
    "merge_columns": merge_columns_impl,
    "extract_datetime_parts": extract_datetime_parts_impl,
    "remove_type_anomalies": remove_type_anomalies_impl,
    "clip_numeric_range": clip_numeric_range_impl,
    "sort_dataset": sort_dataset_impl,
}