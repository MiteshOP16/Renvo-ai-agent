"""
Converts a pandas DataFrame into a small, safe JSON summary.

This is the ONLY representation of the dataset that ever reaches the LLM.
No raw rows, no cell values (except a few aggregated top-category labels for
categorical columns) are sent. This keeps prompts small, keeps costs down,
and avoids leaking sensitive raw data into a third-party LLM API.
"""

import numpy as np
import pandas as pd


def extract_metadata(df: pd.DataFrame, max_categories: int = 5) -> dict:
    metadata = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [],
    }

    for col in df.columns:
        series = df[col]
        col_info = {
            "name": str(col),
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_percentage": round(float(series.isnull().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info.update(
                {
                    "min": _safe_float(series.min()),
                    "max": _safe_float(series.max()),
                    "mean": _safe_float(series.mean()),
                    "std": _safe_float(series.std()),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info.update(
                {
                    "min": _safe_str(series.min()),
                    "max": _safe_str(series.max()),
                }
            )
        else:
            top = series.value_counts(dropna=True).head(max_categories)
            col_info["top_categories"] = {str(k): int(v) for k, v in top.items()}

        metadata["columns"].append(col_info)

    return metadata


def _safe_float(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)
    except Exception:
        return None


def _safe_str(val):
    try:
        if val is None or pd.isna(val):
            return None
        return str(val)
    except Exception:
        return None
