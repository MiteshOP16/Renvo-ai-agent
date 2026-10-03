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
        null_cnt = int(series.isnull().sum())
        col_info = {
            "name": str(col),
            "dtype": str(series.dtype),
            "null_count": null_cnt,
            "null_percentage": round(float(series.isnull().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
            "outlier_count": 0,
            "outlier_examples": [],
            "issues": [],
        }

        clean_series = series.dropna()
        if pd.api.types.is_numeric_dtype(series):
            col_info.update(
                {
                    "min": _safe_float(series.min()),
                    "max": _safe_float(series.max()),
                    "mean": _safe_float(series.mean()),
                    "std": _safe_float(series.std()),
                }
            )
            if len(clean_series) >= 4:
                try:
                    q1 = float(clean_series.quantile(0.25))
                    q3 = float(clean_series.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_b = q1 - 1.5 * iqr
                        upper_b = q3 + 1.5 * iqr
                        outliers = clean_series[(clean_series < lower_b) | (clean_series > upper_b)]
                        col_info["outlier_count"] = int(outliers.count())
                        if col_info["outlier_count"] > 0:
                            col_info["outlier_examples"] = [_safe_float(v) for v in outliers.head(2).tolist()]
                except Exception:
                    pass
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

        issues = []
        if null_cnt > 0:
            issues.append(f"{null_cnt} missing")
        if col_info["outlier_count"] > 0:
            examples_str = ""
            if col_info["outlier_examples"]:
                examples_str = f" ({', '.join(str(x) for x in col_info['outlier_examples'])})"
            issues.append(f"{col_info['outlier_count']} outlier{'s' if col_info['outlier_count'] > 1 else ''}{examples_str}")

        col_info["issues"] = issues
        col_info["is_clean"] = len(issues) == 0

        metadata["columns"].append(col_info)

    metadata["total_outliers"] = sum(c["outlier_count"] for c in metadata["columns"])
    metadata["total_issues"] = sum(len(c["issues"]) for c in metadata["columns"])

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
