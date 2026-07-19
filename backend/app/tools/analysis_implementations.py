"""
Read-only ANALYSIS tools -- these answer questions about the dataset but
never modify it, so they never create a new undo/redo version. Pulled from
the "Column Analysis" / "Visualization" feature areas of the platform
inventory: column profiling, correlation, outlier reporting, value counts,
and missing-data pattern detection.

Every function returns a single human-readable report string (no DataFrame).
This is a deliberate, separate contract from tools/implementations.py's
(new_df, description) mutating tools -- the graph checks which registry a
tool name belongs to and skips the version-stack write for anything here.

Like the mutating tools, every function that takes a `column` argument
resolves it through column_resolver first, so the same case/whitespace
robustness and friendly "column not found" errors apply here too.
"""

import numpy as np
import pandas as pd

from app.tools.column_resolver import resolve_column


def profile_column_impl(df: pd.DataFrame, column: str) -> str:
    column = resolve_column(df, column)
    series = df[column]
    n = len(series)
    missing = int(series.isnull().sum())
    missing_pct = round(missing / n * 100, 2) if n else 0.0
    unique = int(series.nunique(dropna=True))

    # simple, transparent heuristic quality score -- not a statistical model,
    # just a quick signal: penalize missingness and near-constant columns
    quality_score = 100.0
    quality_score -= missing_pct
    if n and unique == 1:
        quality_score -= 20
    quality_score = max(0.0, round(quality_score, 1))

    lines = [
        f"Profile for '{column}' (dtype: {series.dtype}):",
        f"- Row count: {n}, missing: {missing} ({missing_pct}%), unique values: {unique}",
        f"- Estimated data quality score: {quality_score}/100",
    ]

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        skew = series.skew()
        kurt = series.kurt()
        lines.append(
            f"- min={round(desc.get('min', float('nan')), 3)}, "
            f"max={round(desc.get('max', float('nan')), 3)}, "
            f"mean={round(desc.get('mean', float('nan')), 3)}, "
            f"median={round(series.median(), 3)}, "
            f"std={round(desc.get('std', float('nan')), 3)}"
        )
        lines.append(f"- skewness={round(skew, 3) if pd.notna(skew) else 'n/a'}, kurtosis={round(kurt, 3) if pd.notna(kurt) else 'n/a'}")
    else:
        top = series.value_counts(dropna=True).head(5)
        top_str = ", ".join(f"{k!r}: {v}" for k, v in top.items())
        lines.append(f"- Top values: {top_str or 'n/a'}")

    return "\n".join(lines)


def compute_correlation_impl(df: pd.DataFrame, method: str = "pearson", threshold: float = 0.7) -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns to compute correlations (need at least 2)."

    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    corr = numeric_df.corr(method=method)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(float(r), 3)))

    if not pairs:
        return f"No column pairs found with |correlation| >= {threshold} ({method})."

    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    lines = [f"Column pairs with |correlation| >= {threshold} ({method}):"]
    for a, b, r in pairs:
        lines.append(f"- {a} <-> {b}: r={r}")
    return "\n".join(lines)


def detect_outliers_report_impl(df: pd.DataFrame, column: str | None = None, iqr_multiplier: float = 1.5) -> str:
    if column is not None:
        columns = [resolve_column(df, column)]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        return "No numeric columns available to check for outliers."

    lines = ["Outlier report (IQR method, multiplier={}):".format(iqr_multiplier)]
    for col in columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        n_outliers = int(mask.sum())
        pct = round(n_outliers / len(series) * 100, 2) if len(series) else 0.0
        severity = "none" if n_outliers == 0 else "low" if pct < 2 else "moderate" if pct < 10 else "high"
        lines.append(
            f"- {col}: {n_outliers} outlier(s) ({pct}%), severity={severity}, "
            f"bounds=[{round(lower, 2)}, {round(upper, 2)}]"
        )
    return "\n".join(lines)


def value_counts_impl(df: pd.DataFrame, column: str, top_n: int = 10) -> str:
    column = resolve_column(df, column)
    counts = df[column].value_counts(dropna=False).head(top_n)
    total = len(df)
    lines = [f"Top {min(top_n, len(counts))} value(s) in '{column}' (of {total} rows):"]
    for val, count in counts.items():
        pct = round(count / total * 100, 2) if total else 0.0
        label = "null" if pd.isna(val) else repr(val)
        lines.append(f"- {label}: {count} ({pct}%)")
    return "\n".join(lines)


def detect_missing_pattern_impl(df: pd.DataFrame, column: str) -> str:
    column = resolve_column(df, column)
    series = df[column]
    n = len(series)
    is_null = series.isnull().to_numpy()
    n_missing = int(is_null.sum())

    if n_missing == 0:
        return f"'{column}' has no missing values."
    if n_missing == n:
        return f"'{column}' is entirely missing (100%)."

    # crude positional heuristic: where in the row order do the nulls sit?
    null_positions = np.where(is_null)[0]
    first_third = n / 3
    last_third = 2 * n / 3
    frac_in_first = float((null_positions < first_third).mean())
    frac_in_last = float((null_positions >= last_third).mean())

    # longest consecutive run of missing values
    max_run = 0
    current_run = 0
    for v in is_null:
        current_run = current_run + 1 if v else 0
        max_run = max(max_run, current_run)

    if frac_in_first > 0.6:
        pattern = "front-loaded (missing values cluster near the start of the dataset)"
    elif frac_in_last > 0.6:
        pattern = "tail-loaded (missing values cluster near the end of the dataset)"
    elif max_run >= max(5, n_missing * 0.5):
        pattern = "systematic block (missing values occur in one or more long consecutive runs)"
    else:
        pattern = "sporadic / roughly random"

    pct = round(n_missing / n * 100, 2)
    return (
        f"'{column}' missing-data pattern: {pattern}. "
        f"{n_missing} missing value(s) ({pct}%), longest consecutive run: {max_run}."
    )


# name -> callable(df, **args) -> report string (no DataFrame mutation)
ANALYSIS_TOOL_EXECUTORS = {
    "profile_column": profile_column_impl,
    "compute_correlation": compute_correlation_impl,
    "detect_outliers_report": detect_outliers_report_impl,
    "value_counts": value_counts_impl,
    "detect_missing_pattern": detect_missing_pattern_impl,
}