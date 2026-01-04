#profiling.py
import pandas as pd
from typing import Dict, Any


def infer_column_type(series: pd.Series) -> str:
    """
    Infer logical column type.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Object / string handling
    if pd.api.types.is_object_dtype(series):
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

        # Heuristic: high cardinality → text
        if unique_ratio > 0.5:
            return "text"
        else:
            return "categorical"

    return "unknown"


def profile_column(series: pd.Series) -> Dict[str, Any]:
    """
    Generate profiling info for a single column.
    """
    col_type = infer_column_type(series)

    profile = {
        "dtype": str(series.dtype),
        "logical_type": col_type,
        "non_null_count": int(series.notnull().sum()),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(series.isnull().mean() * 100, 2),
        "unique_values": int(series.nunique(dropna=True)),
    }

    # Type-specific profiling
    if col_type == "numeric":
        profile.update({
            "min": series.min(),
            "max": series.max(),
            "mean": round(series.mean(), 3),
            "median": series.median(),
            "std": round(series.std(), 3),
            "skew": round(series.skew(), 3),
        })

    elif col_type == "categorical":
        top_values = series.value_counts(dropna=True).head(5)
        profile["top_values"] = top_values.to_dict()

    elif col_type == "datetime":
        profile.update({
            "min_date": series.min(),
            "max_date": series.max(),
        })

    elif col_type == "text":
        sample_values = series.dropna().astype(str).head(5).tolist()
        profile["sample_values"] = sample_values

    return profile


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile entire DataFrame column-wise.
    """
    column_profiles = {}

    for col in df.columns:
        column_profiles[col] = profile_column(df[col])

    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": column_profiles
    }
