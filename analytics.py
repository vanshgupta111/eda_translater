#analytics.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def dataset_level_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute dataset-wide statistics.
    """
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "duplicate_rows_pct": round(df.duplicated().mean() * 100, 2),
        "total_missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }


def numeric_column_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Compute stats for numeric columns.
    """
    stats = {}

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        stats[col] = {
            "mean": round(series.mean(), 3),
            "median": round(series.median(), 3),
            "std": round(series.std(), 3),
            "skew": round(series.skew(), 3),
            "outlier_count": int(outliers),
        }

    return stats


def categorical_column_stats(
    df: pd.DataFrame,
    categorical_cols: List[str],
    high_cardinality_threshold: int = 50
) -> Dict[str, Any]:
    """
    Compute stats for categorical columns.
    """
    stats = {}

    for col in categorical_cols:
        series = df[col].dropna()

        stats[col] = {
            "unique_values": int(series.nunique()),
            "high_cardinality": series.nunique() > high_cardinality_threshold,
            "top_categories": series.value_counts().head(10).to_dict(),
        }

    return stats


def correlation_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Compute top positive and negative correlations.
    """
    if len(numeric_cols) < 2:
        return {}

    corr_matrix = df[numeric_cols].corr()

    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["feature_1", "feature_2", "correlation"]

    top_positive = (
        corr_pairs.sort_values("correlation", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )

    top_negative = (
        corr_pairs.sort_values("correlation", ascending=True)
        .head(top_n)
        .to_dict(orient="records")
    )

    return {
        "top_positive_correlations": top_positive,
        "top_negative_correlations": top_negative,
    }


def run_analytics(
    df: pd.DataFrame,
    column_profiles: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrates all hard-coded analytics.
    """
    numeric_cols = [
        col for col, meta in column_profiles.items()
        if meta["logical_type"] == "numeric"
    ]

    categorical_cols = [
        col for col, meta in column_profiles.items()
        if meta["logical_type"] == "categorical"
    ]

    return {
        "dataset": dataset_level_stats(df),
        "numeric_columns": numeric_column_stats(df, numeric_cols),
        "categorical_columns": categorical_column_stats(df, categorical_cols),
        "correlations": correlation_analysis(df, numeric_cols),
    }
