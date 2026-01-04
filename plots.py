#plots.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def _is_numeric(col: str, profiles: Dict[str, Any]) -> bool:
    return profiles[col]["logical_type"] == "numeric"


def _is_categorical(col: str, profiles: Dict[str, Any]) -> bool:
    return profiles[col]["logical_type"] == "categorical"


def _is_datetime(col: str, profiles: Dict[str, Any]) -> bool:
    return profiles[col]["logical_type"] == "datetime"


def generate_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    df[col].dropna().hist(ax=ax, bins=30)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return fig


def generate_boxplot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna(), vert=False)
    ax.set_title(f"Boxplot of {col}")
    ax.set_xlabel(col)
    return fig


def generate_barplot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    counts = df[col].value_counts().head(10)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Top Categories in {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    return fig


def generate_scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y], alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    return fig


def generate_line(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots()
    temp = df[[x, y]].dropna().sort_values(by=x)
    ax.plot(temp[x], temp[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} over {x}")
    return fig


def generate_plots(
    df: pd.DataFrame,
    plot_plan: List[Dict[str, Any]],
    column_profiles: Dict[str, Any]
) -> List[Any]:
    """
    Generate plots based on Gemini plot plan.
    Returns list of matplotlib figures.
    """
    figures = []

    for plot in plot_plan:
        try:
            plot_type = plot.get("type")
            cols = plot.get("columns", [])

            # Validate columns
            if not cols or any(col not in df.columns for col in cols):
                continue

            if plot_type == "hist" and len(cols) == 1:
                col = cols[0]
                if _is_numeric(col, column_profiles):
                    figures.append(generate_histogram(df, col))

            elif plot_type == "box" and len(cols) == 1:
                col = cols[0]
                if _is_numeric(col, column_profiles):
                    figures.append(generate_boxplot(df, col))

            elif plot_type == "bar" and len(cols) == 1:
                col = cols[0]
                if _is_categorical(col, column_profiles):
                    figures.append(generate_barplot(df, col))

            elif plot_type == "scatter" and len(cols) == 2:
                x, y = cols
                if _is_numeric(x, column_profiles) and _is_numeric(y, column_profiles):
                    figures.append(generate_scatter(df, x, y))

            elif plot_type == "line" and len(cols) == 2:
                x, y = cols
                if _is_datetime(x, column_profiles) and _is_numeric(y, column_profiles):
                    figures.append(generate_line(df, x, y))

        except Exception:
            # Fail silently for robustness
            continue

    return figures
