#loader.py
import pandas as pd
from typing import Union
from pathlib import Path


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def load_dataset(file: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Args:
        file: File path or file-like object

    Returns:
        pd.DataFrame

    Raises:
        ValueError: If file format is unsupported or data is invalid
    """

    # Case 1: Streamlit file uploader (file-like object)
    if hasattr(file, "name"):
        ext = Path(file.name).suffix.lower()
    else:
        ext = Path(file).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}")

    try:
        if ext == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    # Basic validation
    if df.empty:
        raise ValueError("Dataset is empty")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 columns")

    if df.shape[0] < 5:
        raise ValueError("Dataset must contain at least 5 rows for meaningful EDA")

    return df
