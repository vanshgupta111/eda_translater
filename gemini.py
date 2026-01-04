# gemini.py
import json
import os
import re
from typing import Dict, Any

import numpy as np
import google.generativeai as genai

# ------------------ CONFIG ------------------ #

MODEL_NAME = "gemini-2.5-flash"

ALLOWED_PLOTS = {
    "numeric": ["hist", "box"],
    "categorical": ["bar"],
    "numeric_numeric": ["scatter"],
    "datetime_numeric": ["line"]
}

# ------------------ INIT ------------------ #

def init_gemini(api_key: str | None = None):
    """
    Initialize Gemini API.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key not provided")

    genai.configure(api_key=key)

# ------------------ TYPE CONVERSION ------------------ #

def convert_to_python(obj):
    """
    Recursively convert numpy/Pandas types to native Python types for JSON.
    """
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

# ------------------ PROMPT ------------------ #

def build_prompt(
    column_profiles: Dict[str, Any],
    analytics: Dict[str, Any]
) -> str:
    """
    Build structured prompt for Gemini.
    """
    # Convert all types to Python-native
    column_profiles_clean = convert_to_python(column_profiles)
    analytics_clean = convert_to_python(analytics)

    prompt = f"""
You are a senior data analyst.

You are given metadata from an automated EDA system.
You DO NOT have access to raw data.

Your tasks:
1. Summarize the dataset in plain English.
2. Identify key data quality issues.
3. Suggest meaningful visualizations.
4. Highlight 4–6 key analytical insights.
5. Suggest potential machine learning tasks (if applicable).

Rules:
- Use ONLY the provided metadata.
- Suggest plots ONLY from allowed types.
- Output MUST be valid JSON.
- Do NOT include explanations outside JSON.

Allowed plot types:
- Numeric: hist, box
- Categorical: bar
- Numeric–Numeric: scatter
- Datetime–Numeric: line

Metadata:
Column profiles:
{json.dumps(column_profiles_clean, indent=2)}

Analytics summary:
{json.dumps(analytics_clean, indent=2)}

Output JSON schema:
{{
  "dataset_summary": "string",
  "data_quality_issues": ["string"],
  "plots": [
    {{
      "type": "hist|box|bar|scatter|line",
      "columns": ["col1", "col2 (if applicable)"]
    }}
  ],
  "key_insights": ["string"],
  "ml_suggestions": ["string"]
}}
"""
    return prompt.strip()

# ------------------ SAFE JSON PARSING ------------------ #

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON safely from Gemini output.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Gemini response did not contain valid JSON.")

# ------------------ GEMINI CALL ------------------ #

def call_gemini(prompt: str) -> Dict[str, Any]:
    """
    Call Gemini and parse JSON response safely.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini.")

        return _extract_json(response.text)

    except Exception as e:
        # IMPORTANT: Never crash the app
        return {
            "dataset_summary": "Gemini insights could not be generated.",
            "data_quality_issues": [str(e)],
            "plots": [],
            "key_insights": [],
            "ml_suggestions": []
        }

# ------------------ ORCHESTRATOR ------------------ #

def get_gemini_insights(
    column_profiles: Dict[str, Any],
    analytics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entry point.
    """
    prompt = build_prompt(column_profiles, analytics)
    return call_gemini(prompt)
