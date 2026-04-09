"""
Helpers for extracting JSON objects from LLM responses.
"""

import json
import re
from typing import Any, Dict, Optional


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from free-form LLM output.

    The model may wrap JSON in markdown fences or include extra explanatory text.
    This helper scans for the first decodable JSON object and returns it.
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace(
        "```", ""
    )

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(candidate, dict):
            return candidate

    return None


def ensure_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
