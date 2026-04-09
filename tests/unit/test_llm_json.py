"""Unit tests for LLM JSON parsing helpers."""

from app.core.services.llm_json import ensure_string_list, extract_first_json_object


def test_extract_first_json_object_from_markdown_fence() -> None:
    text = """
    Here is the result:
    ```json
    {"answer": "ok", "score": 3}
    ```
    """

    parsed = extract_first_json_object(text)

    assert parsed == {"answer": "ok", "score": 3}


def test_extract_first_json_object_from_mixed_text() -> None:
    text = "prefix {" + '"items": [1, 2], "status": "done"' + "} trailing"

    parsed = extract_first_json_object(text)

    assert parsed == {"items": [1, 2], "status": "done"}


def test_extract_first_json_object_returns_none_for_non_json() -> None:
    text = "No structured output available"

    parsed = extract_first_json_object(text)

    assert parsed is None


def test_ensure_string_list_normalizes_values() -> None:
    values = ["one", 2, "  ", "three"]

    normalized = ensure_string_list(values)

    assert normalized == ["one", "2", "three"]
