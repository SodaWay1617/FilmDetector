from __future__ import annotations

from typing import Any


def extract_top_label(predictions: list[dict[str, Any]]) -> str:
    """Return only the best prediction label for the UI."""
    if not predictions:
        return "unknown"

    return str(predictions[0].get("label", "unknown"))
