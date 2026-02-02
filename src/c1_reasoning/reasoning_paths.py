"""Path helpers for traversal artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

__all__ = ["traversal_paths"]


def traversal_paths(model: str, dataset: str, split: str, variant: str) -> Dict[str, Path]:
    """Return standard paths for traversal artifacts."""

    base = Path(f"data/traversal/{model}/{dataset}/{split}/{variant}")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "results": base / "per_query_traversal_results.jsonl",
        "visited_passages": base / "visited_passages.json",
        "stats": base / "final_traversal_stats.json",
    }
