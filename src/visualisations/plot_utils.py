"""Generic helpers for plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


def label_rows(
    rows: Iterable[Dict[str, Any]],
    approach_map: Dict[str, Callable[[Dict[str, Any]], bool]] | None,
    *,
    group_key: str = "approach",
    exclusive: bool = True,
    include_unmatched: bool = False,
    unmatched_label: str = "unmatched",
) -> List[Dict[str, Any]]:
    """Return rows labeled with group_key using approach_map predicates."""
    rows = list(rows)
    if not approach_map:
        return rows

    labeled: List[Dict[str, Any]] = []
    for row in rows:
        matched = False
        for label, pred in approach_map.items():
            if pred(row):
                item = dict(row)
                item[group_key] = label
                labeled.append(item)
                matched = True
                if exclusive:
                    break
        if not matched and include_unmatched:
            item = dict(row)
            item[group_key] = unmatched_label
            labeled.append(item)
    return labeled


def aggregate_by_map(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Callable[[Dict[str, Any]], bool]],
    x_key: str,
    y_key: str,
    agg: str | Callable[[List[float]], float],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in approach_map}
    for row in rows:
        for label, pred in approach_map.items():
            if pred(row):
                grouped[label].append(row)
                break

    points: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        xs = extract_numeric(items, x_key)
        ys = extract_numeric(items, y_key)
        if not xs or not ys:
            continue
        points.append(
            {
                "approach": label,
                x_key: aggregate(xs, agg),
                y_key: aggregate(ys, agg),
            }
        )
    return points


def extract_numeric(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            continue
    return out


def aggregate(values: List[float], agg: str | Callable[[List[float]], float]) -> float:
    if callable(agg):
        return float(agg(values))
    if agg == "mean":
        return float(np.mean(values))
    if agg == "median":
        return float(np.median(values))
    if agg == "min":
        return float(np.min(values))
    if agg == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown agg: {agg}")


def resolve_datasets(base_dir: str, datasets: Sequence[str] | str | None) -> List[str]:
    if datasets is None:
        datasets = "all"
    if isinstance(datasets, str):
        if datasets.lower() in ("all", "*"):
            root = Path(base_dir)
            return sorted(p.name for p in root.iterdir() if p.is_dir())
        return [datasets]
    return list(datasets)
