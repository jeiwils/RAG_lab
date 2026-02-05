"""Scatter plot helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np

CONFIG = {
    "x": "wall_time_sec_mean",
    "y": "mean_f1",
    "approach_key": "approach",
    "agg": "mean",
    "exclusive": True,
    "label_points": True,
    "title": None,
    "figsize": (6, 4),
    "grid_alpha": 0.3,
    "dpi": 200,
}


import matplotlib.pyplot as plt


def scatter_latency_accuracy_approaches(
    rows: Iterable[Dict[str, Any]],
    *,
    x: str | None = None,
    y: str | None = None,
    approach_key: str | None = None,
    approach_map: Mapping[str, Callable[[Dict[str, Any]], bool]] | None = None,
    agg: str | Callable[[List[float]], float] | None = None,
    exclusive: bool | None = None,
    label_points: bool | None = None,
    title: str | None = None,
    ax=None,
    savepath: str | None = None,
    show: bool = False,
):
    """Plot latency vs accuracy aggregated by approach.

    If approach_map is provided, rows are assigned to the first matching label
    (exclusive=True). This is useful for custom groupings like:
      {"System_A": lambda r: r["variant"].startswith("sys_a_"), ...}
    """
    rows = list(rows)
    x = x or CONFIG["x"]
    y = y or CONFIG["y"]
    approach_key = approach_key or CONFIG["approach_key"]
    agg = agg or CONFIG["agg"]
    if exclusive is None:
        exclusive = CONFIG["exclusive"]
    if label_points is None:
        label_points = CONFIG["label_points"]
    if title is None:
        title = CONFIG["title"]

    groups = _group_rows(rows, approach_key, approach_map, exclusive=exclusive)

    points: List[Tuple[str, float, float]] = []
    for label, items in groups.items():
        xs = _extract_numeric(items, x)
        ys = _extract_numeric(items, y)
        if not xs or not ys:
            continue
        points.append((label, _aggregate(xs, agg), _aggregate(ys, agg)))

    if ax is None:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    else:
        fig = ax.figure

    if points:
        x_vals = [p[1] for p in points]
        y_vals = [p[2] for p in points]
        ax.scatter(x_vals, y_vals)
        if label_points:
            for label, xv, yv in points:
                ax.annotate(label, (xv, yv), textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=CONFIG["grid_alpha"])

    if savepath:
        fig.savefig(savepath, dpi=CONFIG["dpi"], bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _group_rows(
    rows: List[Dict[str, Any]],
    approach_key: str,
    approach_map: Mapping[str, Callable[[Dict[str, Any]], bool]] | None,
    *,
    exclusive: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if approach_map:
        for row in rows:
            matched = False
            for label, pred in approach_map.items():
                if pred(row):
                    groups[label].append(row)
                    matched = True
                    if exclusive:
                        break
            if not matched and not exclusive:
                groups.setdefault("unmatched", []).append(row)
        return dict(groups)

    for row in rows:
        label = str(row.get(approach_key, "unknown"))
        groups[label].append(row)
    return dict(groups)


def _extract_numeric(rows: List[Dict[str, Any]], key: str) -> List[float]:
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


def _aggregate(values: List[float], agg: str | Callable[[List[float]], float]) -> float:
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
