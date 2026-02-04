"""Bar chart helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

CONFIG = {
    "metric": "mean_f1",
    "group": "retriever",
    "hue": None,
    "agg": "mean",
    "title": None,
    "figsize": (7, 4),
    "grid_alpha": 0.3,
    "dpi": 200,
}


import matplotlib.pyplot as plt


def bar_metric(
    rows: Iterable[Dict[str, Any]],
    *,
    metric: str | None = None,
    group: str | None = None,
    hue: str | None = None,
    agg: str | Callable[[List[float]], float] | None = None,
    title: str | None = None,
    ax=None,
    savepath: str | None = None,
    show: bool = False,
):
    """Plot a (grouped) bar chart of a metric.

    If hue is set, bars are grouped by `group` and colored by `hue`.
    """
    rows = list(rows)
    metric = metric or CONFIG["metric"]
    group = group or CONFIG["group"]
    if hue is None:
        hue = CONFIG["hue"]
    agg = agg or CONFIG["agg"]
    if title is None:
        title = CONFIG["title"]
    grouped = _group_rows(rows, group, hue)
    groups = sorted(grouped.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    else:
        fig = ax.figure

    if hue is None:
        vals = [_aggregate(_extract_metric(grouped[g], metric), agg) for g in groups]
        ax.bar(groups, vals)
    else:
        hue_keys = sorted({row.get(hue, "unknown") for row in rows})
        width = 0.8 / max(len(hue_keys), 1)
        for i, h in enumerate(hue_keys):
            vals = []
            for g in groups:
                vals.append(
                    _aggregate(
                        _extract_metric(grouped[g].get(h, []), metric),
                        agg,
                    )
                )
            x = np.arange(len(groups)) + i * width
            ax.bar(x, vals, width=width, label=str(h))
        ax.set_xticks(np.arange(len(groups)) + width * (len(hue_keys) - 1) / 2)
        ax.set_xticklabels(groups)
        ax.legend()

    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=CONFIG["grid_alpha"])

    if savepath:
        fig.savefig(savepath, dpi=CONFIG["dpi"], bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _group_rows(
    rows: List[Dict[str, Any]],
    group: str,
    hue: str | None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]] | Dict[str, List[Dict[str, Any]]]:
    if hue is None:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(group, "unknown"))].append(row)
        return grouped

    grouped_hue: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        g = str(row.get(group, "unknown"))
        h = str(row.get(hue, "unknown"))
        grouped_hue[g][h].append(row)
    return grouped_hue


def _extract_metric(rows: List[Dict[str, Any]], metric: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        try:
            vals.append(float(val))
        except (TypeError, ValueError):
            continue
    return vals


def _aggregate(values: List[float], agg: str | Callable[[List[float]], float]) -> float:
    if not values:
        return float("nan")
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
