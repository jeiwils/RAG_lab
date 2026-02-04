"""Pareto front plotting helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    "x": "wall_time_sec_mean",
    "y": "mean_f1",
    "label_key": "approach",
    "title": None,
    "figsize": (6, 4),
    "grid_alpha": 0.3,
    "dpi": 200,
    "color_all": "tab:blue",
    "color_front": "tab:red",
}


def pareto_front_latency_f1(
    rows: Iterable[Dict[str, Any]],
    *,
    x: str | None = None,
    y: str | None = None,
    label_key: str | None = None,
    title: str | None = None,
    label_points: bool = True,
    ax=None,
    savepath: str | None = None,
    show: bool = False,
):
    """Plot latency vs mean_f1 with nondominated points highlighted."""
    rows = list(rows)
    x = x or CONFIG["x"]
    y = y or CONFIG["y"]
    label_key = label_key or CONFIG["label_key"]
    if title is None:
        title = CONFIG["title"]

    points = _extract_points(rows, x, y, label_key)
    if not points:
        if ax is None:
            fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        else:
            fig = ax.figure
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=CONFIG["grid_alpha"])
        if title:
            ax.set_title(title)
        return fig, ax

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    labels = [p[2] for p in points]

    front_mask = _pareto_nondominated(xs, ys)

    if ax is None:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    else:
        fig = ax.figure

    ax.scatter(xs, ys, color=CONFIG["color_all"], alpha=0.7)
    ax.scatter(
        xs[front_mask],
        ys[front_mask],
        color=CONFIG["color_front"],
        alpha=0.9,
    )

    if label_points:
        for xv, yv, label in points:
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


def _extract_points(
    rows: List[Dict[str, Any]],
    x: str,
    y: str,
    label_key: str,
) -> List[Tuple[float, float, str]]:
    points: List[Tuple[float, float, str]] = []
    for row in rows:
        xv = row.get(x)
        yv = row.get(y)
        if xv is None or yv is None:
            continue
        try:
            xv_f = float(xv)
            yv_f = float(yv)
        except (TypeError, ValueError):
            continue
        label = str(row.get(label_key, row.get("variant", "point")))
        points.append((xv_f, yv_f, label))
    return points


def _pareto_nondominated(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return boolean mask of nondominated points (min x, max y)."""
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    max_y = -np.inf
    keep = np.zeros_like(xs_sorted, dtype=bool)
    for i in range(len(xs_sorted)):
        yv = ys_sorted[i]
        if yv > max_y:
            keep[i] = True
            max_y = yv

    mask = np.zeros_like(keep)
    mask[order] = keep
    return mask
