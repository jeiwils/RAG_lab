"""Summary plots for grouped comparisons."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import aggregate, extract_numeric, label_rows


def compare_delta_bar(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any] | None,
    group_a: str,
    group_b: str,
    metrics: Sequence[str],
    group_key: str = "approach",
    agg: str = "mean",
    lower_is_better_metrics: Sequence[str] | None = None,
    figsize: tuple[float, float] = (9, 4),
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 200,
    show: bool = False,
):
    """Bar chart of percent change (group_a vs group_b) for multiple metrics."""
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one metric name.")

    rows = label_rows(rows, approach_map, group_key=group_key) if approach_map else list(rows)

    lower_is_better = set(lower_is_better_metrics or ())
    deltas: List[float] = []
    for metric in metrics:
        a_vals = extract_numeric(
            [r for r in rows if r.get(group_key) == group_a], metric
        )
        b_vals = extract_numeric(
            [r for r in rows if r.get(group_key) == group_b], metric
        )
        if not a_vals or not b_vals:
            deltas.append(float("nan"))
            continue
        a_val = aggregate(a_vals, agg)
        b_val = aggregate(b_vals, agg)
        if b_val == 0 or np.isnan(b_val):
            deltas.append(float("nan"))
            continue
        delta = (a_val - b_val) / abs(b_val) * 100.0
        if metric in lower_is_better:
            delta = -delta
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(metrics))
    ax.bar(xs, deltas)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_ylabel("Improvement vs EXIT (%)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def metric_heatmap(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any] | None,
    metrics: Sequence[str],
    approach_order: Sequence[str] | None = None,
    group_key: str = "approach",
    agg: str = "mean",
    figsize: tuple[float, float] = (10, 4),
    cmap: str = "viridis",
    annotate: bool = True,
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 200,
    show: bool = False,
):
    """Heatmap of groups vs metrics (min-max normalized per metric)."""
    rows = list(rows)
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one metric name.")

    if approach_map:
        order = list(approach_order or approach_map.keys())
        grouped = {
            label: [r for r in rows if approach_map.get(label, lambda _: False)(r)]
            for label in order
        }
    else:
        order = list(
            approach_order
            or sorted({str(r.get(group_key, "unknown")) for r in rows})
        )
        grouped = {label: [r for r in rows if r.get(group_key) == label] for label in order}

    values: List[List[float]] = []
    for label in order:
        items = grouped.get(label, [])
        row_vals: List[float] = []
        for metric in metrics:
            vals = extract_numeric(items, metric)
            if vals:
                row_vals.append(aggregate(vals, agg))
            else:
                row_vals.append(float("nan"))
        values.append(row_vals)

    data = np.array(values, dtype=float)
    normed = _normalize_columns(data)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        normed,
        aspect="auto",
        cmap=cmap,
    )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized per metric (min-max)")

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    continue
                ax.text(j, i, _format_value(data[i, j]), ha="center", va="center")

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _normalize_columns(data: np.ndarray) -> np.ndarray:
    normed = data.copy()
    if normed.size == 0:
        return normed
    for j in range(normed.shape[1]):
        col = normed[:, j]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        vmin = np.nanmin(col)
        vmax = np.nanmax(col)
        if vmax == vmin:
            normed[mask, j] = 0.5
        else:
            normed[mask, j] = (col[mask] - vmin) / (vmax - vmin)
    return normed


def _format_value(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{value:,.0f}"
    if abs_val >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"
