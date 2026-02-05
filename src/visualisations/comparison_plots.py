"""Comparison plots for grouped variants."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .bars import bar_metric
from .plot_utils import label_rows
from .scatter import scatter_latency_accuracy_approaches


def compare_scatter(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any],
    x: str,
    y: str,
    agg: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing groups defined by approach_map."""
    return scatter_latency_accuracy_approaches(
        list(rows),
        x=x,
        y=y,
        approach_map=approach_map,
        agg=agg,
        title=title,
        savepath=savepath,
        show=show,
    )


def compare_bar(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any],
    metric: str,
    group_key: str = "approach",
    agg: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Bar chart comparing a metric across groups."""
    labeled = label_rows(rows, approach_map, group_key=group_key)
    return bar_metric(
        labeled,
        metric=metric,
        group=group_key,
        agg=agg,
        title=title,
        savepath=savepath,
        show=show,
    )


def compare_metric_grid(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any],
    metrics: Sequence[str],
    group_key: str = "approach",
    agg: str | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 200,
    show: bool = False,
):
    """Grid of bar charts for multiple metrics comparing groups."""
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one metric name.")
    labeled = label_rows(rows, approach_map, group_key=group_key)

    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
    )
    axes_list = np.atleast_1d(axes).ravel()
    for idx, metric in enumerate(metrics):
        ax = axes_list[idx]
        bar_metric(
            labeled,
            metric=metric,
            group=group_key,
            agg=agg,
            title=str(metric),
            ax=ax,
            show=False,
        )

    for ax in axes_list[len(metrics) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
