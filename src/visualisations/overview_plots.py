"""Overview plots for grouped comparisons."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .plot_utils import aggregate_by_map
from .pareto import pareto_front_latency_f1
from .scatter import scatter_latency_accuracy_approaches


def approach_scatter(
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


def approach_pareto(
    rows: Iterable[Dict[str, Any]],
    *,
    approach_map: Dict[str, Any],
    x: str,
    y: str,
    agg: str,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Pareto front for grouped approaches (aggregated)."""
    points = aggregate_by_map(
        rows,
        approach_map=approach_map,
        x_key=x,
        y_key=y,
        agg=agg,
    )
    return pareto_front_latency_f1(
        points,
        x=x,
        y=y,
        label_key="approach",
        title=title,
        savepath=savepath,
        show=show,
    )

