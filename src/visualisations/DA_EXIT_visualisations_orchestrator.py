"""Orchestrate DA_EXIT/EXIT and RAG summary plots.

Plot reference (axis meanings):
- Intro scatter: x = wall_time_sec_mean, y = mean_f1, points aggregated by approach
  (DA_EXIT, EXIT, dense, sparse, hybrid).
- Intro cost scatter: x = tokens_per_query_mean, y = mean_f1 (or configured y).
- DA_EXIT vs EXIT scatter: x = wall_time_sec_mean, y = mean_f1, filtered to one retriever.
- DA_EXIT vs EXIT cost scatter: x = tokens_per_query_mean, y = mean_f1 (or configured y).
        - DA_EXIT vs EXIT metric grid: multiple bar charts for compare_metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .bars import bar_metric
from .data import load_summary_metrics
from .pareto import pareto_front_latency_f1
from .scatter import scatter_latency_accuracy_approaches

CONFIG = {
    "base_dir": "data/results/Qwen/Qwen2.5-7B-Instruct",
    "dataset": "musique",
    "datasets": "all",
    "split": "train",
    "save_dir": "data/results/plots",
    "agg": "mean",
    "scatter_x": "wall_time_sec_mean",
    "scatter_y": "mean_f1",
    "cost_x": "tokens_per_query_mean",
    "intro_title": None,
    "compare_title": None,
    "da_exit_variant_prefix": "da_exit_",
    "exit_variant_prefix": "exit_",
    "rag_variant_prefixes": ("rag_", "baseline_rag_", "baseline_embeddings_rag_"),
    "da_exit_sentence_mode": "discourse_aware_sentences",
    "exit_sentence_mode": "standard_sentences",
    "compare_retriever": "hybrid",
    "compare_retrievers": ("dense", "sparse", "hybrid"),
    "compare_metrics": (
        "mean_em",
        "mean_f1",
        "wall_time_sec_mean",
        "tokens_per_query_mean",
    ),
    "compare_grid_figsize": (7, 4),
    "save_ext": "png",
    "save_dpi": 200,
}


def load_rows(
    *,
    base_dir: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    """Load summary metrics rows with defaults from CONFIG."""
    return load_summary_metrics(
        base_dir or CONFIG["base_dir"],
        dataset=dataset or CONFIG["dataset"],
        split=split or CONFIG["split"],
    )


def intro_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing the 5 approaches (DA_EXIT, EXIT, dense, sparse, hybrid)."""
    rows = list(rows) if rows is not None else load_rows()
    approach_map = _intro_approach_map()
    return scatter_latency_accuracy_approaches(
        rows,
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
        approach_map=approach_map,
        agg=CONFIG["agg"],
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_pareto(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Pareto front for the 5 approaches (aggregated)."""
    rows = list(rows) if rows is not None else load_rows()
    approach_map = _intro_approach_map()
    points = _aggregate_by_approach(
        rows,
        approach_map=approach_map,
        x_key=CONFIG["scatter_x"],
        y_key=CONFIG["scatter_y"],
        agg=CONFIG["agg"],
    )
    return pareto_front_latency_f1(
        points,
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
        label_key="approach",
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_cost_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    y: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing approaches by cost vs accuracy."""
    rows = list(rows) if rows is not None else load_rows()
    approach_map = _intro_approach_map()
    return scatter_latency_accuracy_approaches(
        rows,
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
        approach_map=approach_map,
        agg=CONFIG["agg"],
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_cost_pareto(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    y: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Pareto front for cost vs accuracy (aggregated)."""
    rows = list(rows) if rows is not None else load_rows()
    approach_map = _intro_approach_map()
    points = _aggregate_by_approach(
        rows,
        approach_map=approach_map,
        x_key=CONFIG["cost_x"],
        y_key=y or CONFIG["scatter_y"],
        agg=CONFIG["agg"],
    )
    return pareto_front_latency_f1(
        points,
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
        label_key="approach",
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def da_exit_vs_exit_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    retriever: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing DA_EXIT vs EXIT for a single retriever."""
    rows = list(rows) if rows is not None else load_rows()
    retriever = retriever or CONFIG["compare_retriever"]
    rows = _filter_da_exit_exit(rows, retriever=retriever)
    rows = _label_da_exit_exit(rows)
    approach_map = {
        "DA_EXIT": lambda r: r.get("approach") == "DA_EXIT",
        "EXIT": lambda r: r.get("approach") == "EXIT",
    }
    return scatter_latency_accuracy_approaches(
        rows,
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
        approach_map=approach_map,
        agg=CONFIG["agg"],
        title=title or CONFIG["compare_title"],
        savepath=savepath,
        show=show,
    )


def da_exit_vs_exit_cost_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    retriever: str | None = None,
    y: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing DA_EXIT vs EXIT by cost vs accuracy."""
    rows = list(rows) if rows is not None else load_rows()
    retriever = retriever or CONFIG["compare_retriever"]
    rows = _filter_da_exit_exit(rows, retriever=retriever)
    rows = _label_da_exit_exit(rows)
    approach_map = {
        "DA_EXIT": lambda r: r.get("approach") == "DA_EXIT",
        "EXIT": lambda r: r.get("approach") == "EXIT",
    }
    return scatter_latency_accuracy_approaches(
        rows,
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
        approach_map=approach_map,
        agg=CONFIG["agg"],
        title=title or CONFIG["compare_title"],
        savepath=savepath,
        show=show,
    )


def da_exit_vs_exit_bar(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    retriever: str | None = None,
    metric: str = "mean_f1",
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Bar chart comparing DA_EXIT vs EXIT for a single retriever."""
    rows = list(rows) if rows is not None else load_rows()
    retriever = retriever or CONFIG["compare_retriever"]
    rows = _filter_da_exit_exit(rows, retriever=retriever)
    rows = _label_da_exit_exit(rows)
    return bar_metric(
        rows,
        metric=metric,
        group="approach",
        title=title or CONFIG["compare_title"],
        savepath=savepath,
        show=show,
    )


def da_exit_vs_exit_metric_grid(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    retriever: str | None = None,
    metrics: Sequence[str] | None = None,
    ncols: int = 2,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Grid of bar charts for multiple metrics comparing DA_EXIT vs EXIT."""
    rows = list(rows) if rows is not None else load_rows()
    retriever = retriever or CONFIG["compare_retriever"]
    metrics = list(metrics or CONFIG["compare_metrics"])
    if not metrics:
        raise ValueError("metrics must contain at least one metric name.")
    rows = _filter_da_exit_exit(rows, retriever=retriever)
    rows = _label_da_exit_exit(rows)

    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            CONFIG["compare_grid_figsize"][0] * ncols,
            CONFIG["compare_grid_figsize"][1] * nrows,
        ),
    )
    axes_list = np.atleast_1d(axes).ravel()
    for idx, metric in enumerate(metrics):
        ax = axes_list[idx]
        bar_metric(
            rows,
            metric=metric,
            group="approach",
            title=str(metric),
            ax=ax,
            show=False,
        )

    for ax in axes_list[len(metrics) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    if savepath:
        fig.savefig(savepath, dpi=CONFIG["save_dpi"], bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def _intro_approach_map() -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    return {
        "DA_EXIT": _is_da_exit,
        "EXIT": _is_exit,
        "dense": lambda r: _is_rag_variant(r) and r.get("retriever") == "dense",
        "sparse": lambda r: _is_rag_variant(r) and r.get("retriever") == "sparse",
        "hybrid": lambda r: _is_rag_variant(r) and r.get("retriever") == "hybrid",
    }


def _is_da_exit(row: Dict[str, Any]) -> bool:
    return _is_da_exit_variant(row) and _is_sentence_mode(
        row, CONFIG["da_exit_sentence_mode"]
    )


def _is_exit(row: Dict[str, Any]) -> bool:
    return _is_da_exit_variant(row) and _is_sentence_mode(
        row, CONFIG["exit_sentence_mode"]
    )


def _is_da_exit_variant(row: Dict[str, Any]) -> bool:
    return str(row.get("variant", "")).startswith(CONFIG["da_exit_variant_prefix"])


def _is_exit_variant(row: Dict[str, Any]) -> bool:
    return str(row.get("variant", "")).startswith(CONFIG["exit_variant_prefix"])


def _is_rag_variant(row: Dict[str, Any]) -> bool:
    variant = str(row.get("variant", ""))
    if _is_da_exit_variant(row) or _is_exit_variant(row):
        return False
    if any(variant.startswith(p) for p in CONFIG["rag_variant_prefixes"]):
        return True
    retriever = row.get("retriever")
    return retriever in CONFIG["compare_retrievers"]


def _is_sentence_mode(row: Dict[str, Any], mode: str) -> bool:
    return str(row.get("sentence_mode", "")) == mode


def _filter_da_exit_exit(
    rows: Iterable[Dict[str, Any]],
    *,
    retriever: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not (_is_da_exit_variant(row) or _is_exit_variant(row)):
            continue
        if row.get("retriever") != retriever:
            continue
        if row.get("sentence_mode") not in (
            CONFIG["da_exit_sentence_mode"],
            CONFIG["exit_sentence_mode"],
        ):
            continue
        out.append(row)
    return out


def _label_da_exit_exit(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
    for row in rows:
        mode = row.get("sentence_mode")
        approach = "DA_EXIT" if mode == CONFIG["da_exit_sentence_mode"] else "EXIT"
        item = dict(row)
        item["approach"] = approach
        labeled.append(item)
    return labeled


def _aggregate_by_approach(
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
        xs = _extract_numeric(items, x_key)
        ys = _extract_numeric(items, y_key)
        if not xs or not ys:
            continue
        points.append(
            {
                "approach": label,
                x_key: _aggregate(xs, agg),
                y_key: _aggregate(ys, agg),
            }
        )
    return points


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


def _resolve_datasets(base_dir: str, datasets: Sequence[str] | str | None) -> List[str]:
    if datasets is None:
        datasets = "all"
    if isinstance(datasets, str):
        if datasets.lower() in ("all", "*"):
            root = Path(base_dir)
            return sorted(p.name for p in root.iterdir() if p.is_dir())
        return [datasets]
    return list(datasets)


def _plot_all_datasets() -> None:
    base_dir = CONFIG["base_dir"]
    datasets = _resolve_datasets(base_dir, CONFIG.get("datasets"))
    split = CONFIG["split"]
    save_dir = Path(CONFIG["save_dir"])
    ext = CONFIG.get("save_ext", "png")

    for dataset in datasets:
        rows = load_rows(base_dir=base_dir, dataset=dataset, split=split)
        out_dir = save_dir / dataset / split
        out_dir.mkdir(parents=True, exist_ok=True)

        intro_scatter(rows, savepath=str(out_dir / f"intro_scatter.{ext}"))
        intro_cost_scatter(rows, savepath=str(out_dir / f"intro_cost_scatter.{ext}"))

        for retriever in CONFIG["compare_retrievers"]:
            da_exit_vs_exit_scatter(
                rows,
                retriever=retriever,
                savepath=str(out_dir / f"da_exit_vs_exit_scatter_{retriever}.{ext}"),
            )
            da_exit_vs_exit_cost_scatter(
                rows,
                retriever=retriever,
                savepath=str(out_dir / f"da_exit_vs_exit_cost_scatter_{retriever}.{ext}"),
            )
            da_exit_vs_exit_metric_grid(
                rows,
                retriever=retriever,
                metrics=CONFIG["compare_metrics"],
                savepath=str(out_dir / f"da_exit_vs_exit_metric_grid_{retriever}.{ext}"),
            )
        plt.close("all")


def main() -> None:
    _plot_all_datasets()


if __name__ == "__main__":
    main()
