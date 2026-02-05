"""Orchestrate DA_EXIT/EXIT and RAG summary plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

from .comparison_plots import compare_bar, compare_metric_grid, compare_scatter
from .data import load_summary_metrics
from .overview_plots import approach_pareto, approach_scatter
from .plot_utils import resolve_datasets
from .summary_plots import compare_delta_bar, metric_heatmap

CONFIG = {
    "base_dir": "data/results/Qwen/Qwen2.5-7B-Instruct",
    "datasets": "all",
    "split": "train",
    "save_dir": "data/results/plots",
    "agg": "mean",
    "scatter_x": "wall_time_sec_mean",
    "scatter_y": "mean_f1",
    "cost_x": "tokens_per_query_mean",
    "reader_cost_x": "reader_prompt_tokens_per_query_mean",
    "intro_title": None,
    "compare_title": None,
    "da_exit_variant_prefix": "da_exit_",
    "exit_variant_prefix": "exit_",
    "rag_variant_prefixes": ("rag_", "baseline_rag_", "baseline_embeddings_rag_"),
    "compare_retriever": "hybrid",
    "compare_retrievers": ("hybrid",),
    "intro_retrievers": ("dense", "sparse", "hybrid"),
    "compare_top_k_chunks": 10,
    "compare_metrics": (
        "mean_em",
        "mean_f1",
        "mean_hits_at_k_ratio",
        "mean_recall_at_k_ratio",
        "wall_time_sec_mean",
        "tokens_per_query_mean",
        "reader_prompt_tokens_per_query_mean",
    ),
    "heatmap_metrics": (
        "mean_em",
        "mean_f1",
        "mean_hits_at_k_ratio",
        "mean_recall_at_k_ratio",
        "wall_time_sec_mean",
        "tokens_per_query_mean",
        "reader_prompt_tokens_per_query_mean",
    ),
    "lower_is_better_metrics": (
        "wall_time_sec_mean",
        "tokens_per_query_mean",
        "reader_prompt_tokens_per_query_mean",
    ),
    "approach_order": ("DA_EXIT", "EXIT", "dense", "sparse", "hybrid"),
    "extractor_metrics": (
        "mean_hits_at_k_ratio",
        "mean_recall_at_k_ratio",
    ),
    "delta_figsize": (9, 4),
    "compare_grid_figsize": (7, 4),
    "heatmap_figsize": (10, 4),
    "heatmap_cmap": "viridis",
    "heatmap_annotate": True,
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
        dataset=dataset,
        split=split or CONFIG["split"],
    )


def intro_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing approaches (DA_EXIT, EXIT, dense, sparse, hybrid)."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_scatter(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
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
    """Pareto front for the approaches (aggregated)."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_pareto(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
        agg=CONFIG["agg"],
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
    """Scatter plot comparing total cost vs accuracy."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_scatter(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
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
    return approach_pareto(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
        agg=CONFIG["agg"],
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_reader_cost_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    y: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot comparing reader-only cost vs accuracy."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_scatter(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["reader_cost_x"],
        y=y or CONFIG["scatter_y"],
        agg=CONFIG["agg"],
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_metric_heatmap(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    metrics: Sequence[str] | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Heatmap of approaches vs metrics (normalized)."""
    rows = list(rows) if rows is not None else load_rows()
    return metric_heatmap(
        rows,
        approach_map=_intro_approach_map(),
        metrics=list(metrics or CONFIG["heatmap_metrics"]),
        approach_order=CONFIG["approach_order"],
        agg=CONFIG["agg"],
        figsize=CONFIG["heatmap_figsize"],
        cmap=CONFIG["heatmap_cmap"],
        annotate=CONFIG["heatmap_annotate"],
        title=title,
        savepath=savepath,
        dpi=CONFIG["save_dpi"],
        show=show,
    )


def intro_extractor_latency_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    metric: str = "mean_hits_at_k_ratio",
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot of latency vs a quality metric."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_scatter(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["scatter_x"],
        y=metric,
        agg=CONFIG["agg"],
        title=title or CONFIG["intro_title"],
        savepath=savepath,
        show=show,
    )


def intro_extractor_cost_scatter(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    metric: str = "mean_hits_at_k_ratio",
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Scatter plot of total cost vs a quality metric."""
    rows = list(rows) if rows is not None else load_rows()
    return approach_scatter(
        rows,
        approach_map=_intro_approach_map(),
        x=CONFIG["cost_x"],
        y=metric,
        agg=CONFIG["agg"],
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
    return compare_scatter(
        rows,
        approach_map=_da_exit_exit_map(retriever),
        x=CONFIG["scatter_x"],
        y=CONFIG["scatter_y"],
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
    return compare_scatter(
        rows,
        approach_map=_da_exit_exit_map(retriever),
        x=CONFIG["cost_x"],
        y=y or CONFIG["scatter_y"],
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
    return compare_bar(
        rows,
        approach_map=_da_exit_exit_map(retriever),
        metric=metric,
        agg=CONFIG["agg"],
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
    return compare_metric_grid(
        rows,
        approach_map=_da_exit_exit_map(retriever),
        metrics=list(metrics or CONFIG["compare_metrics"]),
        agg=CONFIG["agg"],
        ncols=ncols,
        figsize=CONFIG["compare_grid_figsize"],
        title=title,
        savepath=savepath,
        dpi=CONFIG["save_dpi"],
        show=show,
    )


def da_exit_vs_exit_delta_bar(
    rows: Iterable[Dict[str, Any]] | None = None,
    *,
    retriever: str | None = None,
    metrics: Sequence[str] | None = None,
    title: str | None = None,
    savepath: str | None = None,
    show: bool = False,
):
    """Bar chart of percent change (DA_EXIT vs EXIT) for multiple metrics."""
    rows = list(rows) if rows is not None else load_rows()
    retriever = retriever or CONFIG["compare_retriever"]
    return compare_delta_bar(
        rows,
        approach_map=_da_exit_exit_map(retriever),
        group_a="DA_EXIT",
        group_b="EXIT",
        metrics=list(metrics or CONFIG["compare_metrics"]),
        agg=CONFIG["agg"],
        lower_is_better_metrics=CONFIG["lower_is_better_metrics"],
        figsize=CONFIG["delta_figsize"],
        title=title,
        savepath=savepath,
        dpi=CONFIG["save_dpi"],
        show=show,
    )


def _plot_all_datasets() -> None:
    base_dir = CONFIG["base_dir"]
    datasets = resolve_datasets(base_dir, CONFIG.get("datasets"))
    split = CONFIG["split"]
    save_dir = Path(CONFIG["save_dir"])
    ext = CONFIG.get("save_ext", "png")

    for dataset in datasets:
        rows = load_rows(base_dir=base_dir, dataset=dataset, split=split)
        out_dir = save_dir / dataset / split
        out_dir.mkdir(parents=True, exist_ok=True)
        base_title = f"{dataset}/{split}"

        intro_scatter(
            rows,
            title=f"Overview (Latency vs F1) — {base_title}",
            savepath=str(out_dir / f"intro_scatter.{ext}"),
        )
        intro_cost_scatter(
            rows,
            title=f"Overview (Total Cost vs F1) — {base_title}",
            savepath=str(out_dir / f"intro_cost_scatter.{ext}"),
        )
        intro_reader_cost_scatter(
            rows,
            title=f"Overview (Reader Cost vs F1) — {base_title}",
            savepath=str(out_dir / f"intro_reader_cost_scatter.{ext}"),
        )
        intro_metric_heatmap(
            rows,
            title=f"Overview (Metric Heatmap) — {base_title}",
            savepath=str(out_dir / f"intro_metric_heatmap.{ext}"),
        )

        for metric in CONFIG.get("extractor_metrics", ()):
            intro_extractor_latency_scatter(
                rows,
                metric=metric,
                title=f"Extractor Quality vs Latency ({metric}) — {base_title}",
                savepath=str(out_dir / f"intro_extractor_latency_{metric}.{ext}"),
            )
            intro_extractor_cost_scatter(
                rows,
                metric=metric,
                title=f"Extractor Quality vs Cost ({metric}) — {base_title}",
                savepath=str(out_dir / f"intro_extractor_cost_{metric}.{ext}"),
            )

        for retriever in CONFIG["compare_retrievers"]:
            da_exit_vs_exit_scatter(
                rows,
                retriever=retriever,
                title=f"DA_EXIT vs EXIT (Latency vs F1, {retriever}) — {base_title}",
                savepath=str(out_dir / f"da_exit_vs_exit_scatter_{retriever}.{ext}"),
            )
            da_exit_vs_exit_cost_scatter(
                rows,
                retriever=retriever,
                title=f"DA_EXIT vs EXIT (Cost vs F1, {retriever}) — {base_title}",
                savepath=str(out_dir / f"da_exit_vs_exit_cost_scatter_{retriever}.{ext}"),
            )
            da_exit_vs_exit_metric_grid(
                rows,
                retriever=retriever,
                metrics=CONFIG["compare_metrics"],
                title=f"DA_EXIT vs EXIT Metrics ({retriever}) — {base_title}",
                savepath=str(out_dir / f"da_exit_vs_exit_metric_grid_{retriever}.{ext}"),
            )
            da_exit_vs_exit_delta_bar(
                rows,
                retriever=retriever,
                metrics=CONFIG["compare_metrics"],
                title=f"DA_EXIT vs EXIT Δ (%) ({retriever}) — {base_title}",
                savepath=str(out_dir / f"da_exit_vs_exit_delta_bar_{retriever}.{ext}"),
            )
        plt.close("all")


def _intro_approach_map() -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    return {
        "DA_EXIT": _is_da_exit,
        "EXIT": _is_exit,
        "dense": lambda r: _is_rag_variant(r) and r.get("retriever") == "dense",
        "sparse": lambda r: _is_rag_variant(r) and r.get("retriever") == "sparse",
        "hybrid": lambda r: _is_rag_variant(r) and r.get("retriever") == "hybrid",
    }


def _da_exit_exit_map(retriever: str) -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    return {
        "DA_EXIT": lambda r: _is_da_exit(r) and r.get("retriever") == retriever,
        "EXIT": lambda r: _is_exit(r) and r.get("retriever") == retriever,
    }


def _is_da_exit(row: Dict[str, Any]) -> bool:
    return _is_da_exit_variant(row) and _matches_top_k(row)


def _is_exit(row: Dict[str, Any]) -> bool:
    return _is_exit_variant(row) and _matches_top_k(row)


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
    return retriever in CONFIG.get("intro_retrievers", CONFIG["compare_retrievers"])


def _matches_top_k(row: Dict[str, Any]) -> bool:
    expected = CONFIG.get("compare_top_k_chunks")
    if expected is None:
        return True
    value = row.get("top_k_chunks")
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def main() -> None:
    _plot_all_datasets()


if __name__ == "__main__":
    main()
