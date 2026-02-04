"""Plotting helpers for RAG_lab results."""

from __future__ import annotations

from importlib import import_module

from .bars import bar_metric
from .curves import recall_at_k_curve, recall_at_k_values
from .data import assign_approach, load_summary_metrics, parse_variant_name
from .pareto import pareto_front_latency_f1
from .scatter import scatter_latency_accuracy_approaches

_ORCHESTRATOR_EXPORTS = {
    "da_exit_vs_exit_bar",
    "da_exit_vs_exit_cost_scatter",
    "da_exit_vs_exit_metric_grid",
    "da_exit_vs_exit_scatter",
    "intro_cost_pareto",
    "intro_cost_scatter",
    "intro_pareto",
    "intro_scatter",
    "load_rows",
}

__all__ = [
    "assign_approach",
    "load_summary_metrics",
    "parse_variant_name",
    "scatter_latency_accuracy_approaches",
    "bar_metric",
    "pareto_front_latency_f1",
    "recall_at_k_curve",
    "recall_at_k_values",
    "da_exit_vs_exit_bar",
    "da_exit_vs_exit_cost_scatter",
    "da_exit_vs_exit_metric_grid",
    "da_exit_vs_exit_scatter",
    "intro_cost_pareto",
    "intro_cost_scatter",
    "intro_pareto",
    "intro_scatter",
    "load_rows",
]


def __getattr__(name: str):
    if name in _ORCHESTRATOR_EXPORTS:
        module = import_module(".DA_EXIT_visualisations_orchestrator", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
