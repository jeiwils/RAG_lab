"""Statistical comparisons and summary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import wilcoxon

from src.d1_evaluation.usage_metrics import load_token_usage

__all__ = ["compare_runs", "load_metrics", "append_percentiles"]


def load_metrics(path: str | Path) -> np.ndarray:
    """Load metric values from a text file into a NumPy array."""
    with open(path, "r", encoding="utf-8") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values, dtype=float)


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict:
    """Compare two runs and return evaluation summary with p-value."""
    metrics_a = load_metrics(run_a)
    metrics_b = load_metrics(run_b)

    if metrics_a.shape != metrics_b.shape:
        raise ValueError("Metric arrays must have the same length for a paired test")

    stat, p_value = wilcoxon(metrics_a, metrics_b)

    summary = {
        "run_a_mean": float(metrics_a.mean()),
        "run_b_mean": float(metrics_b.mean()),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
    }

    print(json.dumps(summary, indent=2))
    return summary


def append_percentiles(metrics_path: str | Path, summary_path: str | Path) -> Dict[str, float]:
    """Append median and p90 metrics to an existing summary file."""
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)

    if not metrics_path.exists():
        return {}

    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return {}

    f1s = [r.get("f1", 0.0) for r in records]
    ems = [r.get("em", 0.0) for r in records]

    stats: Dict[str, float] = {}
    if f1s:
        stats["median_f1"] = float(np.median(f1s))
        stats["p90_f1"] = float(np.percentile(f1s, 90))
    if ems:
        stats["median_em"] = float(np.median(ems))
        stats["p90_em"] = float(np.percentile(ems, 90))

    token_usage_path = summary_path.parent / "token_usage.json"
    if token_usage_path.exists():
        usage = load_token_usage(token_usage_path)
        per_query = usage.get("per_query", {})

        global_usage = usage.get("global", usage)
        q_latency = global_usage.get("query_latency_ms")
        c_latency = global_usage.get("call_latency_ms")
        if q_latency is not None:
            stats["query_latency_ms"] = float(q_latency)
        if c_latency is not None:
            stats["call_latency_ms"] = float(c_latency)

        tokens: list[float] = []
        query_latencies_ms: list[float] = []
        call_latencies_ms: list[float] = []
        tps: list[float] = []
        query_qps_read: list[float] = []
        cps_read: list[float] = []
        total_t_reader_ms = 0.0
        total_n_reader_calls = 0.0

        for metrics in per_query.values():
            tok = float(metrics.get("tokens_total", 0))
            t_ms = float(metrics.get("t_total_ms", 0))
            tokens.append(tok)
            query_latencies_ms.append(t_ms)
            tps.append(
                float(metrics.get("tps_overall", tok / (t_ms / 1000) if t_ms else 0.0))
            )

            n_reader_calls = float(metrics.get("n_reader_calls", 0))
            total_n_reader_calls += n_reader_calls

            t_reader_ms = float(metrics.get("t_reader_ms", 0))
            total_t_reader_ms += t_reader_ms

            latency = float(
                metrics.get(
                    "call_latency_ms", t_reader_ms / max(n_reader_calls, 1)
                )
            )
            call_latencies_ms.append(latency)

            query_qps_read.append(1 / (t_reader_ms / 1000) if t_reader_ms else 0.0)
            cps_read.append(n_reader_calls / (t_reader_ms / 1000) if t_reader_ms else 0.0)

        num_queries = len(per_query)
        if total_t_reader_ms:
            stats["overall_qps"] = num_queries / (total_t_reader_ms / 1000)
            stats["overall_cps"] = total_n_reader_calls / (total_t_reader_ms / 1000)
        else:
            stats["overall_qps"] = 0.0
            stats["overall_cps"] = 0.0

        if tokens:
            stats["median_tokens_total"] = float(np.median(tokens))
            stats["p90_tokens_total"] = float(np.percentile(tokens, 90))
        if query_latencies_ms:
            stats["median_latency_ms"] = float(np.median(query_latencies_ms))
            stats["p90_latency_ms"] = float(np.percentile(query_latencies_ms, 90))
        if call_latencies_ms:
            stats["median_call_latency_ms"] = float(np.median(call_latencies_ms))
            stats["p90_call_latency_ms"] = float(np.percentile(call_latencies_ms, 90))
        if tps:
            stats["median_tps_overall"] = float(np.median(tps))
            stats["p90_tps_overall"] = float(np.percentile(tps, 90))
        if query_qps_read:
            stats["median_query_qps_reader"] = float(np.median(query_qps_read))
            stats["p90_query_qps_reader"] = float(np.percentile(query_qps_read, 90))
        if cps_read:
            stats["median_cps_reader"] = float(np.median(cps_read))
            stats["p90_cps_reader"] = float(np.percentile(cps_read, 90))

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    if "dense_eval" in summary and isinstance(summary["dense_eval"], dict):
        summary["dense_eval"].update(stats)
    else:
        summary.update(stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats

