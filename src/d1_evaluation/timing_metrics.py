"""Timing and throughput helpers."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, TypeVar

import numpy as np

__all__ = [
    "wall_time",
    "summarize_reader_wall_times",
    "compute_throughput_stats",
]

T = TypeVar("T")


def wall_time(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """Run ``func`` and return ``(result, elapsed_sec)``."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, float(elapsed)


def summarize_reader_wall_times(
    reader_wall_times: List[float],
    *,
    n_reader_calls: float = 0,
) -> Dict[str, float]:
    """Summarize reader wall times and derive latency stats."""
    total_sec = sum(reader_wall_times)
    mean_sec = total_sec / len(reader_wall_times) if reader_wall_times else 0.0
    median_sec = float(np.median(reader_wall_times)) if reader_wall_times else 0.0
    t_reader_ms = total_sec * 1000
    query_latency_ms = mean_sec * 1000
    call_latency_ms = t_reader_ms / max(n_reader_calls, 1)

    return {
        "reader_wall_time_total_sec": total_sec,
        "reader_wall_time_mean_sec": mean_sec,
        "reader_wall_time_median_sec": median_sec,
        "t_reader_ms": t_reader_ms,
        "query_latency_ms": query_latency_ms,
        "call_latency_ms": call_latency_ms,
    }


def compute_throughput_stats(
    tokens_total: float,
    t_total_ms: float,
    num_queries: float,
    n_reader_calls: float,
    *,
    t_reader_ms: float | None = None,
) -> Dict[str, float]:
    """Compute throughput metrics from totals and wall times."""
    t_total_ms = float(t_total_ms or 0.0)
    t_reader_ms = float(t_reader_ms if t_reader_ms is not None else t_total_ms)
    tokens_total = float(tokens_total or 0.0)
    num_queries = float(num_queries or 0.0)
    n_reader_calls = float(n_reader_calls or 0.0)

    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    query_qps_reader = num_queries / (t_reader_ms / 1000) if t_reader_ms else 0.0
    cps_reader = n_reader_calls / (t_reader_ms / 1000) if t_reader_ms else 0.0

    return {
        "tps_overall": tps_overall,
        "query_qps_reader": query_qps_reader,
        "cps_reader": cps_reader,
    }
