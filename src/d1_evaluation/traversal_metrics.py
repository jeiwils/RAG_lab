"""Traversal-specific metrics and summaries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "compute_hop_metrics",
    "compute_gold_attention",
    "compute_traversal_summary",
    "append_traversal_percentiles",
]


def compute_hop_metrics(hop_trace, gold_passages):
    """Compute precision, recall, and F1 per hop and final."""

    gold_set = set(gold_passages)
    visited_cumulative = set()
    results = []

    for hop_log in hop_trace:
        visited_cumulative.update(hop_log.get("new_passages", []))
        tp = len(visited_cumulative & gold_set)
        fp = len(visited_cumulative - gold_set)
        fn = len(gold_set - visited_cumulative)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        hop_log["metrics"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        results.append(hop_log)

    final_metrics = (
        results[-1]["metrics"] if results else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )
    return results, final_metrics


def compute_gold_attention(
    ccount: Dict[str, int], gold_passages: List[str]
) -> Tuple[Dict[str, int], float]:
    """Compute visitation stats for gold passages."""

    gold_counts = {pid: ccount.get(pid, 0) for pid in gold_passages}
    total_visits = sum(ccount.values()) or 1
    attention_ratio = sum(gold_counts.values()) / total_visits
    return gold_counts, attention_ratio


def compute_traversal_summary(
    results_path: str,
    include_ids: Optional[Set[str]] = None,
) -> dict:
    """Summarize traversal-wide metrics across all dev results."""

    total_queries = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_hits = 0.0
    sum_recall_at_k = 0.0
    sum_gold_attention = 0.0
    sum_traversal_calls = 0

    total_none = 0
    total_repeat = 0
    passage_coverage_all_gold_found = 0
    initial_retrieval_coverage = 0
    first_gold_hops = []
    query_hop_depths: List[int] = []
    traversal_wall_times: List[float] = []

    with open(results_path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if include_ids is not None and entry["question_id"] not in include_ids:
                continue

            total_queries += 1

            final = entry["final_metrics"]
            sum_precision += final["precision"]
            sum_recall += final["recall"]
            sum_f1 += final.get("f1", 0.0)
            sum_hits += entry.get("hits_at_k", 0.0)
            sum_recall_at_k += entry.get("recall_at_k", 0.0)
            sum_gold_attention += entry.get("gold_attention_ratio", 0.0)
            sum_traversal_calls += entry.get("n_traversal_calls", 0)

            if "traversal_wall_time_sec" in entry:
                traversal_wall_times.append(entry["traversal_wall_time_sec"])

            if set(entry["gold_passages"]).issubset(set(entry["visited_passages"])):
                passage_coverage_all_gold_found += 1

            gold_set = set(entry["gold_passages"])
            hop_trace = entry.get("hop_trace", [])
            if hop_trace:
                hop0_passages = set(hop_trace[0].get("expanded_from", []))
                if hop0_passages & gold_set:
                    initial_retrieval_coverage += 1
                    first_gold_hop = 0
                else:
                    first_gold_hop = None
            else:
                first_gold_hop = None

            deepest_non_empty_hop = -1
            for hop_log in entry["hop_trace"]:
                total_none += hop_log["none_count"]
                total_repeat += hop_log["repeat_visit_count"]

                if hop_log.get("expanded_from") or hop_log.get("new_passages"):
                    deepest_non_empty_hop = hop_log["hop"]

                if first_gold_hop is None and set(hop_log.get("new_passages", [])) & gold_set:
                    first_gold_hop = hop_log["hop"]

            query_hop_depths.append(deepest_non_empty_hop + 1)

            if first_gold_hop is not None:
                first_gold_hops.append(first_gold_hop)

    mean_precision = sum_precision / total_queries if total_queries else 0
    mean_recall = sum_recall / total_queries if total_queries else 0
    mean_f1 = sum_f1 / total_queries if total_queries else 0
    mean_hits = sum_hits / total_queries if total_queries else 0
    mean_recall_at_k = sum_recall_at_k / total_queries if total_queries else 0
    mean_gold_attention = sum_gold_attention / total_queries if total_queries else 0
    mean_traversal_calls = sum_traversal_calls / total_queries if total_queries else 0

    avg_first_gold = (
        round(sum(first_gold_hops) / len(first_gold_hops), 2) if first_gold_hops else None
    )

    hop_depth_counter = Counter(query_hop_depths)
    max_depth = max(query_hop_depths) if query_hop_depths else 0
    hop_depth_distribution = [hop_depth_counter.get(i, 0) for i in range(max_depth + 1)]

    traversal_wall_time_total = sum(traversal_wall_times)
    traversal_wall_time_mean = (
        traversal_wall_time_total / len(traversal_wall_times)
        if traversal_wall_times
        else 0.0
    )
    traversal_wall_time_median = (
        float(np.median(traversal_wall_times)) if traversal_wall_times else 0.0
    )
    query_latency_ms = traversal_wall_time_mean * 1000
    call_latency_ms = (
        traversal_wall_time_total * 1000 / sum_traversal_calls
        if sum_traversal_calls
        else 0.0
    )

    summary = {
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_f1": round(mean_f1, 4),
        "mean_hits_at_k": round(mean_hits, 4),
        "mean_recall_at_k": round(mean_recall_at_k, 4),
        "mean_gold_attention_ratio": round(mean_gold_attention, 4),
        "avg_traversal_calls": round(mean_traversal_calls, 2),
        "total_traversal_calls": sum_traversal_calls,
        "passage_coverage_all_gold_found": passage_coverage_all_gold_found,
        "initial_retrieval_coverage": initial_retrieval_coverage,
        "avg_hops_before_first_gold": avg_first_gold,
        "avg_total_hops": round(sum(query_hop_depths) / total_queries, 2)
        if total_queries
        else 0,
        "avg_repeat_visits": round(total_repeat / total_queries, 2) if total_queries else 0,
        "avg_none_count_per_query": round(total_none / total_queries, 2)
        if total_queries
        else 0,
        "max_hop_depth_reached": max_depth,
        "hop_depth_distribution": hop_depth_distribution,
        "traversal_wall_time_total_sec": round(traversal_wall_time_total, 4),
        "traversal_wall_time_mean_sec": round(traversal_wall_time_mean, 4),
        "traversal_wall_time_median_sec": round(traversal_wall_time_median, 4),
        "query_latency_ms": round(query_latency_ms, 2),
        "call_latency_ms": round(call_latency_ms, 2),
    }

    summary["query_qps_traversal"] = (
        total_queries / summary["traversal_wall_time_total_sec"]
        if summary["traversal_wall_time_total_sec"]
        else 0.0
    )
    summary["cps_traversal"] = (
        summary["total_traversal_calls"] / summary["traversal_wall_time_total_sec"]
        if summary["traversal_wall_time_total_sec"]
        else 0.0
    )

    return summary


def append_traversal_percentiles(
    results_path: str | Path, stats_path: str | Path
) -> Dict[str, float]:
    """Append median and p90 traversal metrics to ``final_traversal_stats.json``."""

    results_path = Path(results_path)
    stats_path = Path(stats_path)

    if not results_path.exists():
        return {}

    f1s: List[float] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fm = obj.get("final_metrics", {})
            f1 = fm.get("f1")
            if f1 is not None:
                f1s.append(float(f1))

    stats: Dict[str, float] = {}
    if f1s:
        stats["median_final_f1"] = float(np.median(f1s))
        stats["p90_final_f1"] = float(np.percentile(f1s, 90))

    token_usage_path = stats_path.parent / "token_usage.json"
    if token_usage_path.exists():
        try:
            with open(token_usage_path, "r", encoding="utf-8") as f:
                usage = json.load(f)
        except json.JSONDecodeError:
            usage = {}

        global_usage = usage.get("global", usage)
        q_latency = global_usage.get("query_latency_ms")
        c_latency = global_usage.get("call_latency_ms")
        if q_latency is not None:
            stats["query_latency_ms"] = float(q_latency)
        if c_latency is not None:
            stats["call_latency_ms"] = float(c_latency)

        per_trav = usage.get("per_query_traversal", {}) or {}

        trav_tokens: List[float] = []
        query_latencies_ms: List[float] = []
        call_latencies_ms: List[float] = []
        tps: List[float] = []
        query_qps: List[float] = []
        cps: List[float] = []
        total_t_ms = 0.0
        total_n_calls = 0.0

        for q in per_trav.values():
            tok = float(q.get("trav_tokens_total", 0))
            t_ms = float(q.get("t_traversal_ms", 0))
            trav_tokens.append(tok)
            query_latencies_ms.append(t_ms)
            tps.append(tok / (t_ms / 1000) if t_ms else 0.0)

            n_calls = float(q.get("n_traversal_calls", 0))
            query_qps.append(1 / (t_ms / 1000) if t_ms else 0.0)
            cps.append(n_calls / (t_ms / 1000) if t_ms else 0.0)
            total_t_ms += t_ms
            total_n_calls += n_calls
            if "call_latency_ms" in q:
                latency = float(q.get("call_latency_ms", 0))
            else:
                latency = t_ms / max(n_calls, 1)
            call_latencies_ms.append(latency)

        num_queries = len(per_trav)
        if total_t_ms:
            stats["overall_qps"] = num_queries / (total_t_ms / 1000)
            stats["overall_cps"] = total_n_calls / (total_t_ms / 1000)
        else:
            stats["overall_qps"] = 0.0
            stats["overall_cps"] = 0.0

        if trav_tokens:
            stats["median_trav_tokens_total"] = float(np.median(trav_tokens))
            stats["p90_trav_tokens_total"] = float(np.percentile(trav_tokens, 90))
        if query_latencies_ms:
            stats["median_latency_ms"] = float(np.median(query_latencies_ms))
            stats["p90_latency_ms"] = float(np.percentile(query_latencies_ms, 90))
        if tps:
            stats["median_tps_overall"] = float(np.median(tps))
            stats["p90_tps_overall"] = float(np.percentile(tps, 90))
        if query_qps:
            stats["median_query_qps_traversal"] = float(np.median(query_qps))
            stats["p90_query_qps_traversal"] = float(np.percentile(query_qps, 90))
        if cps:
            stats["median_cps_traversal"] = float(np.median(cps))
            stats["p90_cps_traversal"] = float(np.percentile(cps, 90))
        if call_latencies_ms:
            stats["median_call_latency_ms"] = float(np.median(call_latencies_ms))
            stats["p90_call_latency_ms"] = float(np.percentile(call_latencies_ms, 90))

    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary.update(stats)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats
