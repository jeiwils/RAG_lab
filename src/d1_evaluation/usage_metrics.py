"""Token usage aggregation helpers for traversal and reader stages."""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

__all__ = ["merge_token_usage", "load_token_usage"]


def _merge_numeric(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge numeric values from ``src`` into ``dst``."""
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v
        else:
            dst[k] = v
    return dst


def merge_token_usage(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    cleanup: bool = False,
) -> Path:
    """Merge ``token_usage`` shards in ``output_dir`` into one.

    The function aggregates global token counts and per-query metrics across
    multiple partial usage files. If no usage files are found, an empty
    ``token_usage.json`` file is created. The merged result is written to
    ``token_usage.json`` inside ``output_dir``.
    Parameters
    ----------
    output_dir:
        Directory containing ``token_usage_*.json`` shard files.
    run_id:
        Optional identifier used in shard filenames. When provided, only files
        matching ``token_usage_{run_id}_*.json`` are merged. Any non-matching
        shards are ignored and, if ``cleanup`` is ``True``, removed.
    cleanup:
        If ``True``, the individual shard files are removed after the merged
        ``token_usage.json`` is written. Defaults to ``False`` to preserve
        backward compatibility with callers expecting shards to remain.
    """

    out_dir = Path(output_dir)
    if run_id is not None:
        stale = [fp for fp in out_dir.glob("token_usage_*.json") if run_id not in fp.name]
        for fp in stale:
            try:
                fp.unlink()
            except OSError:
                pass
        pattern = f"token_usage_{run_id}_*.json"
    else:
        pattern = "token_usage_*.json"
    usage_files = sorted(out_dir.glob(pattern))
    if not usage_files:
        out_path = out_dir / "token_usage.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return out_path

    per_query_trav: Dict[str, Dict[str, Any]] = {}
    per_query_reader: Dict[str, Dict[str, Any]] = {}
    global_totals: Dict[str, Any] = defaultdict(float)

    for fp in usage_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if pq := data.get("per_query_traversal"):
            for qid, metrics in pq.items():
                per_query_trav[qid] = _merge_numeric(per_query_trav.get(qid, {}), metrics)
        if pq := data.get("per_query_reader"):
            for qid, metrics in pq.items():
                per_query_reader[qid] = _merge_numeric(per_query_reader.get(qid, {}), metrics)

        for k, v in data.items():
            if k.startswith("per_query") or k in {
                "tokens_total",
                "t_total_ms",
                "tps_overall",
                "query_latency_ms",
                "call_latency_ms",
                "query_qps_traversal",
                "cps_traversal",
                "query_qps_reader",
                "cps_reader",
            }:
                continue
            if isinstance(v, (int, float)):
                global_totals[k] += v
            else:
                existing = global_totals.get(k)
                if existing is None:
                    global_totals[k] = v
                elif existing != v:
                    warnings.warn(
                        f"Conflicting values for '{k}': keeping {existing!r}, ignoring {v!r}",
                        stacklevel=1,
                    )

    tokens_total = (
        global_totals.get("trav_tokens_total", 0)
        + global_totals.get("reader_total_tokens", 0)
    )
    t_total_ms = global_totals.get("t_traversal_ms", 0) + global_totals.get(
        "t_reader_ms", 0
    )

    merged: Dict[str, Any] = {}
    if per_query_trav:
        merged["per_query_traversal"] = per_query_trav
    if per_query_reader:
        merged["per_query_reader"] = per_query_reader
    merged.update(global_totals)
    merged["tokens_total"] = tokens_total
    merged["t_total_ms"] = t_total_ms
    merged["tps_overall"] = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    num_queries = merged.get("num_queries", 0)
    t_trav_ms = global_totals.get("t_traversal_ms", 0)
    merged["query_qps_traversal"] = (
        num_queries / (t_trav_ms / 1000) if t_trav_ms else 0.0
    )
    merged["cps_traversal"] = (
        merged.get("n_traversal_calls", 0) / (t_trav_ms / 1000)
        if t_trav_ms
        else 0.0
    )
    t_reader_ms = global_totals.get("t_reader_ms", 0)
    merged["query_qps_reader"] = (
        num_queries / (t_reader_ms / 1000) if t_reader_ms else 0.0
    )
    merged["cps_reader"] = (
        merged.get("n_reader_calls", 0) / (t_reader_ms / 1000) if t_reader_ms else 0.0
    )
    merged["query_latency_ms"] = t_total_ms / num_queries if num_queries else 0.0

    n_trav_calls = merged.get("n_traversal_calls", 0)
    n_reader_calls = merged.get("n_reader_calls", 0)
    merged["call_latency_ms_traversal"] = t_trav_ms / max(n_trav_calls, 1)
    merged["call_latency_ms_reader"] = t_reader_ms / max(n_reader_calls, 1)
    merged["call_latency_ms"] = t_total_ms / max(n_trav_calls + n_reader_calls, 1)

    out_path = out_dir / "token_usage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    if cleanup:
        for fp in usage_files:
            try:
                fp.unlink()
            except OSError:
                pass

    return out_path


def load_token_usage(token_usage_path: str | Path) -> dict:
    """Load token usage statistics from ``token_usage.json``."""
    token_usage_path = Path(token_usage_path)
    try:
        with open(token_usage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"per_query": {}, "global": {}}

    per_trav = data.get("per_query_traversal") or {}
    per_reader = data.get("per_query_reader") or {}

    merged_per_query: dict = {}
    for qid in set(per_trav) | set(per_reader):
        t = per_trav.get(qid, {})
        r = per_reader.get(qid, {})

        n_trav = float(t.get("n_traversal_calls", 0))
        n_reader = float(r.get("n_reader_calls", 0))
        t_trav_ms = float(t.get("t_traversal_ms", 0))
        t_reader_ms = float(r.get("t_reader_ms", 0))
        tok_trav = float(t.get("trav_tokens_total", 0))
        tok_reader = float(r.get("reader_total_tokens", 0))
        tokens_total = tok_trav + tok_reader
        t_total_ms = t_trav_ms + t_reader_ms
        tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0

        call_lat_trav = float(
            t.get("call_latency_ms", t_trav_ms / max(n_trav, 1))
        )
        call_lat_reader = float(
            r.get("call_latency_ms", t_reader_ms / max(n_reader, 1))
        )
        call_lat = (t_trav_ms + t_reader_ms) / max(n_trav + n_reader, 1)

        merged = {}
        merged.update(t)
        merged.update(r)
        merged.update(
            {
                "tokens_total": tokens_total,
                "t_total_ms": t_total_ms,
                "tps_overall": tps_overall,
                "n_traversal_calls": n_trav,
                "n_reader_calls": n_reader,
                "t_traversal_ms": t_trav_ms,
                "t_reader_ms": t_reader_ms,
                "call_latency_ms_traversal": call_lat_trav,
                "call_latency_ms_reader": call_lat_reader,
                "call_latency_ms": call_lat,
            }
        )
        merged_per_query[qid] = merged

    global_usage = {k: v for k, v in data.items() if not k.startswith("per_query")}
    t_trav_ms = float(global_usage.get("t_traversal_ms", 0))
    t_reader_ms = float(global_usage.get("t_reader_ms", 0))
    tokens_total = float(
        global_usage.get(
            "tokens_total",
            global_usage.get("trav_tokens_total", 0)
            + global_usage.get("reader_total_tokens", 0),
        )
    )
    t_total_ms = float(global_usage.get("t_total_ms", t_trav_ms + t_reader_ms))
    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    n_trav = float(global_usage.get("n_traversal_calls", 0))
    n_reader = float(global_usage.get("n_reader_calls", 0))
    num_queries = float(global_usage.get("num_queries", 0))
    query_latency_ms = global_usage.get("query_latency_ms")
    if query_latency_ms is None:
        query_latency_ms = t_total_ms / num_queries if num_queries else 0.0
    call_lat_trav = global_usage.get("call_latency_ms_traversal")
    if call_lat_trav is None:
        call_lat_trav = t_trav_ms / max(n_trav, 1)
    call_lat_reader = global_usage.get("call_latency_ms_reader")
    if call_lat_reader is None:
        call_lat_reader = t_reader_ms / max(n_reader, 1)
    call_latency_ms = global_usage.get("call_latency_ms")
    if call_latency_ms is None:
        call_latency_ms = (t_trav_ms + t_reader_ms) / max(n_trav + n_reader, 1)

    global_usage.update(
        {
            "tokens_total": tokens_total,
            "t_total_ms": t_total_ms,
            "tps_overall": tps_overall,
            "query_latency_ms": query_latency_ms,
            "call_latency_ms_traversal": call_lat_trav,
            "call_latency_ms_reader": call_lat_reader,
            "call_latency_ms": call_latency_ms,
        }
    )

    return {"per_query": merged_per_query, "global": global_usage}
