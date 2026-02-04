"""Load and parse result summaries for plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

RETRIEVERS = {"dense", "sparse", "hybrid"}

CONFIG = {
    "base_dir": "data/results/Qwen/Qwen2.5-7B-Instruct",
    "dataset": None,
    "split": None,
}


def load_summary_metrics(
    base_dir: str | Path | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    """Load summary_metrics_*.json files under base_dir.

    If dataset/split are provided, only those subfolders are scanned.
    """
    if base_dir is None:
        base_dir = CONFIG["base_dir"]
    if dataset is None:
        dataset = CONFIG["dataset"]
    if split is None:
        split = CONFIG["split"]

    root = Path(base_dir)
    if dataset:
        root = root / dataset
    if split:
        root = root / split

    rows: List[Dict[str, Any]] = []
    for path in root.rglob("summary_metrics_*.json"):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        row = dict(obj)
        _flatten_schema(row)
        _merge_eval_sections(row)
        _normalize_summary_fields(row)
        _ensure_queries_total(row)
        if "dataset" not in row:
            row["dataset"] = _infer_dataset(path)
        if "split" not in row:
            row["split"] = _infer_split(path)
        row["summary_path"] = str(path)

        variant = row.get("variant", "")
        if variant:
            row.update(parse_variant_name(variant))
        if "approach" not in row:
            row["approach"] = assign_approach(row)

        _derive_tokens_per_query_mean(row)

        rows.append(row)
    return rows


def parse_variant_name(variant: str) -> Dict[str, Any]:
    """Parse variant naming conventions into structured fields."""
    info: Dict[str, Any] = {}
    rest = variant
    if variant.startswith("da_exit_"):
        info["method"] = "DA_EXIT"
        rest = variant[len("da_exit_") :]
    elif variant.startswith("exit_"):
        info["method"] = "EXIT"
        rest = variant[len("exit_") :]

    tokens = [t for t in rest.split("_") if t]
    retriever = next((t for t in tokens if t in RETRIEVERS), None)
    if retriever:
        info["retriever"] = retriever
        idx = tokens.index(retriever)
        pre = tokens[:idx]
        post = tokens[idx + 1 :]
    else:
        pre = tokens
        post = []

    # Parse k/s parameters if present.
    for t in post:
        if t.startswith("k") and t[1:].isdigit():
            info["top_k_chunks"] = int(t[1:])
        if t.startswith("s") and t[1:].isdigit():
            info["top_k_sentences"] = int(t[1:])

    # Parse chunking/sentence modes.
    if "chunks" in pre:
        idx_chunks = pre.index("chunks")
        info["chunking_mode"] = "_".join(pre[: idx_chunks + 1])
        remaining = pre[idx_chunks + 1 :]
        if "sentences" in remaining:
            idx_sent = remaining.index("sentences")
            info["sentence_mode"] = "_".join(remaining[: idx_sent + 1])
    else:
        if pre:
            if pre[:2] == ["discourse", "aware"]:
                info["chunking_mode"] = "discourse_aware_chunks"
            elif pre[0] == "standard":
                info["chunking_mode"] = "standard_chunks"
            else:
                info["chunking_mode"] = "_".join(pre)
        info.setdefault("sentence_mode", "standard_sentences")

    info.setdefault("sentence_mode", "standard_sentences")
    return info


def assign_approach(row: Dict[str, Any]) -> str:
    """Assign a high-level approach label for plotting."""
    variant = str(row.get("variant", ""))
    if variant.startswith("da_exit_"):
        return "DA_EXIT"
    if variant.startswith("exit_"):
        return "EXIT"
    retriever = row.get("retriever")
    if retriever in RETRIEVERS:
        return str(retriever)
    return "unknown"


def _flatten_schema(row: Dict[str, Any]) -> None:
    meta = row.get("meta")
    if isinstance(meta, dict):
        row.update(meta)
    for section in ("accuracy", "latency", "cost", "retrieval", "throughput", "artifacts"):
        data = row.get(section)
        if isinstance(data, dict):
            row.update(data)


def _merge_eval_sections(row: Dict[str, Any]) -> None:
    eval_sections = {
        key: val
        for key, val in row.items()
        if key.endswith("_eval") and isinstance(val, dict)
    }
    if not eval_sections:
        return
    retriever = row.get("retriever")
    if retriever:
        preferred = eval_sections.get(f"{retriever}_eval")
        if preferred:
            row.update(preferred)
            return
    if len(eval_sections) == 1:
        row.update(next(iter(eval_sections.values())))


def _normalize_summary_fields(row: Dict[str, Any]) -> None:
    if "mean_em" not in row:
        if "EM" in row:
            row["mean_em"] = row["EM"]
        elif "em" in row:
            row["mean_em"] = row["em"]
    if "mean_f1" not in row:
        if "F1" in row:
            row["mean_f1"] = row["F1"]
        elif "f1" in row:
            row["mean_f1"] = row["f1"]

    if "wall_time_sec_mean" not in row:
        if "wall_time_mean_sec" in row:
            row["wall_time_sec_mean"] = row["wall_time_mean_sec"]
        elif "reader_wall_time_sec_mean" in row:
            row["wall_time_sec_mean"] = row["reader_wall_time_sec_mean"]
        elif "reader_wall_time_mean_sec" in row:
            row["wall_time_sec_mean"] = row["reader_wall_time_mean_sec"]

    if "wall_time_sec_median" not in row:
        if "wall_time_median_sec" in row:
            row["wall_time_sec_median"] = row["wall_time_median_sec"]
        elif "reader_wall_time_sec_median" in row:
            row["wall_time_sec_median"] = row["reader_wall_time_sec_median"]

    if "wall_time_sec_total" not in row:
        if "wall_time_total_sec" in row:
            row["wall_time_sec_total"] = row["wall_time_total_sec"]
        elif "reader_wall_time_total_sec" in row:
            row["wall_time_sec_total"] = row["reader_wall_time_total_sec"]


def _ensure_queries_total(row: Dict[str, Any]) -> None:
    if row.get("queries_total"):
        return
    for key in ("num_queries", "n_queries", "n_reader_calls"):
        val = row.get(key)
        if val:
            row["queries_total"] = val
            return


def _derive_tokens_per_query_mean(row: Dict[str, Any]) -> None:
    if "tokens_per_query_mean" in row:
        return
    tokens_total = row.get("tokens_total")
    if tokens_total is None:
        return
    queries_total = row.get("queries_total")
    if not queries_total:
        return
    try:
        row["tokens_per_query_mean"] = float(tokens_total) / float(queries_total)
    except (TypeError, ValueError, ZeroDivisionError):
        return


def _infer_dataset(path: Path) -> str | None:
    try:
        return path.parents[2].name
    except IndexError:
        return None


def _infer_split(path: Path) -> str | None:
    try:
        return path.parents[1].name
    except IndexError:
        return None
