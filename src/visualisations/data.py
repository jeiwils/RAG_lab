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
        if "dataset" not in row:
            row["dataset"] = _infer_dataset(path)
        if "split" not in row:
            row["split"] = _infer_split(path)
        row["summary_path"] = str(path)

        variant = row.get("variant", "")
        if variant:
            parsed = parse_variant_name(variant)
            for key, value in parsed.items():
                if key not in row:
                    row[key] = value
        if "approach" not in row:
            row["approach"] = assign_approach(row)

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
