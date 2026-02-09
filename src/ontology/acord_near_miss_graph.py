"""ACORD near-miss graph utilities for category siblings and confusions.

This is a lightweight concept graph (sometimes called an ontology here),
not a formal semantic-web ontology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from src.a1_ingestion.dataset_preprocessing_functions import load_acord_queries
from src.utils.__utils__ import raw_dataset_dir

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - surfaced to caller
    yaml = None

DEFAULT_ACORD_RAW_DIR = raw_dataset_dir("ACORD")
DEFAULT_NEAR_MISS_PATH = DEFAULT_ACORD_RAW_DIR / "near_miss.yaml"

__all__ = [
    "DEFAULT_ACORD_RAW_DIR",
    "DEFAULT_NEAR_MISS_PATH",
    "build_category_index",
    "build_near_miss_map",
    "build_sibling_map",
    "load_acord_ontology",
    "load_near_miss_pairs",
    "write_acord_ontology_summary",
]


def _coerce_pair(left: Any, right: Any) -> Tuple[str, str] | None:
    a = str(left).strip() if left is not None else ""
    b = str(right).strip() if right is not None else ""
    if not a or not b or a == b:
        return None
    return a, b


def _pairs_from_list(items: Iterable[Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pair = _coerce_pair(item[0], item[1])
            if pair:
                pairs.append(pair)
            continue
        if isinstance(item, dict):
            keys = list(item.keys())
            if {"q1", "q2"}.issubset(keys):
                pair = _coerce_pair(item.get("q1"), item.get("q2"))
            elif {"a", "b"}.issubset(keys):
                pair = _coerce_pair(item.get("a"), item.get("b"))
            elif len(keys) == 2:
                pair = _coerce_pair(item.get(keys[0]), item.get(keys[1]))
            else:
                pair = None
            if pair:
                pairs.append(pair)
    return pairs


def load_near_miss_pairs(path: str | Path) -> List[Tuple[str, str]]:
    """Load near-miss pairs from a YAML file.

    Supported YAML formats:
      1) Mapping form:
         Query A: [Query B, Query C]
         Query D: Query E
      2) List form:
         - [Query A, Query B]
         - {q1: Query C, q2: Query D}
      3) Nested under "pairs":
         pairs:
           - [Query A, Query B]
    """
    if yaml is None:
        raise ModuleNotFoundError(
            "pyyaml is required to read near_miss.yaml. "
            "Install it via `pip install pyyaml`."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"near_miss.yaml not found at {path}")

    with open(path, "rt", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []

    pairs: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        if "pairs" in data and isinstance(data["pairs"], list):
            pairs.extend(_pairs_from_list(data["pairs"]))
        else:
            for key, value in data.items():
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        pair = _coerce_pair(key, item)
                        if pair:
                            pairs.append(pair)
                else:
                    pair = _coerce_pair(key, value)
                    if pair:
                        pairs.append(pair)
    elif isinstance(data, list):
        pairs.extend(_pairs_from_list(data))

    # Deduplicate while preserving order
    seen: set[Tuple[str, str]] = set()
    deduped: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def build_near_miss_map(pairs: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Return a bidirectional mapping of near-miss query ids."""
    mapping: Dict[str, Set[str]] = {}
    for a, b in pairs:
        if a == b:
            continue
        mapping.setdefault(a, set()).add(b)
        mapping.setdefault(b, set()).add(a)
    return mapping


def build_category_index(queries: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return category -> query_id list."""
    categories: Dict[str, List[str]] = {}
    for qid, q in queries.items():
        category = str(q.get("category", "")).strip() or "uncategorized"
        categories.setdefault(category, []).append(qid)
    return categories


def build_sibling_map(queries: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Return query_id -> other query_ids in the same category."""
    categories = build_category_index(queries)
    siblings: Dict[str, Set[str]] = {}
    for qids in categories.values():
        for qid in qids:
            siblings[qid] = {other for other in qids if other != qid}
    return siblings


def _filter_unknown(
    mapping: Dict[str, Set[str]], known: Set[str]
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    unknown: set[str] = set()
    filtered: Dict[str, Set[str]] = {}
    for qid, targets in mapping.items():
        if qid not in known:
            unknown.add(qid)
            continue
        kept = {t for t in targets if t in known}
        unknown.update({t for t in targets if t not in known})
        if kept:
            filtered[qid] = kept
    return filtered, unknown


def load_acord_ontology(
    *,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    near_miss_path: str | Path | None = None,
    require_near_miss: bool = True,
) -> Dict[str, Any]:
    """Load ACORD queries with category siblings and near-miss mappings."""
    raw_dir = Path(raw_dir)
    queries_path = raw_dir / "queries.jsonl"
    queries = load_acord_queries(queries_path)

    siblings = build_sibling_map(queries)

    if near_miss_path is None:
        near_miss_path = DEFAULT_NEAR_MISS_PATH
    near_miss_path = Path(near_miss_path)

    near_miss_pairs: List[Tuple[str, str]] = []
    near_miss_map: Dict[str, Set[str]] = {}
    unknown_near_miss: Set[str] = set()

    if near_miss_path.exists():
        near_miss_pairs = load_near_miss_pairs(near_miss_path)
        near_miss_map = build_near_miss_map(near_miss_pairs)
        near_miss_map, unknown_near_miss = _filter_unknown(
            near_miss_map, set(queries.keys())
        )
    elif require_near_miss:
        raise FileNotFoundError(
            f"Missing near_miss.yaml at {near_miss_path}. "
            "Create it to define near-miss graph pairs."
        )

    return {
        "queries": queries,
        "siblings": siblings,
        "near_miss_pairs": near_miss_pairs,
        "near_miss_map": near_miss_map,
        "unknown_near_miss": sorted(unknown_near_miss),
        "near_miss_path": str(near_miss_path),
    }


def write_acord_ontology_summary(
    *,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    near_miss_path: str | Path | None = None,
    out_path: str | Path,
    require_near_miss: bool = True,
) -> Dict[str, Any]:
    """Write a JSON summary of the ACORD near-miss graph configuration."""
    data = load_acord_ontology(
        raw_dir=raw_dir,
        near_miss_path=near_miss_path,
        require_near_miss=require_near_miss,
    )

    queries = data["queries"]
    categories = build_category_index(queries)
    summary = {
        "queries_total": len(queries),
        "categories_total": len(categories),
        "categories": {k: len(v) for k, v in categories.items()},
        "near_miss_pairs_total": len(data.get("near_miss_pairs", [])),
        "near_miss_path": data.get("near_miss_path"),
        "unknown_near_miss": data.get("unknown_near_miss", []),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)

    return summary
