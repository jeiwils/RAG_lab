"""Automate near-miss pair generation for the ACORD near-miss graph.

This module proposes candidate near-miss query pairs using lexical similarity
and (optionally) an LLM filter. It outputs a YAML file compatible with the
ACORD near-miss graph loader.
"""

from __future__ import annotations

import json
import random
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.a1_ingestion.dataset_preprocessing_functions import load_acord_queries
from src.utils.__utils__ import raw_dataset_dir
from src.utils.z_llm_utils import build_prompt, is_r1_like, query_llm, strip_think

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - surfaced to caller
    yaml = None

DEFAULT_ACORD_RAW_DIR = raw_dataset_dir("ACORD")

__all__ = [
    "DEFAULT_ACORD_RAW_DIR",
    "mine_near_miss_pairs",
    "write_near_miss_yaml",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ScoredPair:
    left: str
    right: str
    score: float
    overlap: float
    jaccard: float
    seq_ratio: float


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _similarity(a: str, b: str) -> ScoredPair:
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return ScoredPair(a, b, 0.0, 0.0, 0.0, 0.0)
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(inter) / max(len(union), 1)
    overlap = len(inter) / max(min(len(tokens_a), len(tokens_b)), 1)
    # Simple character ratio to capture minor edits
    seq_ratio = _sequence_ratio(a, b)
    score = 0.4 * jaccard + 0.3 * overlap + 0.3 * seq_ratio
    return ScoredPair(a, b, score, overlap, jaccard, seq_ratio)


def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _select_top_pairs(
    items: Iterable[ScoredPair],
    *,
    min_score: float,
    max_pairs: int,
) -> List[ScoredPair]:
    filtered = [p for p in items if p.score >= min_score]
    filtered.sort(key=lambda x: x.score, reverse=True)
    return filtered[:max_pairs] if max_pairs > 0 else filtered


def _llm_select_candidates(
    *,
    query_id: str,
    query_text: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    server_url: str,
    seed: int | None,
    max_keep: int,
) -> Tuple[List[str], Dict[str, int]]:
    system = (
        "You are a legal drafting assistant. Select near-miss clause labels that "
        "are confusingly similar but not identical in meaning."
    )
    lines = [f"Query: {query_text}", "", "Candidates:"]
    for idx, cand in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. {cand['query_id']} | {cand['text']} "
            f"(category: {cand['category']})"
        )
    user = "\n".join(lines)
    user += (
        f"\n\nPick up to {max_keep} candidates that are easy to confuse with the "
        "query but still semantically distinct. Return JSON only: "
        "{\"near_miss_ids\": [\"...\"]}."
    )
    prompt = build_prompt(model_name, system, user)
    raw, usage = query_llm(
        prompt,
        server_url=server_url,
        max_tokens=256,
        temperature=0.2,
        model_name=model_name,
        phase="answer_generation",
        seed=seed,
    )
    if is_r1_like(model_name):
        raw = strip_think(raw)
    try:
        payload = json.loads(raw)
        ids = payload.get("near_miss_ids", [])
        if not isinstance(ids, list):
            return [], usage
        return [str(x).strip() for x in ids if str(x).strip()], usage
    except Exception:
        return [], usage


def mine_near_miss_pairs(
    *,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    min_score: float = 0.35,
    max_candidates_per_query: int = 6,
    include_cross_category: bool = True,
    cross_category_min_score: float = 0.55,
    cross_category_max_per_query: int = 2,
    use_llm: bool = True,
    llm_model_name: str | None = None,
    llm_server_url: str | None = None,
    llm_max_keep: int = 3,
    seed: int | None = 1,
) -> Dict[str, Any]:
    """Return proposed near-miss pairs plus a detailed report."""
    raw_dir = Path(raw_dir)
    queries = load_acord_queries(raw_dir / "queries.jsonl")
    rng = random.Random(seed) if seed is not None else None

    # Organize queries by category for intra-category pairing
    by_category: Dict[str, List[str]] = {}
    for qid, q in queries.items():
        category = str(q.get("category", "")).strip() or "uncategorized"
        by_category.setdefault(category, []).append(qid)

    scored_pairs: Dict[Tuple[str, str], ScoredPair] = {}

    # Intra-category candidate pairs
    for category, qids in by_category.items():
        for i, left in enumerate(qids):
            left_text = str(queries[left].get("text", left))
            candidates: List[ScoredPair] = []
            for right in qids[i + 1 :]:
                right_text = str(queries[right].get("text", right))
                scored = _similarity(left_text, right_text)
                candidates.append(
                    ScoredPair(left, right, scored.score, scored.overlap, scored.jaccard, scored.seq_ratio)
                )
            top = _select_top_pairs(
                candidates,
                min_score=min_score,
                max_pairs=max_candidates_per_query,
            )
            for item in top:
                key = _pair_key(item.left, item.right)
                scored_pairs[key] = item

    # Cross-category pairs (optional)
    if include_cross_category:
        qids = list(queries.keys())
        for left in qids:
            left_text = str(queries[left].get("text", left))
            left_cat = str(queries[left].get("category", "")).strip()
            cross_candidates: List[ScoredPair] = []
            for right in qids:
                if right == left:
                    continue
                right_cat = str(queries[right].get("category", "")).strip()
                if right_cat == left_cat:
                    continue
                right_text = str(queries[right].get("text", right))
                scored = _similarity(left_text, right_text)
                if scored.score < cross_category_min_score:
                    continue
                cross_candidates.append(
                    ScoredPair(left, right, scored.score, scored.overlap, scored.jaccard, scored.seq_ratio)
                )
            cross_top = _select_top_pairs(
                cross_candidates,
                min_score=cross_category_min_score,
                max_pairs=cross_category_max_per_query,
            )
            for item in cross_top:
                key = _pair_key(item.left, item.right)
                scored_pairs[key] = item

    # Optional LLM filtering per query
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if use_llm:
        if not llm_model_name or not llm_server_url:
            raise ValueError("LLM filtering requires llm_model_name and llm_server_url")

        kept_pairs: Dict[Tuple[str, str], ScoredPair] = {}
        for qid, q in queries.items():
            query_text = str(q.get("text", qid))
            candidates = []
            for (a, b), scored in scored_pairs.items():
                if qid not in (a, b):
                    continue
                other = b if qid == a else a
                candidates.append(
                    {
                        "query_id": other,
                        "text": str(queries.get(other, {}).get("text", other)),
                        "category": str(queries.get(other, {}).get("category", "")).strip(),
                        "score": scored.score,
                    }
                )
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if not candidates:
                continue
            if rng:
                rng.shuffle(candidates)
            candidates = candidates[:max_candidates_per_query]

            picked, llm_usage = _llm_select_candidates(
                query_id=qid,
                query_text=query_text,
                candidates=candidates,
                model_name=llm_model_name,
                server_url=llm_server_url,
                seed=seed,
                max_keep=llm_max_keep,
            )
            for k in usage:
                usage[k] += int(llm_usage.get(k, 0))
            if not picked:
                continue
            for other in picked:
                key = _pair_key(qid, other)
                if key in scored_pairs:
                    kept_pairs[key] = scored_pairs[key]
        scored_pairs = kept_pairs

    sorted_pairs = sorted(
        scored_pairs.items(), key=lambda item: item[1].score, reverse=True
    )
    pairs = [list(key) for key, _ in sorted_pairs]
    report_pairs = []
    for (a, b), scored in sorted_pairs:
        report_pairs.append(
            {
                "a": a,
                "b": b,
                "a_text": str(queries.get(a, {}).get("text", a)),
                "b_text": str(queries.get(b, {}).get("text", b)),
                "score": scored.score,
                "jaccard": scored.jaccard,
                "overlap": scored.overlap,
                "seq_ratio": scored.seq_ratio,
                "a_category": str(queries.get(a, {}).get("category", "")),
                "b_category": str(queries.get(b, {}).get("category", "")),
            }
        )

    return {
        "pairs": pairs,
        "report": {
            "meta": {
                "raw_dir": str(raw_dir),
                "min_score": min_score,
                "max_candidates_per_query": max_candidates_per_query,
                "include_cross_category": include_cross_category,
                "cross_category_min_score": cross_category_min_score,
                "cross_category_max_per_query": cross_category_max_per_query,
                "use_llm": use_llm,
                "llm_model_name": llm_model_name or "",
                "llm_server_url": llm_server_url or "",
                "llm_max_keep": llm_max_keep,
                "seed": seed,
            },
            "usage": usage,
            "pairs": report_pairs,
        },
    }


def write_near_miss_yaml(
    *,
    output_path: str | Path,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    description: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Write near-miss pairs to a YAML file and return the report."""
    if yaml is None:
        raise ModuleNotFoundError(
            "pyyaml is required to write near_miss.yaml. "
            "Install it via `pip install pyyaml`."
        )
    payload = mine_near_miss_pairs(raw_dir=raw_dir, **kwargs)
    pairs = payload["pairs"]
    report = payload["report"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "version": 1,
        "description": description
        or "Auto-generated near-miss candidates for ACORD clause retrieval.",
        "pairs": pairs,
    }
    with open(output_path, "wt", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)

    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "wt", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
