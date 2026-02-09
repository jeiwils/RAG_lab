"""Dataset builders for the ACORD near-miss-graph-aware retriever."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.a1_ingestion.dataset_preprocessing_functions import (
    ACORD_DEFAULT_HARD_THRESHOLD,
    ACORD_DEFAULT_NEG_THRESHOLD,
    ACORD_DEFAULT_POS_THRESHOLD,
    load_acord_corpus,
    load_acord_qrels_tsv,
    load_acord_queries,
)
from src.ontology.acord_near_miss_graph import (
    DEFAULT_NEAR_MISS_PATH,
    load_acord_ontology,
)
from src.utils.__utils__ import load_jsonl, raw_dataset_dir, save_jsonl

__all__ = [
    "DEFAULT_ACORD_RAW_DIR",
    "build_acord_retriever_examples",
    "load_acord_retriever_examples",
    "split_examples_by_difficulty",
    "write_acord_retriever_examples",
]

DEFAULT_ACORD_RAW_DIR = raw_dataset_dir("ACORD")
DEFAULT_DATASET_VERSION = "v1"


def _normalize_acord_split(split: str) -> str:
    if split == "dev":
        return "valid"
    return split


def _dedupe_qrels(rows: Iterable[Dict[str, int]]) -> List[Dict[str, int]]:
    best: Dict[str, int] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        score = int(row.get("score", 0))
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    return [{"doc_id": doc_id, "score": score} for doc_id, score in best.items()]


def _sample(
    items: List[Any], *, max_items: int | None, rng: random.Random | None
) -> List[Any]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return list(items)
    if rng is None:
        return list(items)[:max_items]
    return rng.sample(list(items), max_items)


_TRICK_LABEL_RE = re.compile(
    r"^\s*[\(\[]?\s*([AB])\s*[\)\]]?\s*[:\-\|]\s*(.+)$",
    flags=re.I,
)


def _attach_metadata(
    example: Dict[str, Any],
    *,
    metadata: Dict[str, Any] | None,
    extra_fields: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if extra_fields:
        example.update(extra_fields)
    if metadata:
        example["metadata"] = dict(metadata)
    return example


def _build_labelled_examples(
    *,
    qid: str,
    query: str,
    category: str,
    docs: Iterable[Dict[str, Any]],
    corpus: Dict[str, str],
    label: int,
    label_type: str,
    difficulty: str,
    metadata: Dict[str, Any] | None = None,
    extra_fields: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        text = corpus.get(doc_id, "")
        if not text:
            continue
        examples.append(
            _attach_metadata(
                {
                    "query_id": qid,
                    "query": query,
                    "category": category,
                    "doc_id": doc_id,
                    "document": text,
                    "label": int(label),
                    "label_type": label_type,
                    "difficulty": difficulty,
                    "score": int(doc.get("score", 0)),
                },
                metadata=metadata,
                extra_fields=extra_fields,
            )
        )
    return examples


def _parse_trick_query_item(
    item: Any, *, query_id: str, near_miss_id: str
) -> Dict[str, str] | None:
    if isinstance(item, dict):
        text = str(
            item.get("text")
            or item.get("query")
            or item.get("trick_query")
            or ""
        ).strip()
        target_raw = str(item.get("target") or item.get("label") or "").strip()
        target_query_id = str(item.get("target_query_id") or "").strip()
        target = target_raw.upper() if target_raw else ""
        if not target and target_query_id:
            if target_query_id == query_id:
                target = "A"
            elif target_query_id == near_miss_id:
                target = "B"
        if target not in {"A", "B"}:
            return None
        if not text:
            return None
        return {
            "text": text,
            "target": target,
            "target_query_id": query_id if target == "A" else near_miss_id,
        }

    if isinstance(item, str):
        match = _TRICK_LABEL_RE.match(item)
        if not match:
            return None
        target = match.group(1).upper()
        text = match.group(2).strip()
        if not text:
            return None
        return {
            "text": text,
            "target": target,
            "target_query_id": query_id if target == "A" else near_miss_id,
        }
    return None


def _load_trick_queries(
    path: str | Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "trick_pairs_total": 0,
        "trick_pairs_skipped": 0,
        "trick_queries_total": 0,
        "trick_queries_skipped": 0,
        "trick_queries_deduped": 0,
    }
    seen: set[tuple[str, str]] = set()
    for row in load_jsonl(str(path)):
        query_id = str(row.get("query_id", "")).strip()
        near_miss_id = str(row.get("near_miss_id", "")).strip()
        if not query_id or not near_miss_id:
            stats["trick_pairs_skipped"] += 1
            continue
        stats["trick_pairs_total"] += 1
        for item in row.get("trick_queries", []) or []:
            parsed = _parse_trick_query_item(
                item, query_id=query_id, near_miss_id=near_miss_id
            )
            if not parsed:
                stats["trick_queries_skipped"] += 1
                continue
            key = (parsed["target_query_id"], parsed["text"])
            if key in seen:
                stats["trick_queries_deduped"] += 1
                continue
            seen.add(key)
            stats["trick_queries_total"] += 1
            records.append(
                {
                    "text": parsed["text"],
                    "target": parsed["target"],
                    "target_query_id": parsed["target_query_id"],
                    "source_query_id": query_id,
                    "near_miss_id": near_miss_id,
                    "pair_id": f"{query_id}__{near_miss_id}",
                }
            )
    return records, stats


def _build_examples_for_query(
    *,
    qid: str,
    query_text: str,
    category: str,
    pos_rows: List[Dict[str, int]],
    hard_rows: List[Dict[str, int]],
    neg_rows: List[Dict[str, int]],
    corpus: Dict[str, str],
    pos_by_qid: Dict[str, List[Dict[str, int]]],
    near_miss_map: Dict[str, List[str]],
    siblings: Dict[str, set[str]],
    score_lookup: Dict[Tuple[str, str], int],
    include_near_miss: bool,
    include_siblings: bool,
    max_pos: int | None,
    max_easy_neg: int | None,
    max_hard_neg: int | None,
    max_ontology_neg: int | None,
    rng: random.Random | None,
    metadata: Dict[str, Any] | None,
    extra_fields: Dict[str, Any] | None,
    output_qid: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    qid_out = output_qid or qid
    pos_rows = _sample(pos_rows, max_items=max_pos, rng=rng)
    easy_rows = _sample(neg_rows, max_items=max_easy_neg, rng=rng)
    hard_rows = _sample(hard_rows, max_items=max_hard_neg, rng=rng)

    used_doc_ids = {row["doc_id"] for row in pos_rows}

    pos_examples = _build_labelled_examples(
        qid=qid_out,
        query=query_text,
        category=category,
        docs=pos_rows,
        corpus=corpus,
        label=1,
        label_type="pos",
        difficulty="pos",
        metadata=metadata,
        extra_fields=extra_fields,
    )

    easy_examples = _build_labelled_examples(
        qid=qid_out,
        query=query_text,
        category=category,
        docs=[row for row in easy_rows if row["doc_id"] not in used_doc_ids],
        corpus=corpus,
        label=0,
        label_type="neg_easy",
        difficulty="easy",
        metadata=metadata,
        extra_fields=extra_fields,
    )

    used_doc_ids.update({row["doc_id"] for row in easy_rows})

    hard_examples = _build_labelled_examples(
        qid=qid_out,
        query=query_text,
        category=category,
        docs=[row for row in hard_rows if row["doc_id"] not in used_doc_ids],
        corpus=corpus,
        label=0,
        label_type="neg_hard_score",
        difficulty="hard",
        metadata=metadata,
        extra_fields=extra_fields,
    )

    used_doc_ids.update({row["doc_id"] for row in hard_rows})

    ontology_candidates: Dict[str, Dict[str, Any]] = {}
    if include_near_miss:
        for nm_qid in near_miss_map.get(qid, []):
            for row in pos_by_qid.get(nm_qid, []):
                doc_id = row.get("doc_id")
                if not doc_id or doc_id in used_doc_ids:
                    continue
                entry = ontology_candidates.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": score_lookup.get((nm_qid, doc_id), 0),
                        "source_query_id": nm_qid,
                        "label_type": "neg_near_miss",
                    },
                )
                entry.setdefault("source_query_ids", set()).add(nm_qid)

    if include_siblings:
        for sib_qid in siblings.get(qid, set()):
            for row in pos_by_qid.get(sib_qid, []):
                doc_id = row.get("doc_id")
                if not doc_id or doc_id in used_doc_ids:
                    continue
                entry = ontology_candidates.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": score_lookup.get((sib_qid, doc_id), 0),
                        "source_query_id": sib_qid,
                        "label_type": "neg_sibling",
                    },
                )
                entry.setdefault("source_query_ids", set()).add(sib_qid)

    ontology_rows: List[Dict[str, Any]] = []
    for entry in ontology_candidates.values():
        entry["source_query_ids"] = sorted(entry.get("source_query_ids", []))
        ontology_rows.append(entry)

    if max_ontology_neg:
        if rng is None:
            ontology_rows = ontology_rows[:max_ontology_neg]
        else:
            if len(ontology_rows) > max_ontology_neg:
                ontology_rows = rng.sample(ontology_rows, max_ontology_neg)

    ontology_examples: List[Dict[str, Any]] = []
    skipped_missing_docs = 0
    for row in ontology_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id or doc_id in used_doc_ids:
            continue
        text = corpus.get(doc_id, "")
        if not text:
            skipped_missing_docs += 1
            continue
        ontology_examples.append(
            _attach_metadata(
                {
                    "query_id": qid_out,
                    "query": query_text,
                    "category": category,
                    "doc_id": doc_id,
                    "document": text,
                    "label": 0,
                    "label_type": row.get("label_type", "neg_near_miss"),
                    "difficulty": "hard",
                    "score": int(row.get("score", 0)),
                    "source_query_id": row.get("source_query_id", ""),
                    "source_query_ids": row.get("source_query_ids", []),
                },
                metadata=metadata,
                extra_fields=extra_fields,
            )
        )
        used_doc_ids.add(doc_id)

    examples = pos_examples + easy_examples + hard_examples + ontology_examples
    counts = {
        "examples_total": len(examples),
        "examples_pos": len(pos_examples),
        "examples_easy_neg": len(easy_examples),
        "examples_hard_score_neg": len(hard_examples),
        "examples_near_miss_neg": sum(
            1
            for ex in ontology_examples
            if ex.get("label_type") == "neg_near_miss"
        ),
        "examples_sibling_neg": sum(
            1
            for ex in ontology_examples
            if ex.get("label_type") == "neg_sibling"
        ),
        "skipped_missing_docs": skipped_missing_docs,
    }

    return examples, counts


def build_acord_retriever_examples(
    *,
    split: str,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    near_miss_path: str | Path | None = DEFAULT_NEAR_MISS_PATH,
    pos_threshold: int = ACORD_DEFAULT_POS_THRESHOLD,
    neg_threshold: int = ACORD_DEFAULT_NEG_THRESHOLD,
    hard_threshold: int | None = ACORD_DEFAULT_HARD_THRESHOLD,
    max_pos: int | None = None,
    max_easy_neg: int | None = None,
    max_hard_neg: int | None = None,
    max_ontology_neg: int | None = None,
    trick_queries_path: str | Path | None = None,
    include_trick_queries: bool = False,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    seed: int | None = None,
    include_siblings: bool = True,
    include_near_miss: bool = True,
    drop_no_positives: bool = True,
    require_near_miss: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build ACORD query-document examples with near-miss-graph-aware hard negatives.

    Optionally augments the dataset with labelled trick queries (adversarial
    paraphrases) and stamps examples with dataset metadata.
    """
    raw_dir = Path(raw_dir)
    split_alias = _normalize_acord_split(split)
    queries_path = raw_dir / "queries.jsonl"
    corpus_path = raw_dir / "corpus.jsonl"
    qrels_path = raw_dir / f"{split_alias}.tsv"

    queries = load_acord_queries(queries_path)
    corpus = load_acord_corpus(corpus_path)
    qrels = load_acord_qrels_tsv(qrels_path)

    ontology = load_acord_ontology(
        raw_dir=raw_dir,
        near_miss_path=near_miss_path,
        require_near_miss=require_near_miss,
    )
    siblings = ontology.get("siblings", {})
    near_miss_map = ontology.get("near_miss_map", {})

    pos_by_qid: Dict[str, List[Dict[str, int]]] = {}
    hard_by_qid: Dict[str, List[Dict[str, int]]] = {}
    neg_by_qid: Dict[str, List[Dict[str, int]]] = {}
    score_lookup: Dict[Tuple[str, str], int] = {}

    for qid, rels in qrels.items():
        q = queries.get(qid)
        if not q:
            continue
        if q.get("split") and str(q.get("split")) != split_alias:
            continue
        deduped = _dedupe_qrels(rels)
        for row in deduped:
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                score_lookup[(qid, doc_id)] = int(row.get("score", 0))
        pos_by_qid[qid] = [
            r for r in deduped if int(r.get("score", 0)) >= pos_threshold
        ]
        if hard_threshold is not None:
            hard_by_qid[qid] = [
                r
                for r in deduped
                if hard_threshold <= int(r.get("score", 0)) < pos_threshold
            ]
        else:
            hard_by_qid[qid] = []
        neg_by_qid[qid] = [
            r for r in deduped if int(r.get("score", 0)) <= neg_threshold
        ]

    rng = random.Random(seed) if seed is not None else None

    metadata: Dict[str, Any] = {
        "dataset_version": dataset_version,
        "includes_trick_queries": include_trick_queries,
    }
    if dataset_version in {None, ""}:
        metadata.pop("dataset_version", None)
    if not metadata:
        metadata = {}

    examples: List[Dict[str, Any]] = []
    summary = {
        "queries_total": 0,
        "queries_used": 0,
        "examples_total": 0,
        "examples_pos": 0,
        "examples_easy_neg": 0,
        "examples_hard_score_neg": 0,
        "examples_near_miss_neg": 0,
        "examples_sibling_neg": 0,
        "skipped_missing_docs": 0,
        "examples_trick_total": 0,
        "trick_pairs_total": 0,
        "trick_pairs_skipped": 0,
        "trick_queries_total": 0,
        "trick_queries_skipped": 0,
        "trick_queries_deduped": 0,
        "trick_queries_used": 0,
        "trick_queries_skipped_missing_qid": 0,
        "trick_queries_skipped_no_positives": 0,
        "includes_trick_queries": include_trick_queries,
        "dataset_version": dataset_version,
    }

    def _accumulate_counts(counts: Dict[str, int]) -> None:
        summary["examples_pos"] += counts.get("examples_pos", 0)
        summary["examples_easy_neg"] += counts.get("examples_easy_neg", 0)
        summary["examples_hard_score_neg"] += counts.get("examples_hard_score_neg", 0)
        summary["examples_near_miss_neg"] += counts.get("examples_near_miss_neg", 0)
        summary["examples_sibling_neg"] += counts.get("examples_sibling_neg", 0)
        summary["skipped_missing_docs"] += counts.get("skipped_missing_docs", 0)

    for qid, pos_rows in pos_by_qid.items():
        summary["queries_total"] += 1
        q = queries.get(qid)
        if not q:
            continue
        if q.get("split") and str(q.get("split")) != split_alias:
            continue
        if drop_no_positives and not pos_rows:
            continue

        query_text = q.get("text", qid)
        category = str(q.get("category", "")).strip()
        built, counts = _build_examples_for_query(
            qid=qid,
            query_text=query_text,
            category=category,
            pos_rows=pos_rows,
            hard_rows=hard_by_qid.get(qid, []),
            neg_rows=neg_by_qid.get(qid, []),
            corpus=corpus,
            pos_by_qid=pos_by_qid,
            near_miss_map=near_miss_map,
            siblings=siblings,
            score_lookup=score_lookup,
            include_near_miss=include_near_miss,
            include_siblings=include_siblings,
            max_pos=max_pos,
            max_easy_neg=max_easy_neg,
            max_hard_neg=max_hard_neg,
            max_ontology_neg=max_ontology_neg,
            rng=rng,
            metadata=metadata,
            extra_fields={"query_source": "base"},
        )
        examples.extend(built)

        summary["queries_used"] += 1
        _accumulate_counts(counts)

    if include_trick_queries:
        if trick_queries_path is None:
            raise ValueError(
                "include_trick_queries=True but no trick_queries_path was provided."
            )
        trick_path = Path(trick_queries_path)
        if not trick_path.exists():
            raise FileNotFoundError(f"Missing trick_queries file: {trick_path}")
        trick_queries, trick_stats = _load_trick_queries(trick_path)
        for key, val in trick_stats.items():
            summary[key] = summary.get(key, 0) + val

        for trick in trick_queries:
            target_qid = str(trick.get("target_query_id", "")).strip()
            if not target_qid:
                summary["trick_queries_skipped_missing_qid"] += 1
                continue
            q = queries.get(target_qid)
            if not q:
                summary["trick_queries_skipped_missing_qid"] += 1
                continue
            pos_rows = pos_by_qid.get(target_qid, [])
            if drop_no_positives and not pos_rows:
                summary["trick_queries_skipped_no_positives"] += 1
                continue
            query_text = str(trick.get("text", "")).strip()
            if not query_text:
                summary["trick_queries_skipped"] += 1
                continue
            category = str(q.get("category", "")).strip()
            extra_fields = {
                "query_source": "trick",
                "trick_target": trick.get("target"),
                "trick_source_query_id": trick.get("source_query_id", ""),
                "trick_near_miss_id": trick.get("near_miss_id", ""),
                "trick_pair_id": trick.get("pair_id", ""),
            }
            built, counts = _build_examples_for_query(
                qid=target_qid,
                query_text=query_text,
                category=category,
                pos_rows=pos_rows,
                hard_rows=hard_by_qid.get(target_qid, []),
                neg_rows=neg_by_qid.get(target_qid, []),
                corpus=corpus,
                pos_by_qid=pos_by_qid,
                near_miss_map=near_miss_map,
                siblings=siblings,
                score_lookup=score_lookup,
                include_near_miss=include_near_miss,
                include_siblings=include_siblings,
                max_pos=max_pos,
                max_easy_neg=max_easy_neg,
                max_hard_neg=max_hard_neg,
                max_ontology_neg=max_ontology_neg,
                rng=rng,
                metadata=metadata,
                extra_fields=extra_fields,
            )
            examples.extend(built)
            summary["trick_queries_used"] += 1
            summary["examples_trick_total"] += counts.get("examples_total", 0)
            _accumulate_counts(counts)

    if rng is not None:
        rng.shuffle(examples)

    summary["examples_total"] = len(examples)
    summary["pos_threshold"] = pos_threshold
    summary["neg_threshold"] = neg_threshold
    summary["hard_threshold"] = hard_threshold
    summary["split"] = split
    return examples, summary


def split_examples_by_difficulty(
    examples: Iterable[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (pos, easy_neg, hard_neg) lists."""
    pos: List[Dict[str, Any]] = []
    easy: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    for ex in examples:
        label = int(ex.get("label", 0))
        if label == 1:
            pos.append(ex)
            continue
        difficulty = ex.get("difficulty")
        if difficulty == "easy":
            easy.append(ex)
        else:
            hard.append(ex)
    return pos, easy, hard


def write_acord_retriever_examples(
    *,
    split: str,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    near_miss_path: str | Path | None = DEFAULT_NEAR_MISS_PATH,
    out_path: str | Path | None = None,
    easy_out_path: str | Path | None = None,
    hard_out_path: str | Path | None = None,
    pos_threshold: int = ACORD_DEFAULT_POS_THRESHOLD,
    neg_threshold: int = ACORD_DEFAULT_NEG_THRESHOLD,
    hard_threshold: int | None = ACORD_DEFAULT_HARD_THRESHOLD,
    max_pos: int | None = None,
    max_easy_neg: int | None = None,
    max_hard_neg: int | None = None,
    max_ontology_neg: int | None = None,
    trick_queries_path: str | Path | None = None,
    include_trick_queries: bool = False,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    seed: int | None = None,
    include_siblings: bool = True,
    include_near_miss: bool = True,
    drop_no_positives: bool = True,
    require_near_miss: bool = True,
) -> Dict[str, Any]:
    """Write retriever examples to JSONL and return a summary."""
    examples, summary = build_acord_retriever_examples(
        split=split,
        raw_dir=raw_dir,
        near_miss_path=near_miss_path,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        hard_threshold=hard_threshold,
        max_pos=max_pos,
        max_easy_neg=max_easy_neg,
        max_hard_neg=max_hard_neg,
        max_ontology_neg=max_ontology_neg,
        trick_queries_path=trick_queries_path,
        include_trick_queries=include_trick_queries,
        dataset_version=dataset_version,
        seed=seed,
        include_siblings=include_siblings,
        include_near_miss=include_near_miss,
        drop_no_positives=drop_no_positives,
        require_near_miss=require_near_miss,
    )

    if out_path is not None:
        save_jsonl(str(out_path), examples)

    if easy_out_path is not None or hard_out_path is not None:
        pos, easy, hard = split_examples_by_difficulty(examples)
        if easy_out_path is not None:
            save_jsonl(str(easy_out_path), pos + easy)
        if hard_out_path is not None:
            save_jsonl(str(hard_out_path), pos + hard)

    return summary


def load_acord_retriever_examples(path: str | Path) -> List[Dict[str, Any]]:
    """Load retriever examples from a JSONL file."""
    return list(load_jsonl(str(path)))
