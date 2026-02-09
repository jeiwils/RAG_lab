"""Evaluation for the ACORD near-miss-graph-aware retriever (nDCG + CCR)."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.a1_ingestion.dataset_preprocessing_functions import (
    ACORD_DEFAULT_POS_THRESHOLD,
    load_acord_corpus,
    load_acord_qrels_tsv,
    load_acord_queries,
)
from src.b2_reranking.acord_retriever_infer import (
    DEFAULT_MAX_LEN,
    LOCAL_FILES_ONLY,
    score_documents_lora,
)
from src.d1_evaluation.retrieval_metrics import compute_ndcg
from src.ontology.acord_near_miss_graph import (
    DEFAULT_NEAR_MISS_PATH,
    load_acord_ontology,
)
from src.utils.__utils__ import raw_dataset_dir

__all__ = ["evaluate_acord_retriever"]


def _normalize_acord_split(split: str) -> str:
    if split == "dev":
        return "valid"
    return split


def _dedupe_qrels(rows: List[Dict[str, int]]) -> List[Dict[str, int]]:
    best: Dict[str, int] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        score = int(row.get("score", 0))
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    return [{"doc_id": doc_id, "score": score} for doc_id, score in best.items()]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _score_token_overlap(query: str, document: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = _tokenize(document)
    return float(len(q_tokens & d_tokens))


def _score_documents_baseline(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    mode: str,
    rng: random.Random | None,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in candidates:
        doc_text = str(item.get("document", ""))
        if mode == "token_overlap":
            score = _score_token_overlap(query, doc_text)
        elif mode == "random":
            score = float(rng.random()) if rng is not None else 0.0
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
        scored.append({**item, "score": score})
    return scored


def evaluate_acord_retriever(
    *,
    model_checkpoint: str | Path | None = None,
    split: str,
    raw_dir: str | Path = raw_dataset_dir("ACORD"),
    near_miss_path: str | Path | None = DEFAULT_NEAR_MISS_PATH,
    pos_threshold: int = ACORD_DEFAULT_POS_THRESHOLD,
    top_k: int = 10,
    batch_size: int = 32,
    max_length: int = DEFAULT_MAX_LEN,
    local_files_only: bool = LOCAL_FILES_ONLY,
    output_dir: str | Path | None = None,
    max_query_samples: int = 50,
    scoring_mode: str = "lora",
    run_tag: str | None = None,
    baseline_seed: int | None = 1,
) -> Dict[str, Any]:
    """Evaluate a retriever and write a confusion report."""
    raw_dir = Path(raw_dir)
    split_alias = _normalize_acord_split(split)

    queries = load_acord_queries(raw_dir / "queries.jsonl")
    corpus = load_acord_corpus(raw_dir / "corpus.jsonl")
    qrels = load_acord_qrels_tsv(raw_dir / f"{split_alias}.tsv")

    ontology = load_acord_ontology(
        raw_dir=raw_dir,
        near_miss_path=near_miss_path,
        require_near_miss=False,
    )
    near_miss_map = ontology.get("near_miss_map", {})

    pos_by_qid: Dict[str, set[str]] = {}
    for qid, rels in qrels.items():
        deduped = _dedupe_qrels(rels)
        pos_by_qid[qid] = {
            row["doc_id"] for row in deduped if int(row.get("score", 0)) >= pos_threshold
        }

    total_queries = 0
    ndcg_sum = 0.0
    ndcg_at_k_sum = 0.0

    total_top_k = 0
    total_confused = 0
    queries_with_confusion = 0
    pair_confusion: Dict[Tuple[str, str], int] = defaultdict(int)

    query_samples: List[Dict[str, Any]] = []
    missing_docs = 0

    rng = random.Random(baseline_seed) if baseline_seed is not None else None

    for qid, rels in tqdm(qrels.items(), desc=f"acord/{split} eval"):
        q = queries.get(qid)
        if not q:
            continue
        if q.get("split") and str(q.get("split")) != split_alias:
            continue

        deduped = _dedupe_qrels(rels)
        if not deduped:
            continue

        candidates: List[Dict[str, Any]] = []
        for row in deduped:
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            text = corpus.get(doc_id, "")
            if not text:
                missing_docs += 1
                continue
            candidates.append(
                {
                    "doc_id": doc_id,
                    "document": text,
                    "relevance": int(row.get("score", 0)),
                }
            )

        if not candidates:
            continue

        query_text = q.get("text", qid)
        if scoring_mode == "lora":
            if not model_checkpoint:
                raise ValueError(
                    "model_checkpoint is required when scoring_mode='lora'."
                )
            scored, _ = score_documents_lora(
                query_text,
                candidates,
                checkpoint_dir=Path(model_checkpoint),
                batch_size=batch_size,
                max_length=max_length,
                local_files_only=local_files_only,
            )
        else:
            scored = _score_documents_baseline(
                query_text,
                candidates,
                mode=scoring_mode,
                rng=rng,
            )

        scored.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        true_scores = [float(item.get("relevance", 0)) for item in scored]

        ndcg = compute_ndcg(true_scores, k=None)
        ndcg_k = compute_ndcg(true_scores, k=top_k)
        ndcg_sum += ndcg
        ndcg_at_k_sum += ndcg_k
        total_queries += 1

        top_docs = scored[:top_k] if top_k > 0 else scored
        total_top_k += len(top_docs)

        near_miss_qids = near_miss_map.get(qid, set())
        confused_doc_ids: set[str] = set()
        for doc in top_docs:
            doc_id = str(doc.get("doc_id", ""))
            if not doc_id or doc_id in pos_by_qid.get(qid, set()):
                continue
            for nm_qid in near_miss_qids:
                if doc_id in pos_by_qid.get(nm_qid, set()):
                    pair_confusion[(qid, nm_qid)] += 1
                    confused_doc_ids.add(doc_id)

        if confused_doc_ids:
            queries_with_confusion += 1
            total_confused += len(confused_doc_ids)
            if len(query_samples) < max_query_samples:
                query_samples.append(
                    {
                        "query_id": qid,
                        "query": q.get("text", qid),
                        "confused_hits": len(confused_doc_ids),
                        "confused_doc_ids": sorted(confused_doc_ids),
                    }
                )

    mean_ndcg = ndcg_sum / total_queries if total_queries else 0.0
    mean_ndcg_at_k = ndcg_at_k_sum / total_queries if total_queries else 0.0

    ccr_docs = total_confused / total_top_k if total_top_k else 0.0
    ccr_queries = queries_with_confusion / total_queries if total_queries else 0.0

    pair_confusion_list = [
        {
            "query_id": qid,
            "near_miss_id": nm_qid,
            "hits": hits,
            "hit_rate": hits / max(top_k, 1),
        }
        for (qid, nm_qid), hits in pair_confusion.items()
        if hits > 0
    ]
    pair_confusion_list.sort(key=lambda x: x["hits"], reverse=True)

    report = {
        "meta": {
            "dataset": "acord",
            "split": split,
            "model_checkpoint": str(model_checkpoint) if model_checkpoint else None,
            "scoring_mode": scoring_mode,
            "run_tag": run_tag,
            "near_miss_path": str(near_miss_path) if near_miss_path else None,
            "pos_threshold": pos_threshold,
            "top_k": top_k,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "overall": {
            "queries_total": total_queries,
            "mean_ndcg": round(mean_ndcg, 6),
            "mean_ndcg_at_k": round(mean_ndcg_at_k, 6),
            "ccr_docs": round(ccr_docs, 6),
            "ccr_queries": round(ccr_queries, 6),
            "confused_hits": total_confused,
            "total_top_k": total_top_k,
            "missing_docs": missing_docs,
        },
        "pair_confusion": pair_confusion_list,
        "query_confusion_samples": query_samples,
    }

    if output_dir is None:
        output_dir = Path(f"data/results/acord_retriever/{split}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_tag:
        out_path = output_dir / f"confusion_report_{run_tag}.json"
    elif scoring_mode != "lora":
        out_path = output_dir / f"confusion_report_{scoring_mode}.json"
    else:
        out_path = output_dir / "confusion_report.json"
    with open(out_path, "wt", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    raise SystemExit(
        "Run this module through the ACORD_near_miss_retriever orchestrator "
        "or call evaluate_acord_retriever(...) directly."
    )
