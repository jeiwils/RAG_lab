"""Hybrid dense + sparse retrieval utilities."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Set

import numpy as np

from src.b1_retrieval.dense_retrieval import faiss_search_topk
from src.b1_retrieval.sparse_retrieval import (
    bm25_score,
    compute_bm25_stats,
    retrieve_sparse_candidates,
)

logger = logging.getLogger(__name__)

### DEFAULTS
# Default weighting for combining dense cosine and sparse BM25 similarity.
DEFAULT_HYBRID_ALPHA = 0.5


def _min_max_norm(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    vals = np.array(list(values.values()), dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax == vmin:
        return {k: 0.0 for k in values}
    scale = vmax - vmin
    return {k: (v - vmin) / scale for k, v in values.items()}

def retrieve_hybrid_candidates(
    query_vec: np.ndarray,
    query_keywords: Set[str],
    metadata: List[Dict],
    index,
    top_k: int,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    *,
    keyword_field: str | None = None,
    filter_fn: Callable[[int], bool] | None = None,
    dense_pool: int | None = None,
    sparse_pool: int | None = None,
    normalize_scores: bool = True,
) -> List[Dict[str, float]]:
    """Retrieve and fuse dense + sparse candidates, then rerank by blended score."""
    if top_k <= 0:
        logger.warning("top_k must be > 0; returning empty list")
        return []

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    if keyword_field is None and metadata:
        keyword_field = "keywords" if "keywords" in metadata[0] else "keywords_passage"

    query_vec = query_vec.reshape(1, -1)
    dense_pool = max(top_k * 5, top_k) if dense_pool is None else dense_pool
    sparse_pool = top_k if sparse_pool is None else sparse_pool

    # Dense retrieval
    faiss_idxs, faiss_scores = faiss_search_topk(query_vec, index, top_k=dense_pool)
    n_meta = len(metadata)
    valid_pairs = [
        (int(idx), float(score))
        for idx, score in zip(faiss_idxs, faiss_scores)
        if idx != -1 and 0 <= int(idx) < n_meta
    ]
    dense_candidates = [idx for idx, _ in valid_pairs]
    dense_scores_raw = {idx: score for idx, score in valid_pairs}

    # Sparse retrieval
    sparse_scores: Dict[int, float] = {}
    if query_keywords:
        sparse_candidates = retrieve_sparse_candidates(
            query_keywords,
            metadata,
            top_k=sparse_pool,
            keyword_field=keyword_field,
            filter_fn=filter_fn,
        )
        sparse_scores = {row["idx"]: row["sim_bm25"] for row in sparse_candidates}
    else:
        logger.debug("No query keywords provided; skipping sparse retrieval")

    # Union and filter candidates
    candidate_set = set(dense_candidates)
    if sparse_scores:
        candidate_set.update(sparse_scores.keys())
    if filter_fn is not None:
        candidate_set = {idx for idx in candidate_set if filter_fn(idx)}
    candidate_idxs = sorted(candidate_set)

    # Compute BM25 scores if keywords are provided
    if query_keywords:
        idf, avgdl, doc_count, _ = compute_bm25_stats(
            query_keywords,
            metadata,
            keyword_field=keyword_field,
            filter_fn=filter_fn,
        )
        if doc_count == 0:
            sparse_scores = {idx: 0.0 for idx in candidate_idxs}
        else:
            sparse_scores = {
                idx: bm25_score(
                    query_keywords,
                    set(metadata[idx].get(keyword_field, [])),
                    idf,
                    avgdl,
                )
                for idx in candidate_idxs
            }
    else:
        sparse_scores = {idx: 0.0 for idx in candidate_idxs}

    # Normalize scores
    if normalize_scores:
        dense_scores = _min_max_norm(dense_scores_raw) if dense_scores_raw else {idx: 0.0 for idx in candidate_idxs}
        sparse_scores = _min_max_norm(sparse_scores)
    else:
        dense_scores = {idx: dense_scores_raw.get(idx, 0.0) for idx in candidate_idxs}

    # Blend scores and sort
    results = []
    for idx in candidate_idxs:
        sim_cos = float(dense_scores.get(idx, 0.0))
        sim_bm25 = float(sparse_scores.get(idx, 0.0))
        sim_hybrid = alpha * sim_cos + (1.0 - alpha) * sim_bm25 if query_keywords else sim_cos
        results.append({
            "idx": idx,
            "sim_cos": round(sim_cos, 4),
            "sim_bm25": round(sim_bm25, 4),
            "sim_hybrid": round(sim_hybrid, 4),
        })

    results.sort(key=lambda x: x["sim_hybrid"], reverse=True)
    return results[:top_k]

