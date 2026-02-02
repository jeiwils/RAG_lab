"""Sparse retrieval utilities based on BM25 over keyword terms."""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Set

logger = logging.getLogger(__name__)


def _bm25_idf(num_docs: int, doc_freq: int) -> float:
    return math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def bm25_score(
    query_terms: Set[str],
    doc_terms: Set[str],
    idf: Dict[str, float],
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_terms or not doc_terms or avgdl <= 0:
        return 0.0
    dl = len(doc_terms)
    denom_const = k1 * (1.0 - b + b * (dl / avgdl))
    score = 0.0
    for term in query_terms & doc_terms:
        term_idf = idf.get(term, 0.0)
        tf = 1.0
        score += term_idf * (tf * (k1 + 1.0)) / (tf + denom_const)
    return score


def compute_bm25_stats(
    query_terms: Set[str],
    metadata: List[Dict],
    *,
    keyword_field: str | None = None,
    filter_fn: Callable[[int], bool] | None = None,
) -> tuple[Dict[str, float], float, int, str | None]:
    """Compute BM25 IDF and average document length for the given query terms."""
    if keyword_field is None and metadata:
        if "keywords" in metadata[0]:
            keyword_field = "keywords"
        else:
            keyword_field = "keywords_passage"

    term_df = {term: 0 for term in query_terms}
    total_len = 0
    doc_count = 0

    for idx, row in enumerate(metadata):
        if filter_fn is not None and not filter_fn(idx):
            continue
        terms = set(row.get(keyword_field, []))
        total_len += len(terms)
        doc_count += 1
        for term in query_terms & terms:
            term_df[term] += 1

    avgdl = (total_len / doc_count) if doc_count else 0.0
    idf = (
        {term: _bm25_idf(doc_count, df) for term, df in term_df.items()}
        if doc_count
        else {}
    )

    return idf, avgdl, doc_count, keyword_field


def retrieve_sparse_candidates(
    query_keywords: Set[str],
    metadata: List[Dict],
    top_k: int,
    *,
    keyword_field: str | None = None,
    filter_fn: Callable[[int], bool] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, float]]:
    """Retrieve top candidates using BM25 over keyword sets.

    Parameters
    ----------
    query_keywords:
        Sparse keyword set associated with the query.
    metadata:
        Sequence of metadata dicts aligned with the target collection.
    top_k:
        Number of candidates returned after sorting by BM25 score.
    keyword_field:
        Optional override for the keyword field name in ``metadata``.
    filter_fn:
        Optional callable ``filter_fn(idx) -> bool`` applied to candidate
        indices. Candidates for which the function returns ``False`` are
        discarded.
    """

    if top_k <= 0:
        return []

    if not query_keywords:
        logger.debug("No query keywords provided; returning empty sparse candidates")
        return []

    idf, avgdl, doc_count, keyword_field = compute_bm25_stats(
        query_keywords,
        metadata,
        keyword_field=keyword_field,
        filter_fn=filter_fn,
    )
    if doc_count == 0:
        return []

    results: List[Dict[str, float]] = []
    for idx, row in enumerate(metadata):
        if filter_fn is not None and not filter_fn(idx):
            continue
        terms = set(row.get(keyword_field, []))
        sim_bm25 = bm25_score(query_keywords, terms, idf, avgdl, k1=k1, b=b)
        results.append({"idx": idx, "sim_bm25": round(sim_bm25, 4)})

    results.sort(key=lambda x: x["sim_bm25"], reverse=True)
    return results[:top_k]
