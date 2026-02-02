"""Indexing-stage retrieval helpers for chunk indices."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.a3_representations.sparse_representations import extract_keywords
from src.b1_retrieval import (
    faiss_search_topk,
    retrieve_hybrid_candidates,
    retrieve_sparse_candidates,
)

__all__ = ["_retrieve_chunk_indices"]


def _retrieve_chunk_indices(
    retriever: str,
    query_text: str,
    query_vec: np.ndarray | None,
    metadata: List[Dict[str, Any]],
    index,
    *,
    top_k: int,
    alpha: float,
) -> List[int]:
    if retriever == "dense":
        if query_vec is None:
            raise ValueError("Dense retrieval requires a query vector.")
        idxs, _ = faiss_search_topk(query_vec, index, top_k=top_k)
        return [int(i) for i in idxs if i != -1]

    query_keywords = set(extract_keywords(query_text))
    if retriever == "hybrid":
        if query_vec is None:
            raise ValueError("Hybrid retrieval requires a query vector.")
        candidates = retrieve_hybrid_candidates(
            query_vec,
            query_keywords,
            metadata,
            index,
            top_k=top_k,
            alpha=alpha,
            keyword_field="keywords_passage",
        )
        return [int(c["idx"]) for c in candidates]

    if retriever == "sparse":
        candidates = retrieve_sparse_candidates(
            query_keywords,
            metadata,
            top_k=top_k,
            keyword_field="keywords_passage",
        )
        return [int(c["idx"]) for c in candidates]

    raise ValueError(f"Unknown retriever: {retriever}")
