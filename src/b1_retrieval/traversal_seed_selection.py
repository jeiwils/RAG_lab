"""Seed passage retrieval utilities for hybrid dense/sparse retrieval."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.b1_retrieval.dense_retrieval import faiss_search_topk
from src.b1_retrieval.hybrid_retrieval import (
    DEFAULT_HYBRID_ALPHA,
    retrieve_hybrid_candidates,
)
from src.b1_retrieval.sparse_retrieval import retrieve_sparse_candidates
from src.a3_representations.sparse_representations import extract_keywords

__all__ = ["DEFAULT_SEED_TOP_K", "select_seed_passages"]


DEFAULT_SEED_TOP_K = 20


def select_seed_passages(
    query_text: str,
    query_emb: np.ndarray,
    passage_metadata: List[Dict],
    passage_index,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    question_id: str | None = None,
) -> List[str]:
    """Select top seed passages using dense (FAISS) and sparse (BM25) signals.

    The hybrid score is ``alpha * sim_cos + (1 - alpha) * sim_bm25`` where
    ``alpha`` weights dense vs. sparse similarity. When ``question_id`` is
    provided, the function logs the top dense and BM25 candidates to aid
    debugging.
    """

    if passage_index.ntotal != len(passage_metadata):
        raise ValueError(
            "FAISS index size mismatch: index has "
            f"{passage_index.ntotal} vectors but metadata lists "
            f"{len(passage_metadata)} passages"
        )

    query_keywords = set(extract_keywords(query_text))

    if question_id is not None:
        query_vec = np.asarray(query_emb).reshape(1, -1)
        faiss_idxs, faiss_scores = faiss_search_topk(
            query_vec, passage_index, top_k=seed_top_k
        )
        faiss_pairs = [
            (passage_metadata[int(i)]["passage_id"], float(s))
            for i, s in zip(faiss_idxs, faiss_scores)
        ]

        bm25_pairs = retrieve_sparse_candidates(
            query_keywords,
            passage_metadata,
            top_k=seed_top_k,
            keyword_field="keywords_passage",
        )
        bm25_pairs = [
            (passage_metadata[row["idx"]]["passage_id"], row["sim_bm25"])
            for row in bm25_pairs
        ]

        print(f"[select_seed_passages][{question_id}] FAISS top: {faiss_pairs[:5]}")
        print(f"[select_seed_passages][{question_id}] BM25 top: {bm25_pairs[:5]}")

    candidates = retrieve_hybrid_candidates(
        np.asarray(query_emb),
        query_keywords,
        passage_metadata,
        passage_index,
        top_k=seed_top_k,
        alpha=alpha,
        keyword_field="keywords_passage",
    )

    return [passage_metadata[c["idx"]]["passage_id"] for c in candidates]
