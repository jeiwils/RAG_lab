"""Dense retrieval utilities backed by FAISS indexes."""

from __future__ import annotations

import faiss
import numpy as np


def faiss_search_topk(query_emb: np.ndarray, index, top_k: int = 50):
    """Retrieve ``top_k`` most similar items from a FAISS index."""

    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    norm = np.linalg.norm(query_emb)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(
            f"Query embedding norm invalid ({norm}); check emb_model.encode output."
        )
    faiss.normalize_L2(query_emb)
    scores, idx = index.search(query_emb, top_k)
    return idx[0], scores[0]
