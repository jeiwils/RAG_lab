"""Passage reranking helpers based on traversal helpfulness."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx

__all__ = ["compute_helpfulness", "rerank_passages_by_helpfulness"]


def compute_helpfulness(
    passage_id: str,
    query_similarity: float,
    visit_counts: Dict[str, int],
) -> float:
    """Compute a helpfulness score for a passage.

    The score is the average of the passage-query similarity and the
    normalized visit frequency across the traversal.
    """

    total_visits = sum(visit_counts.values()) or 1
    importance = visit_counts.get(passage_id, 0) / total_visits
    return 0.5 * (query_similarity + importance)


def rerank_passages_by_helpfulness(
    candidate_passages: Iterable[str],
    visit_counts: Dict[str, int],
    graph: nx.DiGraph,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Rank passages by helpfulness using traversal visit counts."""

    reranked: List[Tuple[str, float]] = []
    for pid in candidate_passages:
        node = graph.nodes.get(pid, {})
        query_sim = float(node.get("query_sim", 0.0))
        score = compute_helpfulness(pid, query_sim, visit_counts)
        reranked.append((pid, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
