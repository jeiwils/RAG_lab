"""Retrieval metrics (hits@k, recall@k, precision@k, nDCG)."""

from __future__ import annotations

import math
from typing import List

__all__ = [
    "compute_recall_at_k",
    "compute_hits_at_k",
    "compute_precision_at_k",
    "compute_dcg",
    "compute_ndcg",
]


def compute_recall_at_k(
    pred_passages: List[str], gold_passages: List[str], k: int
) -> float:
    """Return recall of top-``k`` predictions against gold passages.

    Parameters
    ----------
    pred_passages:
        Retrieved passages ordered by relevance.
    gold_passages:
        Gold passage identifiers.
    k:
        Evaluation cutoff.

    Returns
    -------
    float
        Fraction of ``gold_passages`` found among the first ``k`` predictions, or
        ``0.0`` when ``gold_passages`` is empty.
    """

    if k <= 0 or not gold_passages:
        return 0.0

    gold_set = set(gold_passages)
    return len(set(pred_passages[:k]) & gold_set) / len(gold_passages)


def compute_hits_at_k(pred_passages: List[str], gold_passages: List[str], k: int) -> float:
    """Return whether any of the top-``k`` predicted passages match a gold passage.

    Parameters
    ----------
    pred_passages:
        Retrieved passages ordered by relevance.
    gold_passages:
        Gold passage identifiers.
    k:
        Evaluation cutoff.

    Returns
    -------
    float
        ``1.0`` if any gold passage is found in the first ``k`` predictions,
        otherwise ``0.0``.
    """

    if k <= 0:
        return 0.0

    gold_set = set(gold_passages)
    return float(any(pid in gold_set for pid in pred_passages[:k]))


def compute_precision_at_k(
    pred_passages: List[str],
    gold_passages: List[str],
    k: int,
) -> float:
    """Return precision of top-``k`` predictions against gold passages.

    Parameters
    ----------
    pred_passages:
        Retrieved passages ordered by relevance.
    gold_passages:
        Gold passage identifiers.
    k:
        Evaluation cutoff.

    Returns
    -------
    float
        Fraction of the first ``k`` predictions that are gold passages, or
        ``0.0`` when there are no predictions.
    """

    if k <= 0:
        return 0.0
    top_k = pred_passages[:k]
    if not top_k:
        return 0.0
    gold_set = set(gold_passages)
    hits = len(set(top_k) & gold_set)
    return hits / len(set(top_k))


def compute_dcg(relevances: List[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain (DCG).

    Parameters
    ----------
    relevances:
        Relevance scores ordered by predicted rank.
    k:
        Cutoff for evaluation; when ``None`` uses the full list.
    """
    if not relevances:
        return 0.0
    if k is None or k <= 0:
        k = len(relevances)
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k]):
        gain = (2 ** float(rel)) - 1.0
        denom = math.log2(idx + 2)
        dcg += gain / denom
    return dcg


def compute_ndcg(relevances: List[float], k: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG)."""
    if not relevances:
        return 0.0
    if k is None or k <= 0:
        k = len(relevances)
    dcg = compute_dcg(relevances, k=k)
    ideal = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal, k=k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg
