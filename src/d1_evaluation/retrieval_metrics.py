"""Retrieval metrics (hits@k, recall@k)."""

from __future__ import annotations

from typing import List

__all__ = ["compute_recall_at_k", "compute_hits_at_k"]


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
