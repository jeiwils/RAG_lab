"""Classification metrics helpers."""

from __future__ import annotations

from typing import Sequence

__all__ = ["binary_f1"]


def _as_bool_array(values):
    if hasattr(values, "bool"):
        return values.bool()
    if hasattr(values, "astype"):
        return values.astype(bool)
    if isinstance(values, Sequence):
        return [bool(v) for v in values]
    return [bool(v) for v in list(values)]


def _sum_values(values) -> float:
    total = values.sum() if hasattr(values, "sum") else sum(values)
    return float(total.item() if hasattr(total, "item") else total)


def binary_f1(preds, labels) -> float:
    """Compute F1 score for binary predictions and labels."""
    preds_bool = _as_bool_array(preds)
    labels_bool = _as_bool_array(labels)

    if isinstance(preds_bool, list):
        if len(preds_bool) != len(labels_bool):
            raise ValueError("preds and labels must have the same length")
        tp = sum(p and l for p, l in zip(preds_bool, labels_bool))
        fp = sum(p and not l for p, l in zip(preds_bool, labels_bool))
        fn = sum((not p) and l for p, l in zip(preds_bool, labels_bool))
    else:
        tp = _sum_values(preds_bool & labels_bool)
        fp = _sum_values(preds_bool & ~labels_bool)
        fn = _sum_values((~preds_bool) & labels_bool)

    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
