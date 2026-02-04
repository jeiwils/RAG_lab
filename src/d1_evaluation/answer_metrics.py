"""Answer evaluation helpers (normalization, EM, F1)."""

from __future__ import annotations

import re
import string

__all__ = [
    "normalise_answer",
    "compute_exact_match",
    "compute_f1",
    "evaluate_answers",
    "aggregate_answer_scores",
]


def normalise_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(pred: str, gold: str) -> int:
    """EM is 1 if normalized prediction == normalized gold, else 0."""
    return int(normalise_answer(pred) == normalise_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold after normalization."""
    pred_tokens = normalise_answer(pred).split()
    gold_tokens = normalise_answer(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answers(
    predictions: dict[str, str],
    gold_answers: dict[str, list[str]],
) -> dict[str, dict]:
    """Compute per-query EM and F1 scores."""
    results: dict[str, dict] = {}
    for qid, gold_list in gold_answers.items():
        pred = predictions.get(qid, "")
        em = max((compute_exact_match(pred, g) for g in gold_list), default=0)
        f1 = max((compute_f1(pred, g) for g in gold_list), default=0.0)
        results[qid] = {"prediction": pred, "em": em, "f1": f1}
    return results


def aggregate_answer_scores(predictions: dict, gold_answers: dict) -> dict:
    """Aggregate EM and F1 over a set of predicted answers (means in %)."""
    total = len(gold_answers)
    em_total = 0
    f1_total = 0.0

    for qid, gold_list in gold_answers.items():
        pred = predictions.get(qid, "")
        em = max(compute_exact_match(pred, g) for g in gold_list)
        f1 = max(compute_f1(pred, g) for g in gold_list)
        em_total += em
        f1_total += f1

    return {
        "mean_em": 100.0 * em_total / total,
        "mean_f1": 100.0 * f1_total / total,
    }
