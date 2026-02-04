"""Curve plot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

CONFIG = {
    "max_k": 20,
    "title": None,
    "figsize": (6, 4),
    "grid_alpha": 0.3,
    "dpi": 200,
    "gold_key": "gold_passages_full",
    "gold_fallback_key": "gold_passages",
}


def recall_at_k_curve(
    answers_path: str | Path,
    questions_path: str | Path,
    *,
    max_k: int | None = None,
    title: str | None = None,
    ax=None,
    savepath: str | None = None,
    show: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot recall@k for gold passage hits in selected sentences."""
    if max_k is None:
        max_k = CONFIG["max_k"]
    if title is None:
        title = CONFIG["title"]

    ks, recalls = recall_at_k_values(
        answers_path,
        questions_path,
        max_k=max_k,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    else:
        fig = ax.figure

    ax.plot(ks, recalls, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_xticks(ks)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=CONFIG["grid_alpha"])

    if savepath:
        fig.savefig(savepath, dpi=CONFIG["dpi"], bbox_inches="tight")
    if show:
        plt.show()
    return ks, recalls


def recall_at_k_values(
    answers_path: str | Path,
    questions_path: str | Path,
    *,
    max_k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (k_values, recall_values) arrays for recall@k."""
    if max_k is None:
        max_k = CONFIG["max_k"]

    gold_by_qid = _load_gold_passages(questions_path)
    first_hit_ranks = _first_hit_ranks(answers_path, gold_by_qid)
    total = len(first_hit_ranks)

    ks = np.arange(1, max_k + 1, dtype=int)
    recalls = np.zeros_like(ks, dtype=float)
    if total == 0:
        return ks, recalls

    for i, k in enumerate(ks):
        recalls[i] = sum(
            1 for r in first_hit_ranks if r is not None and r <= k
        ) / total
    return ks, recalls


def _load_gold_passages(questions_path: str | Path) -> Dict[str, List[str]]:
    gold_by_qid: Dict[str, List[str]] = {}
    with Path(questions_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid is None:
                continue
            gold = obj.get(CONFIG["gold_key"]) or obj.get(CONFIG["gold_fallback_key"]) or []
            gold_by_qid[qid] = list(gold)
    return gold_by_qid


def _first_hit_ranks(
    answers_path: str | Path,
    gold_by_qid: Dict[str, List[str]],
) -> List[int | None]:
    ranks: List[int | None] = []
    with Path(answers_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("question_id")
            gold_ids = gold_by_qid.get(qid, [])
            selected_ids = _selected_sentence_ids(obj)
            rank = _first_hit_rank(selected_ids, gold_ids)
            ranks.append(rank)
    return ranks


def _selected_sentence_ids(obj: Dict[str, object]) -> List[str]:
    selected = obj.get("selected_sentences")
    if isinstance(selected, list):
        return [str(s.get("sentence_id", "")) for s in selected if isinstance(s, dict)]
    used = obj.get("used_sentence_ids")
    if isinstance(used, list):
        return [str(s) for s in used]
    return []


def _first_hit_rank(selected_ids: Iterable[str], gold_ids: List[str]) -> int | None:
    if not gold_ids:
        return None
    for i, sid in enumerate(selected_ids, start=1):
        for gid in gold_ids:
            if sid == gid or sid.startswith(gid + "__"):
                return i
    return None
