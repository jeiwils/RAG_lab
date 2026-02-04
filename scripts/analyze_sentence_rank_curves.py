#!/usr/bin/env python
"""Analyze ranked sentence curves from DA_EXIT debug logs.

Computes:
- mean hits@k and recall@k over passage ids implied by ranked sentences
- mean/median sentence score by rank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from math import sqrt
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple
import re


_SENT_SUFFIX_RE = re.compile(r"__sent\d+$")
_CHUNK_SUFFIX_RE = re.compile(r"__chunk\d+$")


def _find_latest_debug(root: Path) -> Optional[Path]:
    candidates = list(root.rglob("reader_debug_*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _infer_dataset_split(
    debug_path: Path, first_record: Dict[str, object]
) -> Tuple[Optional[str], Optional[str]]:
    dataset = first_record.get("dataset") if first_record else None
    split = first_record.get("split") if first_record else None
    if dataset and split:
        return str(dataset), str(split)

    parts = list(debug_path.parts)
    if "results" in parts:
        try:
            idx = parts.index("results")
            dataset = parts[idx + 3]
            split = parts[idx + 4]
        except (IndexError, ValueError):
            pass
    return (str(dataset) if dataset else None, str(split) if split else None)


def _load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _questions_path(dataset: str, split: str) -> Path:
    return Path("data") / "processed_datasets" / dataset / split / "questions.jsonl"


def _load_gold_map(dataset: str, split: str) -> Dict[str, List[str]]:
    questions_path = _questions_path(dataset, split)
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")
    gold_map: Dict[str, List[str]] = {}
    for q in _load_jsonl(questions_path):
        qid = q.get("question_id")
        if not qid:
            continue
        gold = q.get("gold_passages_full") or q.get("gold_passages") or []
        gold_map[str(qid)] = list(gold)
    return gold_map


def _normalize_passage_id(value: str) -> str:
    base = _SENT_SUFFIX_RE.sub("", value)
    base = _CHUNK_SUFFIX_RE.sub("", base)
    return base


def extract_passage_ids_from_sentences(
    selected_sentences: Iterable[object],
) -> List[str]:
    passage_ids: List[str] = []
    seen = set()
    for item in selected_sentences:
        raw_id = ""
        if isinstance(item, dict):
            raw_id = str(item.get("chunk_id") or item.get("sentence_id") or "")
        else:
            raw_id = str(item)
        if not raw_id:
            continue
        pid = _normalize_passage_id(raw_id)
        if pid and pid not in seen:
            seen.add(pid)
            passage_ids.append(pid)
    return passage_ids


def compute_recall_at_k(pred_passages: List[str], gold_passages: List[str], k: int) -> float:
    if k <= 0 or not gold_passages:
        return 0.0
    gold_set = set(gold_passages)
    return len(set(pred_passages[:k]) & gold_set) / len(gold_passages)


def compute_hits_at_k(pred_passages: List[str], gold_passages: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold_passages)
    return float(any(pid in gold_set for pid in pred_passages[:k]))


def _safe_mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    return median(values) if values else None


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _sentence_passage_id(item: object) -> str:
    if isinstance(item, dict):
        raw = str(item.get("chunk_id") or item.get("sentence_id") or "")
    else:
        raw = str(item)
    if not raw:
        return ""
    return _normalize_passage_id(raw)


def _point_biserial(scores: List[float], labels: List[int]) -> Optional[float]:
    if not scores or len(scores) != len(labels):
        return None
    n = len(scores)
    n_pos = sum(1 for v in labels if v == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    mean_all = mean(scores)
    var = sum((x - mean_all) ** 2 for x in scores) / n
    if var == 0:
        return None
    s = sqrt(var)
    mean_pos = mean([s for s, y in zip(scores, labels) if y == 1])
    mean_neg = mean([s for s, y in zip(scores, labels) if y == 0])
    p = n_pos / n
    q = n_neg / n
    return (mean_pos - mean_neg) / s * sqrt(p * q)


def _auc(scores: List[float], labels: List[int]) -> Optional[float]:
    if not scores or len(scores) != len(labels):
        return None
    n_pos = sum(1 for v in labels if v == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0 for _ in scores]
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze DA_EXIT ranked sentence curves from debug JSONL."
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        help="Path to reader_debug_*.jsonl. Defaults to latest under data/results.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Max rank cutoff for curves (default: 20).",
    )
    parser.add_argument(
        "--score-field",
        type=str,
        default="score_normalized",
        help="Score field to use (default: score_normalized).",
    )
    parser.add_argument(
        "--include-empty-gold",
        action="store_true",
        help="Include questions with empty gold_passages in hits/recall averages.",
    )
    args = parser.parse_args()

    root = Path("data/results")
    debug_path = Path(args.debug) if args.debug else _find_latest_debug(root)
    if not debug_path or not debug_path.exists():
        print("No debug file found. Provide --debug or ensure data/results exists.")
        return 1

    first_record = None
    with open(debug_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                first_record = json.loads(line)
            except json.JSONDecodeError:
                continue
            break

    dataset, split = _infer_dataset_split(debug_path, first_record or {})
    if not dataset or not split:
        print(
            "Could not infer dataset/split from debug file. "
            "Pass a debug file under data/results/<model>/<dataset>/<split>/..."
        )
        return 1

    gold_map = _load_gold_map(dataset, split)

    max_k = max(args.k, 1)
    hits_sums = [0.0 for _ in range(max_k)]
    recall_sums = [0.0 for _ in range(max_k)]
    hits_counts = [0 for _ in range(max_k)]
    scores_by_rank: List[List[float]] = [[] for _ in range(max_k)]
    gold_by_rank: List[List[int]] = [[] for _ in range(max_k)]

    all_scores: List[float] = []
    all_labels: List[int] = []
    passage_scores: List[float] = []
    passage_labels: List[int] = []

    total_records = 0
    unique_questions = 0
    missing_ranked = 0
    used_selected_fallback = 0
    missing_gold = 0
    skipped_empty_gold = 0
    truncated_ranked = 0
    seen_qids = set()

    def iter_records(path: Path) -> Iterable[Dict[str, object]]:
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    for record in iter_records(debug_path):
        total_records += 1
        qid = record.get("question_id")
        if not qid:
            continue
        qid = str(qid)
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        unique_questions += 1

        ranked = record.get("ranked_sentences")
        if not ranked:
            missing_ranked += 1
            ranked = record.get("selected_sentences")
            if ranked:
                used_selected_fallback += 1
        if not ranked:
            continue
        ranked = list(ranked)
        if record.get("ranked_truncated"):
            truncated_ranked += 1

        gold = gold_map.get(qid)
        if gold is None:
            missing_gold += 1
            gold = []

        include_gold = True
        if not gold and not args.include_empty_gold:
            include_gold = False
            skipped_empty_gold += 1

        per_passage_best: Dict[str, float] = {}
        per_passage_label: Dict[str, int] = {}
        for i in range(min(max_k, len(ranked))):
            item = ranked[i]
            score = None
            if isinstance(item, dict):
                score = item.get(args.score_field)
                if score is None:
                    score = item.get("score_normalized", item.get("score", 0.0))
            try:
                score_val = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_val = None
            if score_val is not None:
                scores_by_rank[i].append(score_val)

            pid = _sentence_passage_id(item)
            label = 1 if pid in set(gold) else 0
            if score_val is not None:
                all_scores.append(score_val)
                all_labels.append(label)
                if pid:
                    best = per_passage_best.get(pid)
                    if best is None or score_val > best:
                        per_passage_best[pid] = score_val
                    per_passage_label[pid] = label
            gold_by_rank[i].append(label)

        for pid, score_val in per_passage_best.items():
            passage_scores.append(score_val)
            passage_labels.append(per_passage_label.get(pid, 0))

        if include_gold:
            for k in range(1, max_k + 1):
                prefix = ranked[:k]
                passage_ids = extract_passage_ids_from_sentences(prefix)
                hits = compute_hits_at_k(passage_ids, gold, k)
                recall = compute_recall_at_k(passage_ids, gold, k)
                hits_sums[k - 1] += hits
                recall_sums[k - 1] += recall
                hits_counts[k - 1] += 1

    print(f"Debug file: {debug_path}")
    print(f"Dataset/split: {dataset}/{split}")
    print(f"Records: {total_records}  Unique questions: {unique_questions}")
    print(f"Missing ranked_sentences: {missing_ranked}")
    if used_selected_fallback:
        print(
            f"Used selected_sentences as fallback: {used_selected_fallback} "
            "(these are post-tau/budget, not full ranking)"
        )
    print(f"Ranked truncated: {truncated_ranked}")
    print(f"Missing gold in questions: {missing_gold}")
    print(f"Skipped empty gold: {skipped_empty_gold}")
    print("")

    print("Hits/Recall@k (passage-level)")
    print("k  mean_hits  mean_recall  delta_recall  n")
    prev_recall = 0.0
    for k in range(1, max_k + 1):
        n = hits_counts[k - 1]
        if n == 0:
            mean_hits = None
            mean_recall = None
            delta_recall = None
        else:
            mean_hits = hits_sums[k - 1] / n
            mean_recall = recall_sums[k - 1] / n
            delta_recall = mean_recall - prev_recall
            prev_recall = mean_recall
        print(
            f"{k:<2} {_format_float(mean_hits):>9} {_format_float(mean_recall):>12} "
            f"{_format_float(delta_recall):>13} {n}"
        )

    print("")
    print("Score by rank")
    print("rank  mean_score  median_score  n")
    for i in range(max_k):
        values = scores_by_rank[i]
        m = _safe_mean(values)
        med = _safe_median(values)
        print(
            f"{i + 1:<4} {_format_float(m):>10} {_format_float(med):>13} {len(values)}"
        )

    print("")
    print("Gold rate by rank (sentence-level)")
    print("rank  gold_rate  n")
    for i in range(max_k):
        vals = gold_by_rank[i]
        if not vals:
            rate = None
        else:
            rate = sum(vals) / len(vals)
        print(f"{i + 1:<4} {_format_float(rate):>9} {len(vals)}")

    def _summary_block(label: str, scores: List[float], labels: List[int]) -> None:
        pos_scores = [s for s, y in zip(scores, labels) if y == 1]
        neg_scores = [s for s, y in zip(scores, labels) if y == 0]
        print(label)
        print(f"  n_total: {len(scores)}  n_pos: {len(pos_scores)}  n_neg: {len(neg_scores)}")
        print(f"  mean_score_pos: {_format_float(_safe_mean(pos_scores))}")
        print(f"  mean_score_neg: {_format_float(_safe_mean(neg_scores))}")
        print(f"  point_biserial_r: {_format_float(_point_biserial(scores, labels))}")
        print(f"  auc: {_format_float(_auc(scores, labels))}")

    print("")
    _summary_block("Sentence-level score vs gold", all_scores, all_labels)
    print("")
    _summary_block("Passage-level (max score per passage) vs gold", passage_scores, passage_labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
