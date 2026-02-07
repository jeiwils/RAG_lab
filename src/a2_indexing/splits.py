"""Utilities for carving validation splits from processed datasets."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.__utils__ import load_jsonl, processed_dataset_paths

__all__ = [
    "carve_validation_split",
]


DEFAULT_OUTPUT_KEYS: Tuple[str, ...] = (
    "questions",
    "passages",
    "passages_discourse_aware",
    "passages_discourse_aware_debug",
    "passages_discourse_aware_aliases",
    "full_passages",
    "full_passages_chunks",
    "full_passages_chunks_discourse_aware",
    "full_passages_chunks_discourse_aware_debug",
)


def _question_id_from_passage_id(
    passage_id: str,
    *,
    split_token: str | None = "__",
) -> str:
    if split_token and split_token in passage_id:
        return passage_id.split(split_token, 1)[0]
    return passage_id


def _extract_question_id(
    row: Dict,
    *,
    split_token: str | None = "__",
) -> str | None:
    qid = row.get("question_id")
    if qid:
        return str(qid)
    for key in ("passage_id", "chunk_id", "sentence_id", "dropped_id", "kept_id"):
        value = row.get(key)
        if value:
            return _question_id_from_passage_id(str(value), split_token=split_token)
    return None


def _resolve_output_path(path: Path, *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        return path
    base, ext = os.path.splitext(str(path))
    i = 1
    candidate = Path(f"{base}.v{i}{ext}")
    while candidate.exists():
        i += 1
        candidate = Path(f"{base}.v{i}{ext}")
    return candidate


def _write_split_jsonl(
    src_path: Path,
    *,
    train_out: Path,
    val_out: Path,
    train_ids: set[str],
    val_ids: set[str],
    split_token: str | None,
    id_fn=None,
    overwrite: bool = False,
) -> Tuple[Path, Path, int, int]:
    train_out = _resolve_output_path(train_out, overwrite=overwrite)
    val_out = _resolve_output_path(val_out, overwrite=overwrite)
    train_count = 0
    val_count = 0
    with open(train_out, "wt", encoding="utf-8") as f_train, open(
        val_out, "wt", encoding="utf-8"
    ) as f_val:
        for row in load_jsonl(str(src_path)):
            qid = id_fn(row) if id_fn is not None else _extract_question_id(
                row, split_token=split_token
            )
            if not qid:
                continue
            if qid in train_ids:
                f_train.write(json.dumps(row, ensure_ascii=False) + "\n")
                train_count += 1
            elif qid in val_ids:
                f_val.write(json.dumps(row, ensure_ascii=False) + "\n")
                val_count += 1
    return train_out, val_out, train_count, val_count


def carve_validation_split(
    dataset: str,
    *,
    source_split: str = "train",
    train_split: str | None = None,
    val_split: str = "val",
    val_ratio: float = 0.1,
    val_size: int | None = None,
    seed: int = 1,
    shuffle: bool = True,
    max_questions: int | None = None,
    passage_id_split_token: str | None = "__",
    include_outputs: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Create a validation split from an existing processed train split.

    This does *not* modify the source split. It writes new splits under
    ``data/processed_datasets/{dataset}/{train_split}/`` and
    ``data/processed_datasets/{dataset}/{val_split}/``.
    """
    # Musique passage IDs embed the question ID before the first "_sent".
    # Default "__" splitting would drop most passages from carved splits.
    if dataset == "musique" and passage_id_split_token in (None, "__"):
        passage_id_split_token = "_sent"
    src_paths = processed_dataset_paths(dataset, source_split)
    src_questions_path = src_paths["questions"]
    if not Path(src_questions_path).exists():
        raise FileNotFoundError(f"Missing questions file: {src_questions_path}")

    questions = list(load_jsonl(str(src_questions_path)))
    if isinstance(max_questions, int):
        questions = questions[:max_questions]

    question_ids = [
        str(q.get("question_id", ""))
        for q in questions
        if q.get("question_id") is not None
    ]
    question_ids = [qid for qid in question_ids if qid]

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(question_ids)

    if val_size is None:
        if val_ratio <= 0:
            val_size = 0
        else:
            val_size = int(round(len(question_ids) * val_ratio))
    val_size = min(max(val_size, 0), len(question_ids))

    val_ids = set(question_ids[:val_size])
    train_ids = set(question_ids[val_size:])

    if train_split is None:
        train_split = f"{source_split}_sub"

    train_paths = processed_dataset_paths(dataset, train_split)
    val_paths = processed_dataset_paths(dataset, val_split)

    if include_outputs is None:
        include_outputs = [
            key
            for key in DEFAULT_OUTPUT_KEYS
            if key in src_paths and Path(src_paths[key]).exists()
        ]

    results: Dict[str, Dict[str, str]] = {}

    # Always write questions first for visibility.
    if "questions" not in include_outputs:
        include_outputs = ["questions"] + list(include_outputs)

    for key in include_outputs:
        if key == "base":
            continue
        src_path = src_paths.get(key)
        if src_path is None or not Path(src_path).exists():
            continue
        train_out = train_paths[key]
        val_out = val_paths[key]

        if key == "questions":
            id_fn = lambda row: str(row.get("question_id", "")) or None
        else:
            id_fn = None

        train_out_path, val_out_path, train_count, val_count = _write_split_jsonl(
            Path(src_path),
            train_out=Path(train_out),
            val_out=Path(val_out),
            train_ids=train_ids,
            val_ids=val_ids,
            split_token=passage_id_split_token,
            id_fn=id_fn,
            overwrite=overwrite,
        )
        results[key] = {
            "train_path": str(train_out_path),
            "val_path": str(val_out_path),
            "train_rows": str(train_count),
            "val_rows": str(val_count),
        }

    results["summary"] = {
        "dataset": dataset,
        "source_split": source_split,
        "train_split": train_split,
        "val_split": val_split,
        "train_questions": str(len(train_ids)),
        "val_questions": str(len(val_ids)),
    }
    return results
