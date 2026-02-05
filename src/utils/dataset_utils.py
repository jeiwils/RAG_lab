"""Dataset utilities for loading processed QA data and building examples."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypeVar

from src.a1_ingestion.dataset_preprocessing_functions import split_text_into_sentences
from src.utils.__utils__ import clean_text, load_jsonl, processed_dataset_paths

__all__ = [
    "build_training_examples",
    "extract_supporting_fact_sentences",
    "load_dataset_split",
    "sample_non_supporting_sentences_same_docs",
    "sample_sentences_unrelated_docs",
]

T = TypeVar("T")

DEFAULT_HARD_NEGATIVES = int(os.environ.get("USEFUL_SENT_HARD_NEGATIVES", "4"))
DEFAULT_RANDOM_NEGATIVES = int(os.environ.get("USEFUL_SENT_RANDOM_NEGATIVES", "4"))


def sample_without_replacement(
    candidates: Sequence[T],
    n: int,
    rng: random.Random,
) -> List[T]:
    if n <= 0 or not candidates:
        return []
    if n >= len(candidates):
        return list(candidates)
    return rng.sample(list(candidates), n)


def _question_id_from_passage_id(
    passage_id: str, *, split_token: str | None = "__"
) -> str:
    if split_token and split_token in passage_id:
        return passage_id.split(split_token, 1)[0]
    return passage_id


def _load_alias_map(path: str | Path) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for row in load_jsonl(str(path)):
        dropped_id = row.get("dropped_id")
        kept_id = row.get("kept_id")
        if dropped_id and kept_id:
            alias_map[str(dropped_id)] = str(kept_id)
    return alias_map


def _resolve_aliases_path(
    passages_path: str | Path,
    *,
    dataset: str,
    split: str,
) -> Path | None:
    candidate = Path(passages_path).with_name("passages_discourse_aware_aliases.jsonl")
    if "passages_discourse_aware" in Path(passages_path).name and candidate.exists():
        return candidate
    paths = processed_dataset_paths(dataset, split)
    std_aliases = paths.get("passages_discourse_aware_aliases")
    if std_aliases is None:
        return None
    std_aliases = Path(std_aliases)
    if "passages_discourse_aware" in str(passages_path) and std_aliases.exists():
        return std_aliases
    return None


def load_dataset_split(
    split: str,
    *,
    dataset: str,
    questions_path: str | Path | None = None,
    passages_path: str | Path | None = None,
    passages_aliases_path: str | Path | None = None,
    max_records: int | None = None,
    question_id_col: str = "question_id",
    question_col: str = "question",
    gold_passages_col: str = "gold_passages",
    passage_id_col: str = "passage_id",
    passage_title_col: str = "title",
    passage_text_col: str = "text",
    passage_question_id_col: str | None = None,
    passage_id_split_token: str | None = "__",
) -> List[Dict[str, Any]]:
    """Load processed dataset questions/passages and attach passages per question."""
    if questions_path is None or passages_path is None:
        paths = processed_dataset_paths(dataset, split)
        if questions_path is None:
            questions_path = paths["questions"]
        if passages_path is None:
            passages_path = paths["passages"]

    alias_map: Dict[str, str] = {}
    if passages_aliases_path is None and passages_path is not None:
        passages_aliases_path = _resolve_aliases_path(
            passages_path, dataset=dataset, split=split
        )
    if passages_aliases_path is not None and Path(passages_aliases_path).exists():
        alias_map = _load_alias_map(passages_aliases_path)

    questions = list(load_jsonl(str(questions_path)))
    if isinstance(max_records, int):
        questions = questions[:max_records]

    question_ids = {
        str(q.get(question_id_col, ""))
        for q in questions
        if q.get(question_id_col)
    }
    passages_by_qid: Dict[str, List[Dict[str, str]]] = {qid: [] for qid in question_ids}

    for passage in load_jsonl(str(passages_path)):
        pid = str(passage.get(passage_id_col, ""))
        if not pid:
            continue
        if passage_question_id_col:
            qid = str(passage.get(passage_question_id_col, ""))
        else:
            qid = _question_id_from_passage_id(pid, split_token=passage_id_split_token)
        if qid not in question_ids:
            continue
        text = str(passage.get(passage_text_col, "")).strip()
        if not text:
            continue
        passages_by_qid[qid].append(
            {
                "passage_id": pid,
                "title": str(passage.get(passage_title_col, "")),
                "text": text,
            }
        )

    records: List[Dict[str, Any]] = []
    for q in questions:
        qid = str(q.get(question_id_col, ""))
        if not qid:
            continue
        question = str(q.get(question_col, ""))
        gold_raw = list(q.get(gold_passages_col) or [])
        if alias_map:
            seen: set[str] = set()
            gold_passages: List[str] = []
            for pid in gold_raw:
                mapped = alias_map.get(str(pid), str(pid))
                if mapped in seen:
                    continue
                seen.add(mapped)
                gold_passages.append(mapped)
        else:
            gold_passages = gold_raw
        records.append(
            {
                "question_id": qid,
                "question": question,
                "gold_passages": gold_passages,
                "passages": passages_by_qid.get(qid, []),
            }
        )

    return records


def extract_supporting_fact_sentences(record: Dict[str, Any]) -> List[str]:
    """Return sentences labeled as supporting facts."""
    context_map = {
        str(p.get("passage_id")): str(p.get("text", ""))
        for p in record.get("passages", [])
    }
    out: List[str] = []
    seen: set[str] = set()
    for passage_id in record.get("gold_passages", []):
        pid = str(passage_id)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        sent = clean_text(context_map.get(pid, ""))
        if sent:
            out.append(sent)
    return out


def _collect_context_passages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    collected: List[Dict[str, str]] = []
    for item in record.get("passages", []):
        pid = str(item.get("passage_id", ""))
        if not pid:
            continue
        text = clean_text(str(item.get("text", "")))
        if not text:
            continue
        collected.append(
            {
                "passage_id": pid,
                "title": str(item.get("title", "")),
                "text": text,
            }
        )
    return collected


def sample_non_supporting_sentences_same_docs(
    record: Dict[str, Any],
    *,
    n: int,
    rng: random.Random,
    exclude: set[str] | None = None,
) -> List[Dict[str, str]]:
    """Sample negative passages from the same question's context.

    This keeps negatives tied to the same question as the positives by drawing
    from the passages already attached to that record.
    """
    exclude = exclude or set()
    support_ids = {str(pid) for pid in record.get("gold_passages", [])}
    candidates = [
        passage
        for passage in _collect_context_passages(record)
        if passage["passage_id"] not in support_ids
        and passage["passage_id"] not in exclude
    ]
    return sample_without_replacement(candidates, n, rng)


def sample_sentences_unrelated_docs(
    record_id: str,
    global_pool: Sequence[Tuple[str, Dict[str, str]]],
    *,
    n: int,
    rng: random.Random,
    exclude: set[str] | None = None,
) -> List[Dict[str, str]]:
    """Sample negative passages from other records (different questions)."""
    exclude = exclude or set()
    candidates = [
        passage
        for rid, passage in global_pool
        if rid != record_id and passage["passage_id"] not in exclude
    ]
    return sample_without_replacement(candidates, n, rng)


def build_training_examples(
    hotpot_records: Sequence[Dict[str, Any]],
    *,
    hard_negatives: int = DEFAULT_HARD_NEGATIVES,
    random_negatives: int = DEFAULT_RANDOM_NEGATIVES,
    seed: int | None = None,
    dataset: str | None = None,
    split: str | None = None,
    full_passages_chunks_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Build sentence-level training examples from processed records.

    Positives come from `gold_passages` for each question. Hard negatives are
    sampled from the same question's attached passages; random negatives are
    sampled from passages belonging to other questions.
    """
    rng = random.Random(seed)
    context_lookup: Dict[str, str] = {}
    if dataset and split:
        try:
            context_lookup = _build_sentence_context_lookup(
                dataset=dataset,
                split=split,
                full_passages_chunks_path=full_passages_chunks_path,
            )
        except FileNotFoundError:
            context_lookup = {}

    record_ids: List[str] = []
    record_passage_pool: List[List[Dict[str, str]]] = []
    for idx, record in enumerate(hotpot_records):
        record_id = str(record.get("question_id") or record.get("id") or idx)
        record_ids.append(record_id)
        record_passage_pool.append(_collect_context_passages(record))

    global_pool = [
        (rid, passage)
        for rid, passages in zip(record_ids, record_passage_pool)
        for passage in passages
    ]

    examples: List[Dict[str, Any]] = []
    for record, record_id in zip(hotpot_records, record_ids):
        question = clean_text(str(record.get("question", "")))
        if not question:
            continue

        passage_map = {
            str(p.get("passage_id")): str(p.get("text", ""))
            for p in record.get("passages", [])
        }
        pos_ids = {str(pid) for pid in record.get("gold_passages", [])}

        hard_neg_sents = sample_non_supporting_sentences_same_docs(
            record,
            n=hard_negatives,
            rng=rng,
            exclude=pos_ids,
        )
        hard_ids = {p["passage_id"] for p in hard_neg_sents}

        rand_neg_sents = sample_sentences_unrelated_docs(
            record_id,
            global_pool,
            n=random_negatives,
            rng=rng,
            exclude=pos_ids | hard_ids,
        )

        for pid in pos_ids:
            sent = clean_text(passage_map.get(pid, ""))
            if not sent:
                continue
            context = context_lookup.get(pid, sent)
            examples.append(
                {
                    "query": question,
                    "context": context,
                    "sentence": sent,
                    "label": 1,
                    "label_type": "pos",
                }
            )
        for sent in hard_neg_sents:
            context = context_lookup.get(sent["passage_id"], sent["text"])
            examples.append(
                {
                    "query": question,
                    "context": context,
                    "sentence": sent["text"],
                    "label": 0,
                    "label_type": "hard",
                }
            )
        for sent in rand_neg_sents:
            context = context_lookup.get(sent["passage_id"], sent["text"])
            examples.append(
                {
                    "query": question,
                    "context": context,
                    "sentence": sent["text"],
                    "label": 0,
                    "label_type": "random",
                }
            )

    rng.shuffle(examples)
    return examples


@lru_cache(maxsize=8)
def _build_sentence_context_lookup(
    *,
    dataset: str,
    split: str,
    full_passages_chunks_path: str | Path | None = None,
) -> Dict[str, str]:
    """Map sentence-level passage IDs to their full chunk context text."""
    if full_passages_chunks_path is None:
        paths = processed_dataset_paths(dataset, split)
        full_passages_chunks_path = paths["full_passages_chunks"]

    chunks_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in load_jsonl(str(full_passages_chunks_path)):
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            continue
        chunks_by_source.setdefault(source_id, []).append(row)

    lookup: Dict[str, str] = {}
    for source_id, chunks in chunks_by_source.items():
        chunks.sort(key=lambda x: int(x.get("chunk_index", 0)))
        sent_idx = 0
        for chunk in chunks:
            chunk_text = clean_text(str(chunk.get("text", "")))
            if not chunk_text:
                continue
            for sentence in split_text_into_sentences(chunk_text):
                sentence_id = f"{source_id}__sent{sent_idx}"
                if sentence_id not in lookup:
                    lookup[sentence_id] = chunk_text
                sent_idx += 1

    return lookup
