"""Utilities for processing raw QA datasets into a unified format.

This module exposes a single :func:`process_dataset` helper which converts a
dataset's raw examples into the project wide question and passage JSONL files.
Callers must supply a ``field_map`` describing how to extract the necessary
fields from each example.  Previous wrappers like ``process_hotpotqa`` and the
``PROCESSORS`` registry have been removed so that new datasets can be processed
without modifying this module.

Example
-------

To process a HotpotQA style JSON file::

    from src.a_dataset_preprocessing import process_dataset
    from src.utils import pid_plus_title

    field_map = {
        "get_id": lambda ex: ex["_id"],
        "get_question": lambda ex: ex["question"],
        "get_answer": lambda ex: ex.get("answer", ""),
        "iter_passages": lambda ex: [
            (pid_plus_title(ex["_id"], title, i), title, sent)
            for title, sents in ex["context"]
            for i, sent in enumerate(sents)
        ],
        "gold_passage_ids": lambda ex: [
            pid_plus_title(ex["_id"], title, idx)
            for title, idx in ex.get("supporting_facts", [])
        ],
    }

    process_dataset(
        dataset="hotpotqa",
        split="dev",
        file_path="data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json",
        field_map=field_map,
    )

The ``field_map`` supplies callables to extract the question id, question
text, gold answer, passage iterator and gold passage IDs.  The processing logic
is otherwise dataset agnostic.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

from src.a1_ingestion.chunking import (
    split_long_passage,
    split_long_passage_discourse_aware,
    split_long_passage_discourse_aware_with_flags,
)
from src.utils.algorithms.discourse_aware import (
    _expand_sentence_text,
    _has_anaphora_marker,
    _has_cataphora_marker,
    iter_sentence_spans,
)
from src.utils.__utils__ import (
    append_jsonl,
    clean_text,
    compute_resume_sets,
    existing_ids,
    pid_plus_title,
    pid_plus_title_full,
    processed_dataset_paths,
)

__all__ = [
    "FieldMap",
    "DATASET_CONFIGS",
    "get_raw_dataset_path",
    "process_dataset",
    "_run_ingestion",
    "sentence_ids_from_full",
    "sentence_passages_from_full",
    "split_text_into_sentences",
]

def _strip_full_suffix(passage_id: str) -> str:
    if passage_id.endswith("_full"):
        return passage_id[: -len("_full")]
    if passage_id.endswith("__full"):
        return passage_id[: -len("__full")]
    return passage_id


def split_text_into_sentences(text: str) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = [s.strip() for _, _, s in iter_sentence_spans(cleaned)]
    return [s for s in sentences if s]


def sentence_passages_from_full(
    passage_id: str,
    title: str,
    text: str,
) -> List[Tuple[str, str, str]]:
    sentences = split_text_into_sentences(text)
    return [
        (f"{passage_id}__sent{idx}", title, sent)
        for idx, sent in enumerate(sentences)
    ]


def sentence_ids_from_full(passage_id: str, text: str) -> List[str]:
    return [
        f"{passage_id}__sent{idx}"
        for idx, _sent in enumerate(split_text_into_sentences(text))
    ]

##### Dataset configs
"""Dataset-specific file paths and field maps for ingestion."""

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hotpotqa": {
        "file_path": lambda split: (
            "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
            if split == "train"
            else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        ),
        "field_map": {
            "get_id": lambda ex: ex["_id"],
            "get_question": lambda ex: ex["question"],
            "get_answer": lambda ex: ex.get("answer", ""),
            "iter_passages": lambda ex: [
                (pid_plus_title(ex["_id"], title, i), title, sent)
                for title, sents in ex["context"]
                for i, sent in enumerate(sents)
            ],
            "iter_full_passages": lambda ex: [
                (pid_plus_title_full(ex["_id"], title), title, " ".join(sents))
                for title, sents in ex["context"]
            ],
            "gold_passage_ids": lambda ex: [
                pid_plus_title(ex["_id"], title, idx)
                for title, idx in ex.get("supporting_facts", [])
            ],
            "gold_full_passage_ids": lambda ex: [
                pid_plus_title_full(ex["_id"], title)
                for title, _idx in ex.get("supporting_facts", [])
            ],
        },
    },
    "2wikimultihopqa": {
        "file_path": lambda split: f"data/raw_datasets/2wikimultihopqa/{split}.json",
        "field_map": {
            "get_id": lambda ex: ex["_id"],
            "get_question": lambda ex: ex["question"],
            "get_answer": lambda ex: ex.get("answer", ""),
            "iter_passages": lambda ex: [
                (pid_plus_title(ex["_id"], title, i), title, sent)
                for title, sents in ex["context"]
                for i, sent in enumerate(sents)
            ],
            "iter_full_passages": lambda ex: [
                (pid_plus_title_full(ex["_id"], title), title, " ".join(sents))
                for title, sents in ex["context"]
            ],
            "gold_passage_ids": lambda ex: [
                pid_plus_title(ex["_id"], title, idx)
                for title, idx in ex.get("supporting_facts", [])
            ],
            "gold_full_passage_ids": lambda ex: [
                pid_plus_title_full(ex["_id"], title)
                for title, _idx in ex.get("supporting_facts", [])
            ],
        },
    },
    "musique": {
        "file_path": lambda split: f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl",
        "field_map": {
            "get_id": lambda ex: ex["id"],
            "get_question": lambda ex: ex.get("question", ""),
            "get_answer": lambda ex: ex.get("answer", ""),
            "iter_passages": lambda ex: [
                item
                for i, p in enumerate(ex.get("paragraphs", []))
                for item in sentence_passages_from_full(
                    f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                    p.get("title", ""),
                    p.get("paragraph_text", ""),
                )
            ],
            "iter_full_passages": lambda ex: [
                (
                    f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                    p.get("title", ""),
                    p.get("paragraph_text", ""),
                )
                for i, p in enumerate(ex.get("paragraphs", []))
            ],
            "gold_passage_ids": lambda ex: [
                pid
                for i, p in enumerate(ex.get("paragraphs", []))
                if p.get("is_supporting")
                for pid in sentence_ids_from_full(
                    f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                    p.get("paragraph_text", ""),
                )
            ],
            "gold_full_passage_ids": lambda ex: [
                f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}"
                for i, p in enumerate(ex.get("paragraphs", []))
                if p.get("is_supporting")
            ],
        },
    },
}


def get_raw_dataset_path(dataset: str, split: str) -> str:
    """Return the raw dataset file path for ``dataset`` and ``split``."""
    config = DATASET_CONFIGS.get(dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    file_path = config.get("file_path")
    if callable(file_path):
        return file_path(split)
    if not file_path:
        raise ValueError(f"Missing file_path for dataset: {dataset}")
    return str(file_path)

##### Types
"""Shared ingestion type aliases."""

FieldMap = Dict[str, Callable[[Dict], Iterable]] #### ?????

##### Generic dataset processing
"""Generic utilities for mapping raw datasets into a unified JSONL format."""


def process_dataset(
    *,
    dataset: str,
    split: str,
    file_path: str,
    field_map: FieldMap,
    max_examples: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    """Process ``file_path`` using ``field_map``.

    Parameters
    ----------
    dataset:
        Name of the dataset being processed.
    split:
        Dataset split (``train``, ``dev`` ...).
    file_path:
        Path to the raw dataset file.  JSON or JSONL files are supported.
    field_map:
        Mapping of callables that extract fields from each example.  Required
        keys are ``get_id``, ``get_question``, ``get_answer``,
        ``iter_passages`` and ``gold_passage_ids``. Optional keys
        ``iter_full_passages`` and ``gold_full_passage_ids`` allow writing a
        separate full-passage JSONL alongside sentence-level passages.
        The callables operate on a single example and either return a value or
        an iterable of values.
    max_examples:
        Optional limit for the number of examples processed.
    overwrite:
        Unused but kept for backward compatibility.
    resume:
        Whether to resume from existing processed files.
    """

    # ---- Load raw examples -------------------------------------------------
    examples: List[Dict]
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            examples = []
            for i, line in enumerate(f):
                if isinstance(max_examples, int) and i >= max_examples:
                    break
                examples.append(json.loads(line))
        else:
            examples = json.load(f)
            if isinstance(max_examples, int):
                examples = examples[:max_examples]

    paths = processed_dataset_paths(dataset, split)
    qa_path = str(paths["questions"])
    passages_path = str(paths["passages"])

    get_id = field_map["get_id"]
    get_question = field_map["get_question"]
    get_answer = field_map.get("get_answer", lambda ex: "")
    iter_passages_fn = field_map["iter_passages"]
    iter_full_passages_fn = field_map.get("iter_full_passages")
    gold_ids_fn = field_map.get("gold_passage_ids", lambda ex: [])
    gold_full_ids_fn = field_map.get("gold_full_passage_ids")

    # ---- Determine resume state --------------------------------------------
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: get_id(ex),
        phase_label=f"{dataset} {split} questions",
        id_field="question_id",
    )

    def iter_pids() -> Iterable[str]:
        for ex in examples:
            for pid, _title, _text in iter_passages_fn(ex):
                yield pid

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"{dataset} {split} passages",
        id_field="passage_id",
    )

    done_full_pids: set[str] = set()
    done_full_chunk_pids: set[str] = set()
    done_full_da_chunk_pids: set[str] = set()
    done_discourse_pids: set[str] = set()
    done_discourse_debug_pids: set[str] = set()
    done_discourse_aliases: set[str] = set()
    done_full_da_chunk_debug_pids: set[str] = set()
    full_passages_path: str | None = None
    full_passages_chunks_path: str | None = None
    full_passages_chunks_discourse_aware_path: str | None = None
    passages_discourse_aware_path: str | None = None
    passages_discourse_aware_debug_path: str | None = None
    passages_discourse_aware_aliases_path: str | None = None
    full_passages_chunks_discourse_aware_debug_path: str | None = None
    if iter_full_passages_fn is not None:
        full_passages_path = str(paths["full_passages"])
        full_passages_chunks_path = str(paths["full_passages_chunks"])
        full_passages_chunks_discourse_aware_path = str(
            paths["full_passages_chunks_discourse_aware"]
        )
        passages_discourse_aware_path = str(paths["passages_discourse_aware"])
        passages_discourse_aware_debug_path = str(
            paths["passages_discourse_aware_debug"]
        )
        passages_discourse_aware_aliases_path = str(
            paths["passages_discourse_aware_aliases"]
        )
        full_passages_chunks_discourse_aware_debug_path = str(
            paths["full_passages_chunks_discourse_aware_debug"]
        )

        def iter_full_pids() -> Iterable[str]:
            for ex in examples:
                for pid, _title, _text in iter_full_passages_fn(ex):
                    yield pid

        done_full_pids, _ = compute_resume_sets(
            resume=resume,
            out_path=full_passages_path,
            items=iter_full_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{dataset} {split} full passages",
            id_field="passage_id",
        )

        def iter_full_chunk_pids() -> Iterable[str]:
            for ex in examples:
                for pid, _title, text in iter_full_passages_fn(ex):
                    cleaned = clean_text(text)
                    chunks = split_long_passage(cleaned)
                    if not chunks:
                        continue
                    if len(chunks) == 1:
                        yield pid
                    else:
                        for idx in range(len(chunks)):
                            yield f"{pid}__chunk{idx}"

        done_full_chunk_pids, _ = compute_resume_sets(
            resume=resume,
            out_path=full_passages_chunks_path,
            items=iter_full_chunk_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{dataset} {split} full passage chunks",
            id_field="passage_id",
        )

        def iter_full_da_chunk_pids() -> Iterable[str]:
            for ex in examples:
                for pid, _title, text in iter_full_passages_fn(ex):
                    cleaned = clean_text(text)
                    chunks = split_long_passage_discourse_aware(cleaned)
                    if not chunks:
                        continue
                    if len(chunks) == 1:
                        yield pid
                    else:
                        for idx in range(len(chunks)):
                            yield f"{pid}__chunk{idx}"

        done_full_da_chunk_pids, _ = compute_resume_sets(
            resume=resume,
            out_path=full_passages_chunks_discourse_aware_path,
            items=iter_full_da_chunk_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{dataset} {split} discourse-aware full passage chunks",
            id_field="passage_id",
        )

        def iter_discourse_pids() -> Iterable[str]:
            for ex in examples:
                for pid, _title, text in iter_full_passages_fn(ex):
                    cleaned = clean_text(text)
                    sentences = [s.strip() for _, _, s in iter_sentence_spans(cleaned)]
                    if not sentences:
                        continue
                    expanded_entries: List[Dict[str, object]] = []
                    covered_by_expansion: set[int] = set()
                    for idx, sent in enumerate(sentences):
                        has_anaphora = _has_anaphora_marker(
                            sent,
                            include_reformulation=True,
                            include_parenthetical=True,
                        )
                        has_cataphora = _has_cataphora_marker(
                            sent,
                            include_reformulation=True,
                            require_forward_punct=True,
                        )
                        _expanded_text, span_start, span_end = _expand_sentence_text(
                            sentences,
                            idx,
                            extension=1,
                            has_anaphora=has_anaphora,
                            has_cataphora=has_cataphora,
                        )
                        expanded = span_start != span_end
                        if expanded:
                            covered_by_expansion.update(range(span_start, span_end + 1))
                        expanded_entries.append(
                            {
                                "idx": idx,
                                "expanded": expanded,
                                "span_start": span_start,
                                "span_end": span_end,
                            }
                        )

                    for entry in expanded_entries:
                        idx = int(entry["idx"])
                        expanded = bool(entry["expanded"])
                        base = _strip_full_suffix(pid)
                        yield f"{base}__sent{idx}"

        done_discourse_pids, _ = compute_resume_sets(
            resume=resume,
            out_path=passages_discourse_aware_path,
            items=iter_discourse_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{dataset} {split} discourse-aware passages",
            id_field="passage_id",
        )
        if passages_discourse_aware_debug_path is not None:
            done_discourse_debug_pids = (
                existing_ids(passages_discourse_aware_debug_path, id_field="passage_id")
                if resume
                else set()
            )
        if passages_discourse_aware_aliases_path is not None:
            done_discourse_aliases = (
                existing_ids(
                    passages_discourse_aware_aliases_path,
                    id_field="dropped_id",
                )
                if resume
                else set()
            )
        if full_passages_chunks_discourse_aware_debug_path is not None:
            done_full_da_chunk_debug_pids = (
                existing_ids(
                    full_passages_chunks_discourse_aware_debug_path,
                    id_field="passage_id",
                )
                if resume
                else set()
            )

    # ---- Write processed files ---------------------------------------------
    for ex in examples:
        qid = get_id(ex)
        if qid not in done_qids:
            gold_ids, seen = [], set()
            for pid in gold_ids_fn(ex):
                if pid not in seen:
                    gold_ids.append(pid)
                    seen.add(pid)
            record = {
                "question_id": qid,
                "dataset": dataset,
                "split": split,
                "question": clean_text(get_question(ex)),
                "gold_answer": clean_text(get_answer(ex)),
                "gold_passages": gold_ids,
            }
            if gold_full_ids_fn is not None:
                gold_full_ids, seen_full = [], set()
                for pid in gold_full_ids_fn(ex):
                    if pid not in seen_full:
                        gold_full_ids.append(pid)
                        seen_full.add(pid)
                record["gold_passages_full"] = gold_full_ids
            append_jsonl(qa_path, record)

        for pid, title, text in iter_passages_fn(ex):
            if pid in done_pids:
                continue
            append_jsonl(
                passages_path,
                {
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(text),
                },
            )

        if iter_full_passages_fn is None or full_passages_path is None:
            continue
        for pid, title, text in iter_full_passages_fn(ex):
            cleaned = clean_text(text)
            if pid not in done_full_pids:
                append_jsonl(
                    full_passages_path,
                    {
                        "passage_id": pid,
                        "title": title,
                        "text": cleaned,
                    },
                )
            if passages_discourse_aware_path is not None:
                sentences = [s.strip() for _, _, s in iter_sentence_spans(cleaned)]
                expanded_entries: List[Dict[str, object]] = []
                covered_by_expansion: set[int] = set()
                cover_anchor: Dict[int, int] = {}
                for idx, sent in enumerate(sentences):
                    has_anaphora = _has_anaphora_marker(
                        sent,
                        include_reformulation=True,
                        include_parenthetical=True,
                    )
                    has_cataphora = _has_cataphora_marker(
                        sent,
                        include_reformulation=True,
                        require_forward_punct=True,
                    )
                    expanded_text, span_start, span_end = _expand_sentence_text(
                        sentences,
                        idx,
                        extension=1,
                        has_anaphora=has_anaphora,
                        has_cataphora=has_cataphora,
                    )
                    expanded = span_start != span_end
                    if expanded:
                        covered_by_expansion.update(range(span_start, span_end + 1))
                        for span_idx in range(span_start, span_end + 1):
                            cover_anchor.setdefault(span_idx, idx)
                    expanded_entries.append(
                        {
                            "idx": idx,
                            "expanded": expanded,
                            "span_start": span_start,
                            "span_end": span_end,
                            "text": expanded_text.strip(),
                        }
                    )

                base = _strip_full_suffix(pid)
                for entry in expanded_entries:
                    idx = int(entry["idx"])
                    expanded = bool(entry["expanded"])
                    passage_id = f"{base}__sent{idx}"
                    anchor_idx = cover_anchor.get(idx)
                    if not expanded and anchor_idx is not None and anchor_idx != idx:
                        if passages_discourse_aware_aliases_path is not None:
                            if passage_id not in done_discourse_aliases:
                                kept_id = f"{base}__sent{anchor_idx}"
                                append_jsonl(
                                    passages_discourse_aware_aliases_path,
                                    {
                                        "dropped_id": passage_id,
                                        "kept_id": kept_id,
                                        "source_id": pid,
                                        "dropped_idx": idx,
                                        "anchor_idx": anchor_idx,
                                    },
                                )
                                done_discourse_aliases.add(passage_id)
                        continue
                    if passage_id in done_discourse_pids:
                        continue
                    append_jsonl(
                        passages_discourse_aware_path,
                        {
                            "passage_id": passage_id,
                            "source_id": pid,
                            "expanded": expanded,
                            "title": title,
                            "text": str(entry["text"]),
                        },
                    )
                    done_discourse_pids.add(passage_id)
                    if (
                        expanded
                        and passages_discourse_aware_debug_path is not None
                        and passage_id not in done_discourse_debug_pids
                    ):
                        append_jsonl(
                            passages_discourse_aware_debug_path,
                            {
                                "passage_id": passage_id,
                                "source_id": pid,
                                "sent_idx": idx,
                                "span_start": int(entry["span_start"]),
                                "span_end": int(entry["span_end"]),
                                "expanded": True,
                                "title": title,
                                "text": str(entry["text"]),
                            },
                        )
                        done_discourse_debug_pids.add(passage_id)
            if full_passages_chunks_path is None:
                continue
            chunks = split_long_passage(cleaned)
            if not chunks:
                continue
            if len(chunks) == 1:
                chunk_id = pid
                if chunk_id in done_full_chunk_pids:
                    continue
                append_jsonl(
                    full_passages_chunks_path,
                    {
                        "passage_id": chunk_id,
                        "source_id": pid,
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "title": title,
                        "text": chunks[0],
                    },
                )
            else:
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{pid}__chunk{idx}"
                    if chunk_id in done_full_chunk_pids:
                        continue
                    append_jsonl(
                        full_passages_chunks_path,
                        {
                            "passage_id": chunk_id,
                            "source_id": pid,
                            "chunk_index": idx,
                            "chunk_count": len(chunks),
                            "title": title,
                            "text": chunk_text,
                        },
                    )

            if full_passages_chunks_discourse_aware_path is None:
                continue
            da_chunks, da_extended_flags = split_long_passage_discourse_aware_with_flags(
                cleaned
            )
            if not da_chunks:
                continue
            if len(da_chunks) == 1:
                chunk_id = pid
                if chunk_id in done_full_da_chunk_pids:
                    continue
                append_jsonl(
                    full_passages_chunks_discourse_aware_path,
                    {
                        "passage_id": chunk_id,
                        "source_id": pid,
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "title": title,
                        "text": da_chunks[0],
                    },
                )
                done_full_da_chunk_pids.add(chunk_id)
                if (
                    full_passages_chunks_discourse_aware_debug_path is not None
                    and chunk_id not in done_full_da_chunk_debug_pids
                ):
                    extended = bool(da_extended_flags[0]) if da_extended_flags else False
                    if not extended:
                        continue
                    append_jsonl(
                        full_passages_chunks_discourse_aware_debug_path,
                        {
                            "passage_id": chunk_id,
                            "source_id": pid,
                            "chunk_index": 0,
                            "chunk_count": 1,
                            "title": title,
                            "text": da_chunks[0],
                            "extended": True,
                        },
                    )
                    done_full_da_chunk_debug_pids.add(chunk_id)
                continue
            for idx, chunk_text in enumerate(da_chunks):
                chunk_id = f"{pid}__chunk{idx}"
                if chunk_id in done_full_da_chunk_pids:
                    continue
                append_jsonl(
                    full_passages_chunks_discourse_aware_path,
                    {
                        "passage_id": chunk_id,
                        "source_id": pid,
                        "chunk_index": idx,
                        "chunk_count": len(da_chunks),
                        "title": title,
                        "text": chunk_text,
                    },
                )
                done_full_da_chunk_pids.add(chunk_id)
                if (
                    full_passages_chunks_discourse_aware_debug_path is not None
                    and chunk_id not in done_full_da_chunk_debug_pids
                ):
                    extended = idx < len(da_extended_flags) and da_extended_flags[idx]
                    if not extended:
                        continue
                    append_jsonl(
                        full_passages_chunks_discourse_aware_debug_path,
                        {
                            "passage_id": chunk_id,
                            "source_id": pid,
                            "chunk_index": idx,
                            "chunk_count": len(da_chunks),
                            "title": title,
                            "text": chunk_text,
                            "extended": True,
                        },
                    )
                    done_full_da_chunk_debug_pids.add(chunk_id)


##### Ingestion pipeline helpers
"""Dataset-specific ingestion helpers for orchestrators."""


def _run_ingestion(
    dataset: str,
    split: str,
    *,
    max_examples: int | None = None,
    resume: bool = True,
) -> None:
    config = DATASET_CONFIGS.get(dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset for ingestion: {dataset}")

    file_path = config["file_path"]
    if callable(file_path):
        file_path = file_path(split)
    field_map = config["field_map"]
    if callable(field_map):
        field_map = field_map(split)

    process_dataset(
        dataset=dataset,
        split=split,
        file_path=file_path,
        field_map=field_map,
        max_examples=max_examples,
        resume=resume,
    )


