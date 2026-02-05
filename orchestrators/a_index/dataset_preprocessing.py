"""Orchestrator for preprocessing datasets into QA/passages JSONL files."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.a1_ingestion.chunking import (
    split_long_passage,
    split_long_passage_discourse_aware_with_flags,
)
from src.utils.algorithms.discourse_aware import (
    _expand_sentence_text,
    _has_anaphora_marker,
    _has_cataphora_marker,
    iter_sentence_spans,
)
from src.a1_ingestion.dataset_preprocessing_functions import (
    DEFAULT_OUTPUTS,
    process_dataset,
    process_acord_split,
    sentence_passages_from_full,
)
from src.a2_indexing.splits import carve_validation_split
from src.utils.__utils__ import (
    append_jsonl,
    clean_text,
    existing_ids,
    load_jsonl,
    pid_plus_title,
    pid_plus_title_full,
    processed_dataset_paths,
)

# Config
DATASETS_TO_PREPROCESS = [
    "musique",
    "hotpotqa",
    "2wikimultihopqa",
    "natural_questions",
    "acord",
    #"fever",
]



# change to train, hotpotqa, 25000

RESUME = True
SPLITS = ["train", "dev"]
MAX_ROWS = 5000  # set to an int to cap rows per split


TO_PREPROCESS = ['passages', 'questions', 'full_passages', 'full_passages_chunks']

ACORD_POS_THRESHOLD = 3
ACORD_NEG_THRESHOLD = 1
ACORD_HARD_THRESHOLD = 2
ACORD_BUILD_TRAINING_PACKAGES = True
ACORD_INCLUDE_TRAINING_TEXT = False
ACORD_MAX_POS = None
ACORD_MAX_NEG = None
ACORD_MAX_HARD_NEG = None
ACORD_DROP_NO_POSITIVES = True

# Validation split carving (from processed train split).
CARVE_VAL_SPLIT = True
VAL_SOURCE_SPLIT = "train"
VAL_TRAIN_SPLIT = "train_sub"
VAL_SPLIT_NAME = "val"
VAL_RATIO = 0.1
VAL_SIZE = None
VAL_SEED = 1
VAL_SHUFFLE = True
VAL_MAX_QUESTIONS = None
VAL_OVERWRITE = False


def _process_natural_questions_csv(
    *,
    split: str,
    file_path: str,
    max_rows: int | None,
    resume: bool,
    include_outputs: set[str] | None = None,
) -> None:
    paths = processed_dataset_paths("natural_questions", split)
    qa_path = str(paths["questions"])
    passages_path = str(paths["passages"])
    full_passages_path = str(paths["full_passages"])
    full_passages_chunks_path = str(paths["full_passages_chunks"])
    full_passages_chunks_discourse_aware_path = str(
        paths["full_passages_chunks_discourse_aware"]
    )
    passages_discourse_aware_path = str(paths["passages_discourse_aware"])
    passages_discourse_aware_debug_path = str(paths["passages_discourse_aware_debug"])
    full_passages_chunks_discourse_aware_debug_path = str(
        paths["full_passages_chunks_discourse_aware_debug"]
    )

    if include_outputs is None:
        include_outputs = set(DEFAULT_OUTPUTS)
    include_outputs = set(include_outputs)
    include_questions = "questions" in include_outputs
    include_passages = "passages" in include_outputs
    include_full_passages = "full_passages" in include_outputs
    include_full_passages_chunks = "full_passages_chunks" in include_outputs
    include_passages_discourse_aware = "passages_discourse_aware" in include_outputs
    include_passages_discourse_aware_debug = (
        "passages_discourse_aware_debug" in include_outputs
    )
    include_full_passages_chunks_discourse_aware = (
        "full_passages_chunks_discourse_aware" in include_outputs
    )
    include_full_passages_chunks_discourse_aware_debug = (
        "full_passages_chunks_discourse_aware_debug" in include_outputs
    )
    include_discourse_outputs = (
        include_passages_discourse_aware or include_passages_discourse_aware_debug
    )
    include_da_chunk_outputs = (
        include_full_passages_chunks_discourse_aware
        or include_full_passages_chunks_discourse_aware_debug
    )

    done_qids = (
        existing_ids(qa_path, id_field="question_id")
        if resume and include_questions
        else set()
    )
    done_pids = (
        existing_ids(passages_path, id_field="passage_id")
        if resume and include_passages
        else set()
    )
    done_full_pids = (
        existing_ids(full_passages_path, id_field="passage_id")
        if resume and include_full_passages
        else set()
    )
    done_full_chunk_pids = (
        existing_ids(full_passages_chunks_path, id_field="passage_id")
        if resume and include_full_passages_chunks
        else set()
    )
    done_full_da_chunk_pids = (
        existing_ids(full_passages_chunks_discourse_aware_path, id_field="passage_id")
        if resume and include_full_passages_chunks_discourse_aware
        else set()
    )
    done_discourse_pids = (
        existing_ids(passages_discourse_aware_path, id_field="passage_id")
        if resume and include_passages_discourse_aware
        else set()
    )
    done_discourse_debug_pids = (
        existing_ids(passages_discourse_aware_debug_path, id_field="passage_id")
        if resume and include_passages_discourse_aware_debug
        else set()
    )
    done_full_da_chunk_debug_pids = (
        existing_ids(
            full_passages_chunks_discourse_aware_debug_path, id_field="passage_id"
        )
        if resume and include_full_passages_chunks_discourse_aware_debug
        else set()
    )

    with open(file_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        total = max_rows if isinstance(max_rows, int) else None
        for idx, row in enumerate(
            tqdm(reader, desc=f"natural_questions/{split} rows", total=total)
        ):
            if isinstance(max_rows, int) and idx >= max_rows:
                break

            qid = f"nq_{split}_{idx}"
            question = clean_text(row.get("question", ""))
            long_answer = clean_text(row.get("long_answers", ""))
            short_answer = clean_text(row.get("short_answers", ""))
            passage_id = f"{qid}__sent0"
            sentence_passages = (
                sentence_passages_from_full(passage_id, "", long_answer)
                if long_answer
                else []
            )
            sentence_ids = [pid for pid, _title, _text in sentence_passages]

            if include_questions and qid not in done_qids:
                append_jsonl(
                    qa_path,
                    {
                        "question_id": qid,
                        "dataset": "natural_questions",
                        "split": split,
                        "question": question,
                        "gold_answer": short_answer,
                        "gold_passages": sentence_ids,
                        "gold_passages_full": [passage_id] if long_answer else [],
                    },
                )

            if include_passages and long_answer:
                for sent_id, _title, sent_text in sentence_passages:
                    if sent_id in done_pids:
                        continue
                    append_jsonl(
                        passages_path,
                        {
                            "passage_id": sent_id,
                            "title": "",
                            "text": sent_text,
                        },
                    )
            if include_full_passages and long_answer and passage_id not in done_full_pids:
                append_jsonl(
                    full_passages_path,
                    {
                        "passage_id": passage_id,
                        "title": "",
                        "text": long_answer,
                    },
                )
            if include_discourse_outputs and long_answer:
                sentences = [s.strip() for _, _, s in iter_sentence_spans(long_answer)]
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
                    expanded_entries.append(
                        {
                            "idx": idx,
                            "expanded": expanded,
                            "span_start": span_start,
                            "span_end": span_end,
                            "text": expanded_text.strip(),
                        }
                    )

                for entry in expanded_entries:
                    idx = int(entry["idx"])
                    expanded = bool(entry["expanded"])
                    if not expanded and idx in covered_by_expansion:
                        continue
                    discourse_id = f"{passage_id}__sent{idx}"
                    if include_passages_discourse_aware:
                        if discourse_id in done_discourse_pids:
                            continue
                        append_jsonl(
                            passages_discourse_aware_path,
                            {
                                "passage_id": discourse_id,
                                "source_id": passage_id,
                                "expanded": expanded,
                                "title": "",
                                "text": str(entry["text"]),
                            },
                        )
                        done_discourse_pids.add(discourse_id)
                    if (
                        include_passages_discourse_aware_debug
                        and expanded
                        and discourse_id not in done_discourse_debug_pids
                    ):
                        append_jsonl(
                            passages_discourse_aware_debug_path,
                            {
                                "passage_id": discourse_id,
                                "source_id": passage_id,
                                "sent_idx": idx,
                                "span_start": int(entry["span_start"]),
                                "span_end": int(entry["span_end"]),
                                "expanded": True,
                                "title": "",
                                "text": str(entry["text"]),
                            },
                        )
                        done_discourse_debug_pids.add(discourse_id)
            if (include_full_passages_chunks or include_da_chunk_outputs) and long_answer:
                if include_full_passages_chunks:
                    chunks = split_long_passage(long_answer)
                    if len(chunks) == 1:
                        chunk_id = passage_id
                        if chunk_id not in done_full_chunk_pids:
                            append_jsonl(
                                full_passages_chunks_path,
                                {
                                    "passage_id": chunk_id,
                                    "source_id": passage_id,
                                    "chunk_index": 0,
                                    "chunk_count": 1,
                                    "title": "",
                                    "text": chunks[0],
                                },
                            )
                    else:
                        for idx, chunk_text in enumerate(chunks):
                            chunk_id = f"{passage_id}__chunk{idx}"
                            if chunk_id in done_full_chunk_pids:
                                continue
                            append_jsonl(
                                full_passages_chunks_path,
                                {
                                    "passage_id": chunk_id,
                                    "source_id": passage_id,
                                    "chunk_index": idx,
                                    "chunk_count": len(chunks),
                                    "title": "",
                                    "text": chunk_text,
                                },
                            )
                if include_da_chunk_outputs:
                    da_chunks, da_extended_flags = (
                        split_long_passage_discourse_aware_with_flags(long_answer)
                    )
                    if len(da_chunks) == 1:
                        chunk_id = passage_id
                        if (
                            include_full_passages_chunks_discourse_aware
                            and chunk_id not in done_full_da_chunk_pids
                        ):
                            append_jsonl(
                                full_passages_chunks_discourse_aware_path,
                                {
                                    "passage_id": chunk_id,
                                    "source_id": passage_id,
                                    "chunk_index": 0,
                                    "chunk_count": 1,
                                    "title": "",
                                    "text": da_chunks[0],
                                },
                            )
                            done_full_da_chunk_pids.add(chunk_id)
                        if (
                            include_full_passages_chunks_discourse_aware_debug
                            and chunk_id not in done_full_da_chunk_debug_pids
                        ):
                            extended = (
                                bool(da_extended_flags[0])
                                if da_extended_flags
                                else False
                            )
                            if not extended:
                                continue
                            append_jsonl(
                                full_passages_chunks_discourse_aware_debug_path,
                                {
                                    "passage_id": chunk_id,
                                    "source_id": passage_id,
                                    "chunk_index": 0,
                                    "chunk_count": 1,
                                    "title": "",
                                    "text": da_chunks[0],
                                    "extended": True,
                                },
                            )
                            done_full_da_chunk_debug_pids.add(chunk_id)
                    else:
                        for idx, chunk_text in enumerate(da_chunks):
                            chunk_id = f"{passage_id}__chunk{idx}"
                            if (
                                include_full_passages_chunks_discourse_aware
                                and chunk_id not in done_full_da_chunk_pids
                            ):
                                append_jsonl(
                                    full_passages_chunks_discourse_aware_path,
                                    {
                                        "passage_id": chunk_id,
                                        "source_id": passage_id,
                                        "chunk_index": idx,
                                        "chunk_count": len(da_chunks),
                                        "title": "",
                                        "text": chunk_text,
                                    },
                                )
                                done_full_da_chunk_pids.add(chunk_id)
                            if (
                                include_full_passages_chunks_discourse_aware_debug
                                and chunk_id not in done_full_da_chunk_debug_pids
                            ):
                                if (
                                    idx >= len(da_extended_flags)
                                    or not da_extended_flags[idx]
                                ):
                                    continue
                                append_jsonl(
                                    full_passages_chunks_discourse_aware_debug_path,
                                    {
                                        "passage_id": chunk_id,
                                        "source_id": passage_id,
                                        "chunk_index": idx,
                                        "chunk_count": len(da_chunks),
                                        "title": "",
                                        "text": chunk_text,
                                        "extended": True,
                                    },
                                )
                                done_full_da_chunk_debug_pids.add(chunk_id)


def _build_fever_passages(*, shared_path: Path, resume: bool) -> None:
    if resume and shared_path.exists():
        return
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    if shared_path.exists() and not resume:
        shared_path.unlink()

    wiki_dir = Path("data/raw_datasets/FEVER/wiki-pages/wiki-pages")
    wiki_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not wiki_files:
        raise FileNotFoundError(f"No FEVER wiki files found in {wiki_dir}")

    for wiki_path in tqdm(wiki_files, desc="fever/wiki files"):
        for page in load_jsonl(str(wiki_path)):
            page_id = str(page.get("id", "")).strip()
            if not page_id:
                continue
            lines_blob = page.get("lines", "")
            if not lines_blob:
                continue
            for raw_line in str(lines_blob).splitlines():
                if not raw_line:
                    continue
                parts = raw_line.split("\t")
                if len(parts) < 2:
                    continue
                sent_idx = parts[0].strip()
                sent = clean_text(parts[1])
                if not sent:
                    continue
                passage_id = f"{page_id}__sent{sent_idx}"
                append_jsonl(
                    str(shared_path),
                    {
                        "passage_id": passage_id,
                        "title": page_id,
                        "text": sent,
                    },
                )


def _ensure_fever_passages(
    *,
    split: str,
    resume: bool,
) -> None:
    shared_path = Path("data/processed_datasets/fever/wiki_passages.jsonl")
    _build_fever_passages(shared_path=shared_path, resume=resume)

    split_passages = processed_dataset_paths("fever", split)["passages"]
    if Path(split_passages).exists():
        return
    try:
        os.link(shared_path, split_passages)
    except OSError:
        shutil.copyfile(shared_path, split_passages)


def _process_fever_split(
    *,
    split: str,
    file_path: str,
    max_rows: int | None,
    resume: bool,
    include_outputs: set[str] | None = None,
) -> None:
    if include_outputs is None:
        include_outputs = set(DEFAULT_OUTPUTS)
    include_outputs = set(include_outputs)
    include_questions = "questions" in include_outputs
    include_passages = "passages" in include_outputs

    if include_passages:
        _ensure_fever_passages(split=split, resume=resume)
    qa_path = processed_dataset_paths("fever", split)["questions"]
    done_qids = (
        existing_ids(qa_path, id_field="question_id")
        if resume and include_questions
        else set()
    )

    total = max_rows if isinstance(max_rows, int) else None
    for idx, ex in enumerate(
        tqdm(load_jsonl(file_path), desc=f"fever/{split} rows", total=total)
    ):
        if isinstance(max_rows, int) and idx >= max_rows:
            break
        qid = str(ex.get("id", ""))
        if not qid or qid in done_qids:
            continue
        question = clean_text(ex.get("claim", ""))
        label = ex.get("label", "")
        verifiable = ex.get("verifiable", "")

        gold_passages: List[str] = []
        seen: set[str] = set()
        for group in ex.get("evidence", []) or []:
            for ev in group or []:
                if len(ev) < 4:
                    continue
                page = ev[2]
                line_idx = ev[3]
                if page is None or line_idx is None:
                    continue
                passage_id = f"{page}__sent{line_idx}"
                if passage_id in seen:
                    continue
                seen.add(passage_id)
                gold_passages.append(passage_id)

        if include_questions:
            append_jsonl(
                str(qa_path),
                {
                    "question_id": qid,
                    "dataset": "fever",
                    "split": split,
                    "question": question,
                    "gold_answer": label,
                    "gold_passages": gold_passages,
                    "verifiable": verifiable,
                },
            )


def main() -> None:
    include_outputs = {item.strip() for item in TO_PREPROCESS}

    for dataset in tqdm(DATASETS_TO_PREPROCESS, desc="datasets"):
        for split in tqdm(SPLITS, desc=f"{dataset}/splits", leave=False):
            if dataset == "natural_questions":
                split_files = {
                    "train": "data/raw_datasets/natural_questions/Natural-Questions-Base.csv",
                    "dev": "data/raw_datasets/natural_questions/Natural-Questions-Filtered.csv",
                }
                file_path = split_files.get(split)
                if file_path is None:
                    print(f"[skip] natural_questions has no {split} split")
                    continue
                if not Path(file_path).exists():
                    print(f"[skip] missing natural_questions {split}: {file_path}")
                    continue
                _process_natural_questions_csv(
                    split=split,
                    file_path=file_path,
                    max_rows=MAX_ROWS,
                    resume=RESUME,
                    include_outputs=include_outputs,
                )
                continue

            if dataset == "acord":
                raw_dir = Path("data/raw_datasets/ACORD")
                split_alias = "valid" if split == "dev" else split
                queries_path = raw_dir / "queries.jsonl"
                corpus_path = raw_dir / "corpus.jsonl"
                qrels_path = raw_dir / f"{split_alias}.tsv"
                missing = [
                    str(p)
                    for p in [queries_path, corpus_path, qrels_path]
                    if not p.exists()
                ]
                if missing:
                    print(f"[skip] missing acord {split}: {', '.join(missing)}")
                    continue
                process_acord_split(
                    split=split,
                    raw_dir=raw_dir,
                    include_outputs=include_outputs,
                    pos_threshold=ACORD_POS_THRESHOLD,
                    hard_threshold=ACORD_HARD_THRESHOLD,
                    neg_threshold=ACORD_NEG_THRESHOLD,
                    resume=RESUME,
                    include_training_packages=ACORD_BUILD_TRAINING_PACKAGES,
                    max_pos=ACORD_MAX_POS,
                    max_neg=ACORD_MAX_NEG,
                    max_hard_neg=ACORD_MAX_HARD_NEG,
                    drop_no_positives=ACORD_DROP_NO_POSITIVES,
                    include_training_text=ACORD_INCLUDE_TRAINING_TEXT,
                )
                continue

            if dataset == "fever":
                split_files = {
                    "train": "data/raw_datasets/FEVER/train.jsonl",
                    "dev": "data/raw_datasets/FEVER/shared_task_dev.jsonl",
                    "test": "data/raw_datasets/FEVER/shared_task_test.jsonl",
                }
                file_path = split_files.get(split)
                if file_path is None:
                    print(f"[skip] fever has no {split} split")
                    continue
                if not Path(file_path).exists():
                    print(f"[skip] missing fever {split}: {file_path}")
                    continue
                _process_fever_split(
                    split=split,
                    file_path=file_path,
                    max_rows=MAX_ROWS,
                    resume=RESUME,
                    include_outputs=include_outputs,
                )
                continue

            if dataset == "hotpotqa":
                if split == "train":
                    file_path = "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
                elif split == "dev":
                    file_path = "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
                elif split == "test":
                    file_path = "data/raw_datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
                else:
                    raise ValueError(f"Unsupported HotpotQA split: {split}")
                field_map = {
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
                }
            elif dataset == "2wikimultihopqa":
                file_path = f"data/raw_datasets/2wikimultihopqa/{split}.json"
                field_map = {
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
                }
            elif dataset == "musique":
                file_path = f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl"
                field_map = {
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
                        for pid, _title, _text in sentence_passages_from_full(
                            f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                            p.get("title", ""),
                            p.get("paragraph_text", ""),
                        )
                    ],
                    "gold_full_passage_ids": lambda ex: [
                        f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}"
                        for i, p in enumerate(ex.get("paragraphs", []))
                        if p.get("is_supporting")
                    ],
                }

            if not Path(file_path).exists():
                print(f"[skip] missing {dataset} {split}: {file_path}")
                continue

            process_dataset(
                dataset=dataset,
                split=split,
                file_path=file_path,
                field_map=field_map,
                max_examples=MAX_ROWS,
                resume=RESUME,
                include_outputs=include_outputs,
            )

        if CARVE_VAL_SPLIT and VAL_SOURCE_SPLIT in SPLITS:
            try:
                results = carve_validation_split(
                    dataset=dataset,
                    source_split=VAL_SOURCE_SPLIT,
                    train_split=VAL_TRAIN_SPLIT,
                    val_split=VAL_SPLIT_NAME,
                    val_ratio=VAL_RATIO,
                    val_size=VAL_SIZE,
                    seed=VAL_SEED,
                    shuffle=VAL_SHUFFLE,
                    max_questions=VAL_MAX_QUESTIONS,
                    include_outputs=list(include_outputs),
                    overwrite=VAL_OVERWRITE,
                )
                summary = results.get("summary", {})
                print(
                    f"[val_split] {dataset} "
                    f"{summary.get('source_split')} -> "
                    f"{summary.get('train_split')}/{summary.get('val_split')} "
                    f"(train_q={summary.get('train_questions')}, "
                    f"val_q={summary.get('val_questions')})"
                )
            except FileNotFoundError as exc:
                print(f"[skip] val split for {dataset}: {exc}")


if __name__ == "__main__":
    main()
