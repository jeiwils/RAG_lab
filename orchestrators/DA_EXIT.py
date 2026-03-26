from __future__ import annotations


import numpy as np
import json
import logging
import os
import random

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from tqdm import tqdm

from src.a3_representations import (
    dataset_rep_paths,
    get_embedding_model,
)
from src.a3_representations.representation_loading import load_chunk_representations
from src.a3_representations.sparse_representations import SPACY_MODEL
from src.a2_indexing.retrieval_indexing import _retrieve_chunk_indices
from src.b2_reranking.useful_sentence_extractor_infer import (
    DEFAULT_MAX_LEN,
    is_lora_checkpoint,
    load_sentence_lora,
    score_sentences_lora,
    select_ranked_sentences,
)
from src.b1_retrieval import DEFAULT_HYBRID_ALPHA
from src.c1_reasoning.sentence_reasoning import (
    _build_sentence_candidates,
    _select_top_sentences,
    compute_passage_hits_recall_at_k,
    compute_passage_precision_at_k,
)
from src.c2_generation.answer_gen import ask_llm_with_passages
from src.d1_evaluation.answer_metrics import aggregate_answer_scores, evaluate_answers
from src.d1_evaluation.timing_metrics import compute_throughput_stats, wall_time
from src.utils.__utils__ import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    limit_questions,
    load_jsonl,
    processed_dataset_paths,
    save_jsonl,
)
from src.utils.token_budgeting import estimate_tokens

__all__ = ["run_da_exit", "main"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

### DATA & SPLITS
DATASETS = ["musique"]
SPLITS = ["dev"]

### RETRIEVAL & PASSAGE
RETRIEVER_CONFIG = {
    "hybrid": True,
    "dense": False,
    "sparse": False,
}
TOP_K_CHUNK_SWEEP = [5, 10]
HYBRID_ALPHA = DEFAULT_HYBRID_ALPHA 

PASSAGE_SOURCE = "full_passages_chunks"

# Sentence mode controls post-ranking expansion for the reader context.
# "standard_sentences" = no expansion; "discourse_aware_sentences" = expand after ranking.
SENTENCE_MODES = ["standard_sentences", "discourse_aware_sentences"]

### SELECTION & BUDGETS
# Sentence usefulness cutoff sweep (set from best_metrics.json of chosen LoRA model)
TAU_LOW_SWEEP = [0.35, 0.465, 0.58] # higher recall;  teacher chosen; higher precision
# Total token budget for reader context (selected sentences).

# Sweep different reader context budgets to evaluate discourse-aware efficiency
READER_CONTEXT_BUDGET_TOKENS_SWEEP = [256, 384, 512]
### READER
READER_MODEL = "Mistral/Mistral-7B-Instruct-v0.1.Q4_0.gguf" 
SERVER_URL = "http://localhost:8005"
USE_IN_PROCESS_READER = False
IN_PROCESS_READER_LOCAL_FILES_ONLY = True

### LORA / RERANKER
SENTENCE_GRADER_MODEL = (
    "data/models/useful_sentence_lora/"
    "checkpoint-epoch2"
)
SENTENCE_LORA_BATCH_SIZE = 32
SENTENCE_LORA_MAX_LEN = DEFAULT_MAX_LEN
SENTENCE_LORA_LOCAL_FILES_ONLY = True

### RUN CONTROL
SEEDS = [1, 2, 3]
MAX_QUESTIONS: int | None = 100  # set to None to use the full split
SHUFFLE_QUESTIONS = False
RESUME = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_da_exit(
    dataset: str,
    split: str,
    retriever: str,
    *,
    reader_model: str,
    sentence_model: str,
    server_url: str | None,
    top_k_chunks: int,
    sentence_mode: str,
    seed: int | None,
    resume: bool,
    tau_low: float,
    reader_context_budget_tokens: int,
    show_query_progress: bool = True,
    ) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass

    if not is_lora_checkpoint(sentence_model):
        raise ValueError(
            "SENTENCE_GRADER_MODEL must be a LoRA checkpoint path with adapter_config.json. "
            f"Got: {sentence_model}"
        )
    if not USE_IN_PROCESS_READER and server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    processed_paths = processed_dataset_paths(dataset, split)
    passage_source = PASSAGE_SOURCE

    rep_paths = dataset_rep_paths(dataset, split, passage_source=passage_source)
    try:
        metadata, index = load_chunk_representations(rep_paths)
    except FileNotFoundError as e:
        logger.error(f"Missing representations for {dataset}/{split}: {e}")
        return {}
    except ValueError as e:
        logger.error(f"Corrupted representations: {e}")
        return {}

    encoder = None
    if retriever in {"dense", "hybrid"}:
        try:
            encoder = get_embedding_model()
        except Exception as e:
            logger.error(f"Failed to load embedding model for {retriever} retrieval: {e}")
            return {}

    questions_path = processed_paths["questions"]
    try:
        questions = limit_questions(
            list(load_jsonl(str(questions_path))),
            max_questions=MAX_QUESTIONS,
            seed=seed,
            shuffle=SHUFFLE_QUESTIONS,
        )
    except FileNotFoundError:
        logger.error(f"Questions file not found: {questions_path}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return {}

    if not questions:
        logger.warning(f"No questions found in {questions_path}")
        return {}

    logger.info(f"Loading sentence grader model from {sentence_model}...")
    grader_model, grader_tokenizer, grader_device = load_sentence_lora(
        checkpoint_dir=Path(sentence_model),
        local_files_only=SENTENCE_LORA_LOCAL_FILES_ONLY,
    )
    method_prefix = "exit" if sentence_mode == "standard_sentences" else "da_exit"
    tau_tag = f"tau{tau_low}".replace(".", "p")
    variant = f"{method_prefix}_{retriever}_k{top_k_chunks}_{tau_tag}_b{reader_context_budget_tokens}"
    if seed is not None:
        variant = f"{variant}_seed{seed}"
    paths = get_result_paths(reader_model, dataset, split, variant)
    paths["base"].mkdir(parents=True, exist_ok=True)
    debug_path = paths["base"] / f"reader_debug_{variant}_{split}.jsonl"
    if not resume and paths["answers"].exists():
        paths["answers"].unlink()
    if not resume and paths["answer_metrics"].exists():
        paths["answer_metrics"].unlink()
    if not resume and debug_path.exists():
        debug_path.unlink()

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(paths["answers"]),
        items=questions,
        get_id=lambda q, i: q["question_id"],
        phase_label=f"DA_EXIT {retriever}",
        id_field="question_id",
    )

    predictions: Dict[str, str] = {}
    gold_all: Dict[str, List[str]] = {}
    wall_times: List[float] = []
    reader_wall_times: List[float] = []

    token_totals = {
        "sentence_prompt_tokens": 0,
        "sentence_output_tokens": 0,
        "sentence_total_tokens": 0,
        "reader_prompt_tokens": 0,
        "reader_output_tokens": 0,
        "reader_total_tokens": 0,
        "n_sentence_calls": 0,
        "n_reader_calls": 0,
    }
    hits_at_k_scores: List[float] = []
    recall_at_k_scores: List[float] = []
    precision_at_k_scores: List[float] = []

    question_iter = (
        tqdm(questions, desc=f"{dataset}/{split}/{retriever}")
        if show_query_progress
        else questions
    )
    for q in question_iter:
        q_id = q.get("question_id")
        if not q_id:
            logger.warning("Question missing question_id, skipping")
            continue
        gold_all[q_id] = [q.get("gold_answer", "")]
        if resume and q_id in done_ids:
            continue
        q_text = q.get("question", "").strip()
        if not q_text:
            logger.warning(f"Question {q_id} has no text, skipping")
            continue

        def _run_query():
            """Execute retrieval, sentence scoring, and reader inference for one query."""
            query_vec = None
            if retriever in {"dense", "hybrid"} and encoder is not None:
                query_vec = encoder.encode(
                    [q_text],
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            chunk_idxs = _retrieve_chunk_indices(
                retriever,
                q_text,
                query_vec,
                metadata,
                index,
                top_k=top_k_chunks,
                alpha=HYBRID_ALPHA,
            )

            chunks = [metadata[i] for i in chunk_idxs]
            sentences: List[Dict[str, Any]] = []
            expanded_by_id: Dict[str, Dict[str, Any]] = {}
            chunk_ids: List[str] = []
            expand_after = sentence_mode == "discourse_aware_sentences"
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", chunk.get("passage_id", "chunk"))
                chunk_ids.append(chunk_id)
                sentences.extend(
                    _build_sentence_candidates(
                        chunk.get("text", ""),
                        chunk_id,
                        mode="standard",
                        extension=1,
                        include_reformulation=True,
                        include_parenthetical=True,
                        require_forward_punct=False,
                    )
                )
                if expand_after:
                    expanded = _build_sentence_candidates(
                        chunk.get("text", ""),
                        chunk_id,
                        mode="discourse_aware",
                        extension=1,
                        include_reformulation=True,
                        include_parenthetical=True,
                        require_forward_punct=False,
                    )
                    for item in expanded:
                        expanded_by_id[item["sentence_id"]] = item

            def _write_reader_debug(
                selected_sentences: List[Dict[str, Any]],
                reason: Optional[str] = None,
                n_candidates: Optional[int] = None,
                n_ranked: Optional[int] = None,
                n_filtered: Optional[int] = None,
                reader_budget: Optional[Dict[str, Any]] = None,
            ) -> None:
                def _pack_sentence(s: Dict[str, Any]) -> Dict[str, Any]:
                    return {
                        "sentence_id": s.get("sentence_id"),
                        "chunk_id": s.get("chunk_id"),
                        "sent_idx": s.get("sent_idx"),
                        "span_start": s.get("span_start", s.get("sent_idx")),
                        "span_end": s.get("span_end", s.get("sent_idx")),
                        "expanded": s.get("expanded", False),
                        "score": s.get("score", 0.0),
                        "score_normalized": s.get("score_normalized", 0.0),
                        "text": s.get("text", ""),
                    }

                payload = {
                    "dataset": dataset,
                    "split": split,
                    "variant": variant,
                    "retriever": retriever,
                    "passage_source": passage_source,
                    "sentence_mode": sentence_mode,
                    "expand_after_rerank": expand_after,
                    "reader_model": reader_model,
                    "question_id": q_id,
                    "question": q_text,
                    "tau_low": tau_low,
                    "token_budget": reader_context_budget_tokens,
                    "top_k_chunks": top_k_chunks,
                    "retrieved_chunk_count": len(chunk_ids),
                    "retrieved_chunk_ids": chunk_ids,
                    "n_selected": len(selected_sentences),
                    "selected_sentences": [_pack_sentence(s) for s in selected_sentences],
                }
                if reason is not None:
                    payload["reason"] = reason
                if n_candidates is not None:
                    payload["n_candidates"] = n_candidates
                if n_ranked is not None:
                    payload["n_ranked"] = n_ranked
                if n_filtered is not None:
                    payload["n_filtered"] = n_filtered
                if reader_budget is not None:
                    payload["reader_budget"] = reader_budget
                append_jsonl(str(debug_path), payload)

            if not sentences:
                _write_reader_debug(
                    [],
                    reason="no_candidates",
                    n_candidates=0,
                )
                return {
                    "selected_sentences": [],
                    "answer": {
                        "raw_answer": "unknown",
                        "raw_clean": "unknown",
                        "normalised_answer": "unknown",
                        "prompt_len": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "sentence_tokens": {
                        "sentence_prompt_tokens": 0,
                        "sentence_output_tokens": 0,
                        "sentence_total_tokens": 0,
                        "n_sentence_calls": 0,
                    },
                    "reader_wall_time_sec": 0.0,
                }

            """Scoring: LoRA checkpoint sentence grader."""
            scored, sentence_tokens = score_sentences_lora(
                q_text,
                sentences,
                model=grader_model,         
                tokenizer=grader_tokenizer, 
                device=grader_device,       
                batch_size=SENTENCE_LORA_BATCH_SIZE,
                max_length=SENTENCE_LORA_MAX_LEN,
            )

            """Ranking: _select_top_sentences(scored, top_k=len(scored)) sorts by score (highest first)."""
            ranked = _select_top_sentences(scored, top_k=len(scored))
            reader_count_tokens = lambda text: estimate_tokens(text, reader_model)

            """Repackaging with tau + budget: select_ranked_sentences(..., tau_low=TAU_LOW, token_budget=READER_CONTEXT_BUDGET_TOKENS) keeps only above tau and within the budget, preserving rank order."""
            filtered = select_ranked_sentences(
                ranked,
                token_budget=reader_context_budget_tokens,
                tau_low=tau_low,
                count_tokens_fn=reader_count_tokens,
            )
            selected = filtered
            if expand_after:
                expanded_selected: List[Dict[str, Any]] = []
                for s in selected:
                    expanded = expanded_by_id.get(s["sentence_id"])
                    if expanded is None:
                        expanded_selected.append(s)
                        continue
                    merged = dict(s)
                    merged["text"] = expanded.get("text", s.get("text", ""))
                    merged["span_start"] = expanded.get(
                        "span_start", s.get("span_start", s["sent_idx"])
                    )
                    merged["span_end"] = expanded.get(
                        "span_end", s.get("span_end", s["sent_idx"])
                    )
                    merged["expanded"] = expanded.get("expanded", False)
                    expanded_selected.append(merged)
                # Re-apply token budget after expansion, preserving rank order.
                expanded_selected = select_ranked_sentences(
                    expanded_selected,
                    token_budget=reader_context_budget_tokens,
                    tau_low=None,
                    count_tokens_fn=reader_count_tokens,
                )
                selected = expanded_selected
            if not selected:
                _write_reader_debug(
                    [],
                    reason="no_selected",
                    n_candidates=len(sentences),
                    n_ranked=len(ranked),
                    n_filtered=len(filtered),
                )
                return {
                    "selected_sentences": [],
                    "answer": {
                        "raw_answer": "unknown",
                        "raw_clean": "unknown",
                        "normalised_answer": "unknown",
                        "prompt_len": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "sentence_tokens": sentence_tokens,
                    "reader_wall_time_sec": 0.0,
                }

            sentence_lookup = {s["sentence_id"]: s["text"] for s in selected}
            sentence_ids = list(sentence_lookup.keys())

            """Sent to reader: the ordered selected list is passed through ask_llm_with_passages in that same order."""
            answer, reader_wall_time_sec = wall_time(
                ask_llm_with_passages,
                query_text=q_text,
                passage_ids=sentence_ids,
                graph=None,
                server_url=server_url,
                passage_lookup=sentence_lookup,
                model_name=reader_model,
                top_k_answer_passages=len(sentence_ids),
                seed=seed,
                in_process=USE_IN_PROCESS_READER,
                local_files_only=IN_PROCESS_READER_LOCAL_FILES_ONLY,
            )

            reader_budget = {
                "context_window_tokens": answer.get("budget_context_window_tokens"),
                "reserved_output_tokens": answer.get("budget_reserved_output_tokens"),
                "safety_margin_tokens": answer.get("budget_safety_margin_tokens"),
                "prompt_budget_tokens": answer.get("budget_prompt_tokens"),
                "base_prompt_tokens": answer.get("budget_base_prompt_tokens"),
                "context_budget_tokens": answer.get("budget_context_tokens"),
                "max_passage_tokens": answer.get("budget_max_passage_tokens"),
                "n_candidate_passages": answer.get("budget_n_candidate_passages"),
                "n_selected_passages": answer.get("budget_n_selected_passages"),
                "n_dropped_passages": answer.get("budget_n_dropped_passages"),
                "n_truncated_passages": answer.get("budget_n_truncated_passages"),
                "used_context_tokens": answer.get("budget_used_context_tokens"),
            }
            _write_reader_debug(
                selected,
                n_candidates=len(sentences),
                n_ranked=len(ranked),
                n_filtered=len(filtered),
                reader_budget=reader_budget,
            )

            return {
                "selected_sentences": selected,
                "answer": answer,
                "sentence_tokens": sentence_tokens,
                "reader_wall_time_sec": reader_wall_time_sec,
            }

        try:
            result, query_wall_sec = wall_time(_run_query)
        except Exception as e:
            logger.error(f"Query {q_id} failed: {e}")
            # Use default values for this failed query
            result = {
                "selected_sentences": [],
                "answer": {
                    "raw_answer": "unknown",
                    "raw_clean": "unknown",
                    "normalised_answer": "unknown",
                    "prompt_len": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "sentence_tokens": {
                    "sentence_prompt_tokens": 0,
                    "sentence_output_tokens": 0,
                    "sentence_total_tokens": 0,
                    "n_sentence_calls": 0,
                },
                "reader_wall_time_sec": 0.0,
            }
            query_wall_sec = 0.0

        wall_times.append(query_wall_sec)
        reader_wall_times.append(result.get("reader_wall_time_sec", 0.0))

        selected_sentences = result.get("selected_sentences", [])
        answer = result.get("answer", {
            "raw_answer": "unknown",
            "raw_clean": "unknown",
            "normalised_answer": "unknown",
            "prompt_len": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        })
        sentence_tokens = result.get("sentence_tokens", {
            "sentence_prompt_tokens": 0,
            "sentence_output_tokens": 0,
            "sentence_total_tokens": 0,
            "n_sentence_calls": 0,
        })

        token_totals["sentence_prompt_tokens"] += sentence_tokens.get(
            "sentence_prompt_tokens", 0
        )
        token_totals["sentence_output_tokens"] += sentence_tokens.get(
            "sentence_output_tokens", 0
        )
        token_totals["sentence_total_tokens"] += sentence_tokens.get(
            "sentence_total_tokens", 0
        )
        token_totals["n_sentence_calls"] += sentence_tokens.get("n_sentence_calls", 0)

        token_totals["reader_prompt_tokens"] += answer.get("prompt_len", 0)
        token_totals["reader_output_tokens"] += answer.get("output_tokens", 0)
        token_totals["reader_total_tokens"] += answer.get("total_tokens", 0)
        token_totals["n_reader_calls"] += 1

        gold_passages = q.get("gold_passages_full") or q.get("gold_passages") or []
        if gold_passages:
            passage_metrics = compute_passage_hits_recall_at_k(
                selected_sentences,
                gold_passages,
                k=top_k_chunks,
            )
            hits_at_k_scores.append(passage_metrics["hits_at_k_ratio"])
            recall_at_k_scores.append(passage_metrics["recall_at_k_ratio"])
            precision_metrics = compute_passage_precision_at_k(
                selected_sentences,
                gold_passages,
                k=None,
            )
            precision_at_k_scores.append(precision_metrics["precision_at_k_ratio"])

        append_jsonl(
            str(paths["answers"]),
            {
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "retriever": retriever,
                "passage_source": passage_source,
                "reader_model": reader_model,
                "sentence_model": sentence_model,
                "question_id": q_id,
                "question": q_text,
                "raw_answer": answer.get("raw_answer", ""),
                "normalised_answer": answer.get("normalised_answer", ""),
                "used_sentence_ids": [s.get("sentence_id", "unknown") for s in selected_sentences],
                "selected_sentences": [
                    {
                        "sentence_id": s.get("sentence_id", "unknown"),
                        "chunk_id": s.get("chunk_id", "unknown"),
                        "sent_idx": s.get("sent_idx", -1),
                        "span_start": s.get("span_start", s.get("sent_idx", -1)),
                        "span_end": s.get("span_end", s.get("sent_idx", -1)),
                        "expanded": s.get("expanded", False),
                        "score": s.get("score", 0),
                        "score_normalized": s.get("score_normalized", 0.0),
                        "text": s.get("text", ""),
                    }
                    for s in selected_sentences
                ],
                "sentence_prompt_tokens": sentence_tokens.get(
                    "sentence_prompt_tokens", 0
                ),
                "sentence_output_tokens": sentence_tokens.get(
                    "sentence_output_tokens", 0
                ),
                "sentence_total_tokens": sentence_tokens.get(
                    "sentence_total_tokens", 0
                ),
                "reader_prompt_tokens": answer.get("prompt_len", 0),
                "reader_output_tokens": answer.get("output_tokens", 0),
                "reader_total_tokens": answer.get("total_tokens", 0),
                "reader_budget": {
                    "context_window_tokens": answer.get("budget_context_window_tokens"),
                    "reserved_output_tokens": answer.get("budget_reserved_output_tokens"),
                    "safety_margin_tokens": answer.get("budget_safety_margin_tokens"),
                    "prompt_budget_tokens": answer.get("budget_prompt_tokens"),
                    "base_prompt_tokens": answer.get("budget_base_prompt_tokens"),
                    "context_budget_tokens": answer.get("budget_context_tokens"),
                    "max_passage_tokens": answer.get("budget_max_passage_tokens"),
                    "n_candidate_passages": answer.get("budget_n_candidate_passages"),
                    "n_selected_passages": answer.get("budget_n_selected_passages"),
                    "n_dropped_passages": answer.get("budget_n_dropped_passages"),
                    "n_truncated_passages": answer.get("budget_n_truncated_passages"),
                    "used_context_tokens": answer.get("budget_used_context_tokens"),
                },
                "wall_time_sec": round(query_wall_sec, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        predictions[q_id] = answer.get("normalised_answer", "")

    eval_predictions = dict(predictions)
    if resume and paths["answers"].exists():
        for row in load_jsonl(str(paths["answers"])):
            qid = row.get("question_id")
            if not qid:
                continue
            if qid in eval_predictions:
                continue
            eval_predictions[qid] = str(row.get("normalised_answer", ""))

    if not eval_predictions:
        print("No query predictions available.")
        return {}

    eval_gold = {qid: gold_all[qid] for qid in eval_predictions if qid in gold_all}
    if not eval_gold:
        print("No gold answers available for evaluated predictions.")
        return {}

    per_query = evaluate_answers(eval_predictions, eval_gold)
    agg_scores = aggregate_answer_scores(eval_predictions, eval_gold)

    metric_records = [
        {
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "retriever": retriever,
            "reader_model": reader_model,
            "question_id": qid,
            **m,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for qid, m in per_query.items()
    ]
    save_jsonl(str(paths["answer_metrics"]), metric_records)

    wall_time_sec_total = sum(wall_times)
    wall_time_sec_mean = (
        wall_time_sec_total / len(wall_times) if wall_times else 0.0
    )
    reader_wall_time_sec_total = sum(reader_wall_times)
    reader_wall_time_sec_mean = (
        reader_wall_time_sec_total / len(reader_wall_times)
        if reader_wall_times
        else 0.0
    )

    tokens_total = (
        token_totals["sentence_total_tokens"] + token_totals["reader_total_tokens"]
    )
    queries_total = len(per_query)
    tokens_per_query_mean = tokens_total / queries_total if queries_total else 0.0
    total_time_ms = wall_time_sec_total * 1000
    total_calls = token_totals["n_sentence_calls"] + token_totals["n_reader_calls"]
    throughput = compute_throughput_stats(
        tokens_total=tokens_total,
        t_total_ms=total_time_ms,
        num_queries=queries_total,
        n_reader_calls=total_calls,
        t_reader_ms=total_time_ms,
    )

    summary = {
        "meta": {
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "retriever": retriever,
            "passage_source": passage_source,
            "sentence_mode": sentence_mode,
            "tau_low": tau_low,
            "top_k": top_k_chunks,
            "reader_model": reader_model,
            "sentence_model": sentence_model,
            "n_chunks": len(metadata),
            "queries_total": queries_total,
        },
        "accuracy": {
            "mean_em": agg_scores["mean_em"],
            "mean_f1": agg_scores["mean_f1"],
        },
        "latency": {
            "wall_time_sec_total": round(wall_time_sec_total, 4),
            "wall_time_sec_mean": round(wall_time_sec_mean, 4),
            "reader_wall_time_sec_total": round(reader_wall_time_sec_total, 4),
            "reader_wall_time_sec_mean": round(reader_wall_time_sec_mean, 4),
        },
        "cost": {
            "overall_tokens_total": tokens_total,
            "overall_tokens_per_query_mean": round(tokens_per_query_mean, 4),
            "sentence_extractor_prompt_tokens_total": token_totals[
                "sentence_prompt_tokens"
            ],
            "sentence_extractor_output_tokens_total": token_totals[
                "sentence_output_tokens"
            ],
            "sentence_extractor_tokens_total": token_totals["sentence_total_tokens"],
            "sentence_extractor_calls_total": token_totals["n_sentence_calls"],
            "sentence_extractor_prompt_tokens_per_query_mean": round(
                token_totals["sentence_prompt_tokens"] / queries_total, 4
            )
            if queries_total
            else 0.0,
            "sentence_extractor_tokens_per_query_mean": round(
                token_totals["sentence_total_tokens"] / queries_total, 4
            )
            if queries_total
            else 0.0,
            "reader_prompt_tokens_total": token_totals["reader_prompt_tokens"],
            "reader_output_tokens_total": token_totals["reader_output_tokens"],
            "reader_tokens_total": token_totals["reader_total_tokens"],
            "reader_calls_total": token_totals["n_reader_calls"],
            "reader_prompt_tokens_per_query_mean": round(
                token_totals["reader_prompt_tokens"] / queries_total, 4
            )
            if queries_total
            else 0.0,
            "reader_tokens_per_query_mean": round(
                token_totals["reader_total_tokens"] / queries_total, 4
            )
            if queries_total
            else 0.0,
        },
        "throughput": {
            "total_time_ms": total_time_ms,
            **throughput,
        },
        "artifacts": {
            "answer_metrics_path": str(paths["answer_metrics"]),
            "answers_path": str(paths["answers"]),
        },
    }
    summary["meta"]["seed"] = seed
    summary["meta"]["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if hits_at_k_scores:
        summary["retrieval"] = {
            "mean_hits_at_k_ratio": round(
                sum(hits_at_k_scores) / len(hits_at_k_scores), 4
            ),
            "mean_recall_at_k_ratio": round(
                sum(recall_at_k_scores) / len(recall_at_k_scores), 4
            ),
            "mean_precision_at_k_ratio": round(
                sum(precision_at_k_scores) / len(precision_at_k_scores), 4
            )
            if precision_at_k_scores
            else 0.0,
        }

    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[summary] {dataset}/{split}/{retriever} "
        f"wall_time={wall_time_sec_total:.2f}s "
        f"tps={throughput.get('tokens_per_sec', 0.0):.2f}"
    )

    return summary


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Keep pipeline logs at INFO while silencing noisy dependency internals.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    
    if RETRIEVER_CONFIG.get("hybrid") or RETRIEVER_CONFIG.get("sparse"):
        logger.info(f"Using spaCy model: {SPACY_MODEL}")

    run_specs: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for split in SPLITS:
            questions_path = processed_dataset_paths(dataset, split)["questions"]
            if not Path(questions_path).exists():
                logger.warning(f"Skipping missing {dataset}/{split} questions: {questions_path}")
                continue
            for retriever, enabled in RETRIEVER_CONFIG.items():
                if not enabled:
                    continue
                for sentence_mode in SENTENCE_MODES:
                    for top_k_chunks in TOP_K_CHUNK_SWEEP:
                        for tau_low in TAU_LOW_SWEEP:
                            for seed in SEEDS:
                                for reader_context_budget_tokens in READER_CONTEXT_BUDGET_TOKENS_SWEEP:
                                    run_specs.append(
                                        {
                                            "dataset": dataset,
                                            "split": split,
                                            "retriever": retriever,
                                            "sentence_mode": sentence_mode,
                                            "top_k_chunks": top_k_chunks,
                                            "tau_low": tau_low,
                                            "seed": seed,
                                            "reader_context_budget_tokens": reader_context_budget_tokens,
                                        }
                                    )

    if not run_specs:
        logger.warning("No runnable DA_EXIT configurations found.")
        return

    with tqdm(total=len(run_specs), desc="DA_EXIT runs", unit="run") as run_pbar:
        for spec in run_specs:
            logger.info(
                f"Running DA_EXIT: dataset={spec['dataset']} split={spec['split']} retriever={spec['retriever']} "
                f"sentence_mode={spec['sentence_mode']} "
                f"top_k_chunks={spec['top_k_chunks']} tau_low={spec['tau_low']} "
                f"budget={spec['reader_context_budget_tokens']} seed={spec['seed']}"
            )

            run_da_exit(
                dataset=spec["dataset"],
                split=spec["split"],
                retriever=spec["retriever"],
                reader_model=READER_MODEL,
                sentence_model=SENTENCE_GRADER_MODEL,
                server_url=SERVER_URL,
                top_k_chunks=spec["top_k_chunks"],
                sentence_mode=spec["sentence_mode"],
                seed=spec["seed"],
                resume=RESUME,
                tau_low=spec["tau_low"],
                reader_context_budget_tokens=spec["reader_context_budget_tokens"],
                show_query_progress=False,
            )
            run_pbar.update(1)

if __name__ == "__main__":
    main()
