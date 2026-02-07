"""Module Overview
---------------
Run a simple Retrieval-Augmented Generation (RAG) pipeline.

This module reuses the existing passage representations to answer questions
directly. It supports dense, sparse, or hybrid retrieval, then asks a reader
model to produce an answer from those passages. Per-query retrieval metrics
such as ``hits_at_k`` and ``recall_at_k`` are logged alongside the generated
answers.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.a2_indexing.retrieval_indexing import _retrieve_chunk_indices
from src.a3_representations.dense_representations import (
    get_embedding_model,
    load_faiss_index,
)
from src.a3_representations.representations_paths import dataset_rep_paths
from src.b1_retrieval import DEFAULT_HYBRID_ALPHA
from src.c2_generation.answer_gen import ask_llm_with_passages
from src.d1_evaluation.answer_metrics import (
    aggregate_answer_scores,
    evaluate_answers,
)
from src.d1_evaluation.retrieval_metrics import (
    compute_hits_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from src.d1_evaluation.timing_metrics import (
    compute_throughput_stats,
    summarize_reader_wall_times,
    wall_time,
)
from src.d1_evaluation.usage_metrics import merge_token_usage
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

__all__ = ["run_baseline_rag", "run_dense_rag"]

# ---------------------------------------------------------------------------
# #### Configs
# ---------------------------------------------------------------------------

# Defaults used by run_baseline_rag
TOP_K_SWEEP = [1, 5, 10, 20]
RETRIEVER_CONFIG = {
    "dense": False,
    "sparse": False,
    "hybrid": True,
}

# Defaults used by main()
DATASETS = ["hotpotqa"] #["hotpotqa", "2wikimultihopqa", "musique", "natural_questions"]
# Use val for tuning, dev for final metrics.
SPLITS = ["dev"] #["val", "dev"]
READER_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]
SERVER_URL = "http://localhost:8005"
SEEDS = [1, 2, 3] # [1] #
MAX_QUESTIONS: int | None = 1000  # set to None to use the full split
SHUFFLE_QUESTIONS = False
# Passage source config (default is sentence-level passages).
PASSAGE_SOURCE = "full_passages_chunks"  # "passages" | "full_passages_chunks" | "full_passages_chunks_discourse_aware"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_baseline_rag(
    dataset: str,
    split: str,
    reader_model: str,
    retriever: str,
    server_url: str | None = None,
    top_k: int = 20,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    seed: int | None = None,
    resume: bool = False,
    passage_source: str = "passages",
) -> Dict[str, Any]:
    """Answer queries using dense/sparse/hybrid retrieval over passages and evaluate EM/F1.

    The function retrieves top-``k`` passages for each query and asks a reader
    model to generate an answer. Retrieval metrics ``hits_at_k`` and
    ``recall_at_k`` are computed per query and included in both the per-query
    JSONL output and the summary metrics file.

    Parameters
    ----------
    dataset: str
        Name of the dataset (e.g. ``"hotpotqa"``).
    split: str
        Dataset split (e.g. ``"dev"``).
    reader_model: str
        Name of the reader model used to generate answers. ``server_url``
        defaults to the first entry returned by
        :func:`src.utils.get_server_configs` for this model when ``None``.
    retriever: str, optional
        Identifier for the passage retriever used (``"dense"``, ``"sparse"``,
        or ``"hybrid"``).
    server_url: str, optional
        URL of the completion endpoint for ``reader_model``. When ``None``,
        the first matching server from :func:`get_server_configs` is used.
    top_k: int, optional
        Number of passages to retrieve for each query. Defaults to
        ``DEFAULT_TOP_K`` from this module.
    alpha: float, optional
        Weight for dense vs. sparse similarity when using hybrid retrieval.
    seed: int, optional
        Seed used to initialize :mod:`random` and :mod:`numpy` for
        deterministic behaviour.
    resume: bool, optional
        Resume a previously interrupted run by reusing existing answers and
        skipping already processed questions. When ``True``,
        :func:`src.utils.compute_resume_sets` determines which question IDs
        have been completed.
    passage_source: str, optional
        Passage source to load representations from. Use ``"passages"`` for
        sentence-level passages or ``"full_passages_chunks"`` to match the
        DA_EXIT chunk retrieval unit.

    Returns
    -------
    Dict[str, Any]
        Nested summary schema with ``meta``, ``accuracy``, ``latency``,
        ``cost``, and optional ``retrieval``/``throughput`` sections.
    """

    if retriever not in RETRIEVER_CONFIG:
        raise ValueError(
            f"Unknown retriever '{retriever}'. Choose from {sorted(RETRIEVER_CONFIG)}."
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    rep_paths = dataset_rep_paths(dataset, split, passage_source=passage_source)
    passages = list(load_jsonl(rep_paths["passages_jsonl"]))
    passage_lookup = {p["passage_id"]: p["text"] for p in passages}
    index = None
    encoder = None
    if retriever in {"dense", "hybrid"}:
        index = load_faiss_index(rep_paths["passages_index"])
        encoder = get_embedding_model()

    query_path = processed_dataset_paths(dataset, split)["questions"]
    queries = limit_questions(
        list(load_jsonl(query_path)),
        max_questions=MAX_QUESTIONS,
        seed=seed,
        shuffle=SHUFFLE_QUESTIONS,
    )

    variant = f"{retriever}_k{top_k}"
    if passage_source and passage_source != "passages":
        variant = f"{variant}_{passage_source}"
    if seed is not None:
        variant = f"{variant}_seed{seed}"
    paths = get_result_paths(reader_model, dataset, split, variant)

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(paths["answers"]),
        items=queries,
        get_id=lambda q, i: q["question_id"],
        phase_label=f"Baseline RAG ({retriever})",
        id_field="question_id",
    )
    paths["base"].mkdir(parents=True, exist_ok=True)
    if not resume and paths["answers"].exists():
        paths["answers"].unlink()

    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}
    hits_at_k_scores: Dict[str, float] = {}
    recall_at_k_scores: Dict[str, float] = {}
    precision_at_k_scores: Dict[str, float] = {}

    token_totals = {
        "reader_prompt_tokens": 0,
        "reader_output_tokens": 0,
        "reader_total_tokens": 0,
        "n_reader_calls": 0,
    }
    per_query_reader: Dict[str, Dict[str, int]] = {}
    reader_wall_times: List[float] = []
    wall_times: List[float] = []
    for q in tqdm(queries, desc="queries"):
        q_id = q["question_id"]
        if resume and q_id in done_ids:
            continue
        q_text = q["question"]
        gold_passages = q.get("gold_passages", [])
        gold[q_id] = [q.get("gold_answer", "")]

        print(f"\n[Query] {q_id} - \"{q_text}\"")
        def _run_query():
            query_vec = None
            if retriever in {"dense", "hybrid"} and encoder is not None:
                query_vec = encoder.encode([q_text], normalize_embeddings=False)
            idxs = _retrieve_chunk_indices(
                retriever,
                q_text,
                query_vec,
                passages,
                index,
                top_k=top_k,
                alpha=alpha,
            )
            passage_ids = [passages[i]["passage_id"] for i in idxs]
            hits_val = compute_hits_at_k(passage_ids, gold_passages, top_k)
            recall_val = compute_recall_at_k(passage_ids, gold_passages, top_k)
            precision_val = compute_precision_at_k(passage_ids, gold_passages, top_k)
            llm_out, reader_elapsed_sec = wall_time(
                ask_llm_with_passages,
                query_text=q_text,
                passage_ids=passage_ids,
                graph=None,
                server_url=server_url,
                passage_lookup=passage_lookup,
                model_name=reader_model,
                top_k_answer_passages=top_k,
                seed=seed,
            )
            return (
                passage_ids,
                hits_val,
                recall_val,
                precision_val,
                llm_out,
                reader_elapsed_sec,
            )

        (
            passage_ids,
            hits_val,
            recall_val,
            precision_val,
            llm_out,
            reader_elapsed_sec,
        ), query_wall_sec = wall_time(_run_query)
        elapsed_sec = reader_elapsed_sec
        elapsed_ms = int(elapsed_sec * 1000)
        reader_wall_times.append(elapsed_sec)
        wall_times.append(query_wall_sec)

        append_jsonl(
            str(paths["answers"]),
            {
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "retriever": retriever,
                "reader_model": reader_model,
                "question_id": q_id,
                "question": q_text,
                "raw_answer": llm_out["raw_answer"],
                "normalised_answer": llm_out["normalised_answer"],
                "used_passages": passage_ids,
                "hits_at_k": hits_val,
                "recall_at_k": recall_val,
                "precision_at_k": precision_val,
                "prompt_len": llm_out.get("prompt_len", 0),
                "output_tokens": llm_out.get("output_tokens", 0),
                "total_tokens": llm_out.get("total_tokens", 0),
                "reader_prompt_tokens": llm_out.get("prompt_len", 0),
                "reader_output_tokens": llm_out.get("output_tokens", 0),
                "reader_total_tokens": llm_out.get(
                    "total_tokens",
                    llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0),
                ),
                "t_reader_ms": elapsed_ms,
                "reader_wall_time_sec": round(elapsed_sec, 4),
                "wall_time_sec": round(query_wall_sec, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "seed": seed,
            },
        )
        predictions[q_id] = llm_out["normalised_answer"]
        hits_at_k_scores[q_id] = hits_val
        recall_at_k_scores[q_id] = recall_val
        precision_at_k_scores[q_id] = precision_val

        token_totals["n_reader_calls"] += 1
        token_totals["reader_prompt_tokens"] += llm_out.get("prompt_len", 0)
        token_totals["reader_output_tokens"] += llm_out.get("output_tokens", 0)
        token_totals["reader_total_tokens"] += llm_out.get(
            "total_tokens", llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0)
        )
        n_calls = 1
        per_query_reader[q_id] = {
            "reader_prompt_tokens": llm_out.get("prompt_len", 0),
            "reader_output_tokens": llm_out.get("output_tokens", 0),
            "reader_total_tokens": llm_out.get(
                "total_tokens",
                llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0),
            ),
            "n_reader_calls": n_calls,
            "t_reader_ms": elapsed_ms,
            "reader_wall_time_sec": round(elapsed_sec, 4),
            "wall_time_sec": round(query_wall_sec, 4),
            "query_latency_ms": elapsed_ms,
            "call_latency_ms": elapsed_ms / max(n_calls, 1),
        }

    if not gold:
        print("No new queries to process.")
        return {}

    per_query = evaluate_answers(predictions, gold)
    agg_scores = aggregate_answer_scores(predictions, gold)

    timing_stats = summarize_reader_wall_times(
        reader_wall_times,
        n_reader_calls=token_totals["n_reader_calls"],
    )
    reader_wall_time_sec_total = timing_stats["reader_wall_time_sec_total"]
    reader_wall_time_sec_mean = timing_stats["reader_wall_time_sec_mean"]
    reader_wall_time_sec_median = timing_stats["reader_wall_time_sec_median"]
    wall_time_sec_total = sum(wall_times)
    wall_time_sec_mean = wall_time_sec_total / len(wall_times) if wall_times else 0.0
    wall_time_sec_median = float(np.median(wall_times)) if wall_times else 0.0
    now_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    metric_records = [
        {
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "retriever": retriever,
            "reader_model": reader_model,
            "question_id": qid,
            **m,
            "timestamp": now_ts,
        }
        for qid, m in per_query.items()
    ]
    if resume:
        for rec in metric_records:
            append_jsonl(str(paths["answer_metrics"]), rec)
    else:
        save_jsonl(str(paths["answer_metrics"]), metric_records)

    reader_time_ms_total = timing_stats["reader_time_ms_total"]
    queries_total = len(per_query)
    query_latency_ms = timing_stats["query_latency_ms"]
    call_latency_ms = timing_stats["call_latency_ms"]

    usage = {
        "per_query_reader": per_query_reader,
        **token_totals,
        "t_reader_ms": reader_time_ms_total,
        "num_queries": queries_total,
        "query_latency_ms": query_latency_ms,
        "call_latency_ms": call_latency_ms,
    }
    run_id = str(int(time.time()))  # Identifier to group token usage shards
    usage_path = paths["base"] / f"token_usage_{run_id}_{os.getpid()}.json"
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    token_usage_path = merge_token_usage(paths["base"], run_id=run_id, cleanup=True)

    tokens_total = token_totals.get("reader_total_tokens", 0)
    tokens_per_query_mean = tokens_total / queries_total if queries_total else 0.0
    total_time_ms = wall_time_sec_total * 1000
    throughput = compute_throughput_stats(
        tokens_total=tokens_total,
        t_total_ms=total_time_ms,
        num_queries=queries_total,
        n_reader_calls=token_totals["n_reader_calls"],
        t_reader_ms=total_time_ms,
    )
    tokens_per_sec = throughput["tokens_per_sec"]
    queries_per_sec = throughput["queries_per_sec"]
    calls_per_sec = throughput["calls_per_sec"]

    retrieval: Dict[str, float] = {}
    if hits_at_k_scores:
        retrieval["mean_hits_at_k_ratio"] = round(
            sum(hits_at_k_scores.values()) / len(hits_at_k_scores), 4
        )
    if recall_at_k_scores:
        retrieval["mean_recall_at_k_ratio"] = round(
            sum(recall_at_k_scores.values()) / len(recall_at_k_scores), 4
        )
    if precision_at_k_scores:
        retrieval["mean_precision_at_k_ratio"] = round(
            sum(precision_at_k_scores.values()) / len(precision_at_k_scores), 4
        )

    summary: Dict[str, Any] = {
        "meta": {
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "retriever": retriever,
            "reader_model": reader_model,
            "passage_source": passage_source,
            "top_k": top_k,
            "n_chunks": len(passages),
            "queries_total": queries_total,
            "timestamp": now_ts,
        },
        "accuracy": {
            "mean_em": agg_scores["mean_em"],
            "mean_f1": agg_scores["mean_f1"],
        },
        "latency": {
            "wall_time_sec_total": round(wall_time_sec_total, 4),
            "wall_time_sec_mean": round(wall_time_sec_mean, 4),
            "wall_time_sec_median": round(wall_time_sec_median, 4),
            "reader_wall_time_sec_total": round(reader_wall_time_sec_total, 4),
            "reader_wall_time_sec_mean": round(reader_wall_time_sec_mean, 4),
            "reader_wall_time_sec_median": round(reader_wall_time_sec_median, 4),
        },
        "cost": {
            "tokens_total": tokens_total,
            "tokens_per_query_mean": round(tokens_per_query_mean, 4),
            "reader_prompt_tokens_total": token_totals.get("reader_prompt_tokens", 0),
            "reader_output_tokens_total": token_totals.get("reader_output_tokens", 0),
            "reader_tokens_total": token_totals.get("reader_total_tokens", 0),
            "reader_calls_total": token_totals.get("n_reader_calls", 0),
            "reader_prompt_tokens_per_query_mean": round(
                token_totals.get("reader_prompt_tokens", 0) / queries_total, 4
            )
            if queries_total
            else 0.0,
        },
        "throughput": {
            "total_time_ms": total_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "queries_per_sec": queries_per_sec,
            "calls_per_sec": calls_per_sec,
        },
        "artifacts": {
            "token_usage_path": str(token_usage_path),
            "answer_metrics_path": str(paths["answer_metrics"]),
            "answers_path": str(paths["answers"]),
        },
    }
    if retrieval:
        summary["retrieval"] = retrieval
    if seed is not None:
        summary["meta"]["seed"] = seed

    print(
        f"[summary] overall throughput: {tokens_per_sec:.2f} tokens/s, "
        f"{queries_per_sec:.2f} queries/s, {calls_per_sec:.2f} calls/s, "
        f"latency: {query_latency_ms:.2f} ms/query, {call_latency_ms:.2f} ms/call"
    )
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_dense_rag(*args, **kwargs) -> Dict[str, Any]:
    """Backward-compatible alias for run_baseline_rag."""
    return run_baseline_rag(*args, **kwargs)


def main() -> None:
    for seed in SEEDS:
        for dataset in DATASETS:
            for split in SPLITS:
                questions_path = processed_dataset_paths(dataset, split)["questions"]
                if not os.path.exists(questions_path):
                    print(f"[skip] missing {dataset}/{split} questions: {questions_path}")
                    continue
                for reader in READER_MODELS:
                    for retriever, enabled in RETRIEVER_CONFIG.items():
                        if not enabled:
                            continue
                        for top_k in TOP_K_SWEEP:
                            print(
                                f"[Baseline RAG] dataset={dataset} split={split} retriever={retriever} "
                                f"reader={reader} top_k={top_k} seed={seed}"
                            )
                            metrics = run_baseline_rag(
                                dataset,
                                split,
                                reader_model=reader,
                                server_url=SERVER_URL,
                                retriever=retriever,
                                top_k=top_k,
                                alpha=DEFAULT_HYBRID_ALPHA,
                                seed=seed,
                                resume=True,
                                passage_source=PASSAGE_SOURCE,
                            )
                            print(metrics)
    print("\nBaseline RAG complete.")


if __name__ == "__main__":
    main()

