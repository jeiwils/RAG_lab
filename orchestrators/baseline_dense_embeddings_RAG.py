"""Module Overview
---------------
Run a simple dense Retrieval-Augmented Generation (RAG) pipeline.

This module reuses the existing FAISS indexes and embedding model to
answer questions directly. It retrieves the
most similar passages for each query and asks a reader model to
produce an answer from those passages. Per-query retrieval metrics such
as ``hits_at_k`` and ``recall_at_k`` are logged alongside the generated
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

from src.a3_representations.dense_representations import (
    get_embedding_model,
    load_faiss_index,
)
from src.b1_retrieval.dense_retrieval import faiss_search_topk
from src.a3_representations.representations_paths import dataset_rep_paths
from src.c2_generation.answer_gen import ask_llm_with_passages
from src.d1_evaluation.answer_metrics import (
    aggregate_answer_scores,
    evaluate_answers,
)
from src.d1_evaluation.retrieval_metrics import (
    compute_hits_at_k,
    compute_recall_at_k,
)
from src.d1_evaluation.stats_metrics import append_percentiles
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
    load_jsonl,
    processed_dataset_paths,
    save_jsonl,
)

__all__ = ["run_dense_rag"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


DEFAULT_TOP_K = 20
DEFAULT_SEED_TOP_K = DEFAULT_TOP_K

def run_dense_rag(
    dataset: str,
    split: str,
    reader_model: str,
    retriever: str = "dense",
    server_url: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Answer queries using dense retrieval over passages and evaluate EM/F1.

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
        Identifier for the passage retriever used (e.g. ``"dense"``).
    server_url: str, optional
        URL of the completion endpoint for ``reader_model``. When ``None``,
        the first matching server from :func:`get_server_configs` is used.
    top_k: int, optional
        Number of passages to retrieve for each query. Defaults to
        ``DEFAULT_TOP_K`` from this module.
    seed: int, optional
        Seed used to initialize :mod:`random` and :mod:`numpy` for
        deterministic behaviour.
    resume: bool, optional
        Resume a previously interrupted run by reusing existing answers and
        skipping already processed questions. When ``True``,
        :func:`src.utils.compute_resume_sets` determines which question IDs
        have been completed.

    Returns
    -------
    Dict[str, Any]
        Top-level metadata with ``dense_eval`` containing EM, F1,
        ``mean_hits_at_k`` and ``mean_recall_at_k`` scores across the
        query set.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    rep_paths = dataset_rep_paths(dataset, split)
    passages = list(load_jsonl(rep_paths["passages_jsonl"]))
    passage_lookup = {p["passage_id"]: p["text"] for p in passages}
    index = load_faiss_index(rep_paths["passages_index"])
    encoder = get_embedding_model()

    query_path = processed_dataset_paths(dataset, split)["questions"]
    queries = list(load_jsonl(query_path))

    variant = "dense" if seed is None else f"dense_seed{seed}"
    paths = get_result_paths(reader_model, dataset, split, variant)

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(paths["answers"]),
        items=queries,
        get_id=lambda q, i: q["question_id"],
        phase_label="Dense RAG",
        id_field="question_id",
    )
    paths["base"].mkdir(parents=True, exist_ok=True)
    if not resume and paths["answers"].exists():
        paths["answers"].unlink()

    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}
    hits_at_k_scores: Dict[str, float] = {}
    recall_at_k_scores: Dict[str, float] = {}

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
            q_emb = encoder.encode([q_text], normalize_embeddings=False)
            idxs, _ = faiss_search_topk(q_emb, index, top_k=top_k)
            passage_ids = [passages[i]["passage_id"] for i in idxs]
            hits_val = compute_hits_at_k(passage_ids, gold_passages, top_k)
            recall_val = compute_recall_at_k(passage_ids, gold_passages, top_k)
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
            return passage_ids, hits_val, recall_val, llm_out, reader_elapsed_sec

        (
            passage_ids,
            hits_val,
            recall_val,
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
    reader_wall_time_total_sec = timing_stats["reader_wall_time_total_sec"]
    reader_wall_time_mean_sec = timing_stats["reader_wall_time_mean_sec"]
    reader_wall_time_median_sec = timing_stats["reader_wall_time_median_sec"]
    wall_time_total_sec = sum(wall_times)
    wall_time_mean_sec = wall_time_total_sec / len(wall_times) if wall_times else 0.0
    wall_time_median_sec = float(np.median(wall_times)) if wall_times else 0.0
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

    dense_eval = {
        "EM": agg_scores["EM"],
        "F1": agg_scores["F1"],
        "reader_wall_time_total_sec": round(reader_wall_time_total_sec, 4),
        "reader_wall_time_mean_sec": round(reader_wall_time_mean_sec, 4),
        "reader_wall_time_median_sec": round(reader_wall_time_median_sec, 4),
        "wall_time_total_sec": round(wall_time_total_sec, 4),
        "wall_time_mean_sec": round(wall_time_mean_sec, 4),
        "wall_time_median_sec": round(wall_time_median_sec, 4),
    }
    if hits_at_k_scores:
        dense_eval["mean_hits_at_k"] = round(
            sum(hits_at_k_scores.values()) / len(hits_at_k_scores), 4
        )
    if recall_at_k_scores:
        dense_eval["mean_recall_at_k"] = round(
            sum(recall_at_k_scores.values()) / len(recall_at_k_scores), 4
        )

    metrics = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "model": reader_model,
        "retriever": retriever,
        "timestamp": now_ts,
        "dense_eval": dense_eval,
    }

    if seed is not None:
        metrics["seed"] = seed
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    t_reader_ms = timing_stats["t_reader_ms"]
    num_queries = len(per_query_reader)
    query_latency_ms = timing_stats["query_latency_ms"]
    call_latency_ms = timing_stats["call_latency_ms"]

    usage = {
        "per_query_reader": per_query_reader,
        **token_totals,
        "t_reader_ms": t_reader_ms,
        "num_queries": num_queries,
        "query_latency_ms": query_latency_ms,
        "call_latency_ms": call_latency_ms,
    }
    run_id = str(int(time.time()))  # Identifier to group token usage shards
    usage_path = paths["base"] / f"token_usage_{run_id}_{os.getpid()}.json"
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    merge_token_usage(paths["base"], run_id=run_id, cleanup=True)

    dense_eval.update(
        {
            **token_totals,
            "t_reader_ms": t_reader_ms,
            "num_queries": num_queries,
            "query_latency_ms": query_latency_ms,
            "call_latency_ms": call_latency_ms,
        }
    )

    tokens_total = dense_eval.get("reader_total_tokens", 0)
    t_total_ms = dense_eval.get("t_reader_ms", 0)
    throughput = compute_throughput_stats(
        tokens_total=tokens_total,
        t_total_ms=t_total_ms,
        num_queries=num_queries,
        n_reader_calls=token_totals["n_reader_calls"],
        t_reader_ms=t_reader_ms,
    )
    tps_overall = throughput["tps_overall"]
    query_qps_reader = throughput["query_qps_reader"]
    cps_reader = throughput["cps_reader"]
    dense_eval.update(
        {
            "tokens_total": tokens_total,
            "t_total_ms": t_total_ms,
            **throughput,
        }
    )
    token_usage_file = paths["base"] / "token_usage.json"
    try:
        with open(token_usage_file, "r", encoding="utf-8") as f:
            token_usage_data = json.load(f)
    except FileNotFoundError:
        token_usage_data = {}
    token_usage_data["query_qps_reader"] = query_qps_reader
    token_usage_data["cps_reader"] = cps_reader
    token_usage_data["num_queries"] = num_queries
    token_usage_data["query_latency_ms"] = query_latency_ms
    token_usage_data["call_latency_ms"] = call_latency_ms
    with open(token_usage_file, "w", encoding="utf-8") as f:
        json.dump(token_usage_data, f, indent=2)

    print(
        f"[summary] overall throughput: {tps_overall:.2f} tokens/s, "
        f"reader throughput: {query_qps_reader:.2f} queries/s, {cps_reader:.2f} calls/s, "
        f"reader latency: {query_latency_ms:.2f} ms/query, {call_latency_ms:.2f} ms/call"
    )
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    extra = append_percentiles(paths["answer_metrics"], paths["summary"])
    dense_eval.update(extra)

    return metrics


def main() -> None:
    DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
    SPLITS = ["dev"]

    READER_MODELS = ["deepseek-r1-distill-qwen-7b"]

    #     "qwen2.5-7b-instruct",
    #     "qwen2.5-14b-instruct",

    #     "deepseek-r1-distill-qwen-7b",
    #     "deepseek-r1-distill-qwen-14b",

    #     "state-of-the-moe-rp-2x7b",

    #     "qwen2.5-2x7b-moe-power-coder-v4"
    # ]

    SEEDS = [1, 2, 3]

    TOP_K = DEFAULT_SEED_TOP_K

    for seed in SEEDS:
        for dataset in DATASETS:
            for split in SPLITS:
                for reader in READER_MODELS:
                    print(
                        f"[Dense RAG] dataset={dataset} split={split} reader={reader} top_k={TOP_K} seed={seed}"
                    )
                    metrics = run_dense_rag(
                        dataset,
                        split,
                        reader_model=reader,
                        top_k=TOP_K,
                        seed=seed,
                        resume=True,
                    )
                    print(metrics)
    print("\nDense RAG complete.")


if __name__ == "__main__":
    main()

