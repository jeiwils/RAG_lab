"""Orchestrator for running multi-hop traversal over datasets."""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from src.a3_representations.dense_representations import (
    build_and_save_faiss_index,
    get_embedding_model,
    load_faiss_index,
)
from src.b1_retrieval.hybrid_retrieval import DEFAULT_HYBRID_ALPHA
from src.a3_representations.representations_paths import dataset_rep_paths
from src.b1_retrieval.traversal_seed_selection import DEFAULT_SEED_TOP_K, select_seed_passages
from src.b2_reranking.traversal_scoring_helpfulness import rerank_passages_by_helpfulness
from src.c1_reasoning.reasoning_paths import traversal_paths
from src.c1_reasoning.traversal import (
    DEFAULT_NUMBER_HOPS,
    DEFAULT_RETRIEVER_NAME,
    DEFAULT_TRAVERSAL_ALPHA,
    DEFAULT_TRAVERSAL_PROMPT,
    enhanced_traversal_algorithm,
    hoprag_traversal_algorithm,
    save_traversal_result,
    traverse_graph,
)
from src.d1_evaluation.retrieval_metrics import compute_hits_at_k, compute_recall_at_k
from src.d1_evaluation.traversal_metrics import (
    append_traversal_percentiles,
    compute_traversal_summary,
)
from src.d1_evaluation.usage_metrics import merge_token_usage
from src.utils.__utils__ import (
    compute_resume_sets,
    get_server_configs,
    get_server_urls,
    load_jsonl,
    processed_dataset_paths,
    run_multiprocess,
    split_jsonl_for_models,
    validate_vec_ids,
)

__all__ = ["process_query_batch", "process_traversal", "run_traversal"]


def run_traversal(
    query_data: List[Dict],
    graph,
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    server_configs: List[Dict],
    output_paths: Dict[str, Path],
    dataset: str,
    split: str,
    variant: str,
    traverser_model: str,
    retriever_name: str,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    n_hops: int = 2,
    traversal_alg: Optional[Callable] = None,
    traversal_prompt: Optional[str] = None,
    seed: int | None = None,
):
    """Run LLM-guided multi-hop traversal over a QA query set."""

    if traversal_alg is None:
        traversal_alg = hoprag_traversal_algorithm

    output_paths["base"].mkdir(parents=True, exist_ok=True)

    token_totals = {
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        "n_traversal_calls": 0,
        "t_traversal_ms": 0,
    }
    per_query_usage: Dict[str, Dict[str, int]] = {}

    if traversal_prompt is None:
        traversal_prompt = DEFAULT_TRAVERSAL_PROMPT

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for entry in tqdm(query_data, desc="queries"):
        question_id = entry["question_id"]
        query_text = entry["question"]
        gold_passages = entry["gold_passages"]
        query_token_totals = {
            "trav_prompt_tokens": 0,
            "trav_output_tokens": 0,
            "trav_tokens_total": 0,
            "n_traversal_calls": 0,
            "t_traversal_ms": 0,
        }
        print(f"\n[Query] {question_id} - \"{query_text}\"")

        query_emb = emb_model.encode(query_text, normalize_embeddings=False)
        norm = np.linalg.norm(query_emb)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                f"Query embedding norm invalid ({norm}); check emb_model.encode output."
            )

        seed_passages = select_seed_passages(
            query_text=query_text,
            query_emb=query_emb,
            passage_metadata=passage_metadata,
            passage_index=passage_index,
            seed_top_k=seed_top_k,
            alpha=alpha,
            question_id=question_id,
        )

        print(f"[Seeds] Retrieved {len(seed_passages)} passages.")

        hits_val = compute_hits_at_k(seed_passages, gold_passages, seed_top_k)
        recall_val = compute_recall_at_k(seed_passages, gold_passages, seed_top_k)

        visited_passages, ccount, hop_trace, stats = traverse_graph(
            graph=graph,
            query_text=query_text,
            query_emb=query_emb,
            passage_emb=passage_emb,
            seed_passages=seed_passages,
            n_hops=n_hops,
            server_configs=server_configs,
            traversal_alg=traversal_alg,
            alpha=alpha,
            traversal_prompt=traversal_prompt,
            token_totals=query_token_totals,
            seed=seed,
        )

        print(
            f"[Traversal] Visited {len(visited_passages)} passages "
            f"(None={stats['none_count']}, Repeat={stats['repeat_visit_count']})"
        )

        helpful_passages = rerank_passages_by_helpfulness(
            candidate_passages=visited_passages,
            visit_counts=ccount,
            graph=graph,
        )

        elapsed_ms = int(query_token_totals.get("t_traversal_ms", 0))
        elapsed = elapsed_ms / 1000

        n_calls_query = query_token_totals.get("n_traversal_calls", 0)
        query_token_totals["query_latency_ms"] = float(elapsed_ms)
        query_token_totals["call_latency_ms"] = (
            float(elapsed_ms) / n_calls_query if n_calls_query else 0.0
        )

        save_traversal_result(
            question_id=question_id,
            gold_passages=gold_passages,
            visited_passages=visited_passages,
            ccount=ccount,
            hop_trace=hop_trace,
            traversal_alg=traversal_alg,
            helpful_passages=helpful_passages,
            hits_at_k=hits_val,
            recall_at_k=recall_val,
            dataset=dataset,
            split=split,
            variant=variant,
            retriever_name=retriever_name,
            traverser_model=traverser_model,
            traversal_wall_time_sec=elapsed,
            output_path=output_paths["results"],
            token_usage=query_token_totals,
            seed=seed,
        )

        for k in token_totals:
            token_totals[k] += query_token_totals.get(k, 0)
        per_query_usage[question_id] = dict(query_token_totals)

    base_usage_path = output_paths.get(
        "token_usage", output_paths["base"] / "token_usage.json"
    )
    unique = f"{os.getpid()}_{int(time.time())}"
    token_usage_path = base_usage_path.with_name(
        f"{base_usage_path.stem}_{unique}{base_usage_path.suffix}"
    )
    global_usage = {k: v for k, v in token_totals.items()}

    tokens_total = global_usage.get("trav_tokens_total", 0)
    t_total_ms = global_usage.get("t_traversal_ms", 0)
    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0

    num_queries = len(per_query_usage)
    n_traversal_calls = global_usage.get("n_traversal_calls", 0)
    query_latency_ms = t_total_ms / num_queries if num_queries else 0.0
    call_latency_ms = t_total_ms / n_traversal_calls if n_traversal_calls else 0.0
    query_qps_traversal = num_queries / (t_total_ms / 1000) if t_total_ms else 0.0
    cps_traversal = (
        n_traversal_calls / (t_total_ms / 1000) if t_total_ms else 0.0
    )
    global_usage.update(
        {
            "tokens_total": tokens_total,
            "t_total_ms": t_total_ms,
            "tps_overall": tps_overall,
            "query_qps_traversal": query_qps_traversal,
            "cps_traversal": cps_traversal,
            "num_queries": num_queries,
            "query_latency_ms": query_latency_ms,
            "call_latency_ms": call_latency_ms,
        }
    )

    usage = {"per_query_traversal": per_query_usage, **global_usage}
    with open(token_usage_path, "wt", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    print(f"[summary] total traversal tokens: {tokens_total}")
    print(f"[summary] traversal LLM inference time: {t_total_ms} ms")
    print(f"[summary] average query latency: {query_latency_ms:.2f} ms")
    print(f"[summary] average call latency: {call_latency_ms:.2f} ms")
    print(
        "[summary] overall throughput: "
        f"{tps_overall:.2f} tokens/s, "
        f"{query_qps_traversal:.2f} queries/s, "
        f"{cps_traversal:.2f} calls/s"
    )

    return global_usage


def process_query_batch(cfg: Dict) -> None:
    """Run traversal on a shard of queries and write partial outputs."""

    server_url = cfg["server_url"]
    model = cfg["model"]
    input_path = cfg["input_path"]
    resume = cfg.get("resume", False)
    resume_path = cfg["resume_path"]

    server_config = next(
        (s for s in get_server_configs(model) if s["server_url"] == server_url),
        None,
    )
    if server_config is None:
        raise ValueError(f"Server URL {server_url} not found for model {model}")

    emb_model = get_embedding_model()
    queries = list(load_jsonl(input_path))

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(resume_path),
        items=queries,
        get_id=lambda q, i: q["question_id"],
        phase_label="Traversal",
        id_field="question_id",
    )
    if resume:
        queries = [q for q in queries if q["question_id"] not in done_ids]
    if not queries:
        return

    run_traversal(
        query_data=queries,
        graph=cfg["graph"],
        passage_metadata=cfg["passage_metadata"],
        passage_emb=cfg["passage_emb"],
        passage_index=cfg["passage_index"],
        emb_model=emb_model,
        server_configs=[server_config],
        output_paths=cfg["output_paths"],
        dataset=cfg["dataset"],
        split=cfg["split"],
        variant=cfg["variant"],
        traverser_model=cfg["traverser_model"],
        retriever_name=cfg["retriever_name"],
        seed_top_k=cfg.get("seed_top_k", DEFAULT_SEED_TOP_K),
        alpha=cfg.get("alpha", DEFAULT_TRAVERSAL_ALPHA),
        n_hops=cfg.get("n_hops", DEFAULT_NUMBER_HOPS),
        traversal_alg=cfg["traversal_alg"],
        seed=cfg.get("seed"),
    )


def process_traversal(cfg: Dict) -> None:
    """Load resources and run traversal for a single configuration."""

    dataset = cfg["dataset"]
    graph_model = cfg["graph_model"]
    model = cfg["model"]
    variant = cfg["variant"]
    split = cfg["split"]
    resume = cfg["resume"]
    seed = cfg.get("seed")
    retriever_name = cfg.get("retriever_name", DEFAULT_RETRIEVER_NAME)

    variant_cfg = {
        "baseline": hoprag_traversal_algorithm,
        "enhanced": enhanced_traversal_algorithm,
    }
    if variant not in variant_cfg:
        raise ValueError(f"Unknown traversal variant: {variant}")

    variant_for_path = f"{variant}_seed{seed}" if seed is not None else variant

    print(
        f"[Run] dataset={dataset} graph_model={graph_model} traversal_model={model} "
        f"variant={variant_for_path} split={split}"
    )

    paths = dataset_rep_paths(dataset, split)
    passage_metadata = list(load_jsonl(paths["passages_jsonl"]))
    passage_emb = np.load(paths["passages_emb"])
    validate_vec_ids(passage_metadata, passage_emb)

    passage_index = load_faiss_index(paths["passages_index"])
    if passage_index.ntotal != len(passage_metadata):
        print(
            f"[process_traversal] FAISS index has {passage_index.ntotal} vectors "
            f"but metadata lists {len(passage_metadata)} passages. Rebuilding index."
        )
        output_dir = str(Path(paths["passages_index"]).parent)
        build_and_save_faiss_index(
            passage_emb,
            dataset,
            "passages",
            output_dir=output_dir,
        )
        passage_index = load_faiss_index(paths["passages_index"])
        assert passage_index.ntotal == len(passage_metadata), (
            "FAISS index rebuild failed to match metadata length"
        )

    query_path = processed_dataset_paths(dataset, split)["questions"]

    graph_path = Path(
        f"data/graphs/{graph_model}/{dataset}/{split}/{variant}/{dataset}_{split}_graph.gpickle"
    )
    with open(graph_path, "rb") as f:
        graph_obj = pickle.load(f)
    validate_vec_ids(
        [
            {"passage_id": pid, "vec_id": data.get("vec_id")}
            for pid, data in graph_obj.nodes(data=True)
        ],
        passage_emb,
    )

    trav_alg = variant_cfg[variant]
    output_paths = traversal_paths(model, dataset, split, variant_for_path)

    urls = get_server_urls(model)
    shards = split_jsonl_for_models(str(query_path), model, resume=resume)

    run_id = str(int(time.time()))
    batch_configs = []
    for i, (url, shard) in enumerate(zip(urls, shards)):
        batch_paths = {
            "base": output_paths["base"],
            "results": output_paths["base"] / f"results_part{i}.jsonl",
            "visited_passages": output_paths["base"] / f"visited_passages_part{i}.json",
            "token_usage": output_paths["base"] / f"token_usage_{run_id}_part{i}.json",
        }
        batch_configs.append(
            {
                "input_path": shard,
                "graph": graph_obj,
                "passage_metadata": passage_metadata,
                "passage_emb": passage_emb,
                "passage_index": passage_index,
                "server_url": url,
                "model": model,
                "output_paths": batch_paths,
                "traversal_alg": trav_alg,
                "resume": resume,
                "resume_path": output_paths["results"],
                "seed": seed,
                "dataset": dataset,
                "split": split,
                "variant": variant_for_path,
                "traverser_model": model,
                "retriever_name": retriever_name,
                "alpha": DEFAULT_TRAVERSAL_ALPHA,
                "n_hops": DEFAULT_NUMBER_HOPS,
                "seed_top_k": DEFAULT_SEED_TOP_K,
            }
        )

    run_multiprocess(process_query_batch, batch_configs)

    merge_token_usage(output_paths["base"], run_id=run_id, cleanup=True)

    new_ids: Set[str] = set()
    with open(output_paths["results"], "at", encoding="utf-8") as fout:
        for i in range(len(urls)):
            part_path = output_paths["base"] / f"results_part{i}.jsonl"
            if part_path.exists():
                with open(part_path, "rt", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        obj = json.loads(line)
                        new_ids.add(obj.get("question_id"))
                part_path.unlink()

    merged_passages: Set[str] = set()
    for i in range(len(urls)):
        part_path = output_paths["base"] / f"visited_passages_part{i}.json"
        if part_path.exists():
            with open(part_path, "rt", encoding="utf-8") as fin:
                merged_passages.update(json.load(fin))
            part_path.unlink()
    visited_path = output_paths["visited_passages"]
    with open(visited_path, "wt", encoding="utf-8") as fout:
        json.dump(sorted(merged_passages), fout, indent=2)

    traversal_metrics = compute_traversal_summary(
        output_paths["results"], include_ids=new_ids
    )
    stats_payload = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "model": model,
        "retriever": retriever_name,
        "seed": seed,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "traversal_eval": traversal_metrics,
    }
    with open(output_paths["stats"], "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    append_traversal_percentiles(output_paths["results"], output_paths["stats"])

    print(
        f"[Done] dataset={dataset} graph_model={graph_model} traversal_model={model} "
        f"variant={variant_for_path} split={split}"
    )


def main() -> None:
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    GRAPH_MODELS = ["llama-3.1-8b-instruct"]
    TRAVERSAL_MODELS = ["deepseek-r1-distill-qwen-7b"]
    VARIANTS = ["baseline"]
    RESUME = True
    SPLIT = "dev"
    SEEDS = [1, 2, 3]

    configs = [
        {
            "dataset": d,
            "graph_model": gm,
            "model": tm,
            "variant": v,
            "split": SPLIT,
            "resume": RESUME,
            "seed": seed,
        }
        for d in DATASETS
        for gm in GRAPH_MODELS
        for tm in TRAVERSAL_MODELS
        for v in VARIANTS
        for seed in SEEDS
    ]

    result_paths = set()
    for cfg in configs:
        seed = cfg.get("seed")
        variant_for_path = f"{cfg['variant']}_seed{seed}" if seed is not None else cfg["variant"]
        out_path = (
            Path(
                f"data/traversal/{cfg['model']}/{cfg['dataset']}/{cfg['split']}/{variant_for_path}"
            )
            / "per_query_traversal_results.jsonl"
        )
        if out_path in result_paths:
            raise ValueError(f"Duplicate output path detected: {out_path}")
        result_paths.add(out_path)

    configs_by_model: Dict[str, List[Dict]] = {}
    for cfg in configs:
        configs_by_model.setdefault(cfg["model"], []).append(cfg)

    for _, model_configs in configs_by_model.items():
        for cfg in model_configs:
            process_traversal(cfg)


if __name__ == "__main__":
    main()
