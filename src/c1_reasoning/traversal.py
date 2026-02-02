"""Traversal utilities for multi-hop reasoning.

Sections
--------
Traversal algorithms
    - llm_choose_edge
    - hoprag_traversal_algorithm
    - enhanced_traversal_algorithm
Traversal orchestration
    - traverse_graph
    - save_traversal_result
"""

from __future__ import annotations

from datetime import datetime
import re
import time
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np

from src.a3_representations.sparse_representations import extract_keywords
from src.b1_retrieval.hybrid_retrieval import DEFAULT_HYBRID_ALPHA
from src.b1_retrieval.sparse_retrieval import bm25_score, compute_bm25_stats
from src.d1_evaluation.traversal_metrics import (
    compute_gold_attention,
    compute_hop_metrics,
)
from src.utils.__utils__ import append_jsonl
from src.utils.x_config import LLM_DEFAULTS, MAX_TOKENS, TEMPERATURE
from src.utils.z_llm_utils import is_r1_like, query_llm, strip_think

__all__ = [
    "DEFAULT_NUMBER_HOPS",
    "DEFAULT_RETRIEVER_NAME",
    "DEFAULT_TRAVERSAL_ALPHA",
    "DEFAULT_TRAVERSAL_PROMPT",
    "TraversalOutputError",
    "llm_choose_edge",
    "hoprag_traversal_algorithm",
    "enhanced_traversal_algorithm",
    "traverse_graph",
    "save_traversal_result",
]


DEFAULT_NUMBER_HOPS = 4
DEFAULT_RETRIEVER_NAME = "hybrid"
DEFAULT_TRAVERSAL_ALPHA = 1.0

DEFAULT_TRAVERSAL_PROMPT = (
    "You are selecting the next auxiliary question to answer a multi-hop query.\n"
    "Main question:\n{MAIN_QUESTION}\n\n"
    "Candidate auxiliary questions:\n{CANDIDATE_LIST}\n\n"
    "Return the index of the best candidate, or 'null' if none apply."
)


class TraversalOutputError(Exception):
    """Raised when the traversal LLM returns an invalid edge selection."""


def llm_choose_edge(
    query_text: str,
    candidate_edges: list,
    graph: nx.DiGraph,
    server_configs: list,
    traversal_prompt: str,
    token_totals: Optional[Dict[str, int]] = None,
    reason: bool = True,
    seed: int | None = None,
):
    """Pick a single outgoing edge using the LLM and return the chosen pair.

    ``candidate_edges`` must be pre-sorted by the caller to define deterministic
    ordering. ``graph`` is accepted for interface compatibility but not used.
    """

    if not candidate_edges:
        return None

    oq_server = server_configs[1] if len(server_configs) > 1 else server_configs[0]
    k = len(candidate_edges)

    option_lines = [
        f"{i}. {edge_data.get('oq_text', '')}"
        for i, (_, edge_data) in enumerate(candidate_edges)
    ]
    options = "\n".join(option_lines)
    prompt = traversal_prompt.format(
        MAIN_QUESTION=query_text,
        CANDIDATE_LIST=options,
    )

    grammar_choices = " | ".join([f'"{i}"' for i in range(k)] + ['"null"'])
    grammar = f"root ::= {grammar_choices}\n"

    def _record_usage(usage: Optional[dict]):
        if token_totals is not None and usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            token_totals["trav_prompt_tokens"] += prompt_tokens
            token_totals["trav_output_tokens"] += completion_tokens
            token_totals["trav_tokens_total"] += usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )

    start = time.perf_counter()
    answer, usage = query_llm(
        prompt,
        server_url=oq_server["server_url"],
        max_tokens=MAX_TOKENS["edge_selection"],
        temperature=TEMPERATURE["edge_selection"],
        model_name=oq_server["model"],
        phase="edge_selection",
        stop=None,
        reason=reason,
        grammar=grammar,
        seed=seed,
        **LLM_DEFAULTS,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if token_totals is not None:
        token_totals["n_traversal_calls"] += 1
        token_totals["t_traversal_ms"] += elapsed_ms

    _record_usage(usage)

    if is_r1_like(oq_server["model"]):
        answer = strip_think(answer)

    answer = answer.strip()
    retry_count = 0
    while True:
        if answer == "null":
            mode = "null"
            break
        if re.fullmatch(r"[0-9]+", answer):
            mode = "int"
            break
        if retry_count == 0:
            retry_count = 1
            start = time.perf_counter()
            answer, usage = query_llm(
                prompt,
                server_url=oq_server["server_url"],
                max_tokens=MAX_TOKENS["edge_selection"],
                temperature=TEMPERATURE["edge_selection"],
                model_name=oq_server["model"],
                phase="edge_selection",
                stop=None,
                reason=reason,
                grammar=grammar,
                seed=seed,
                **LLM_DEFAULTS,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            if token_totals is not None:
                token_totals["n_traversal_calls"] += 1
                token_totals["t_traversal_ms"] += elapsed_ms
            _record_usage(usage)
            if is_r1_like(oq_server["model"]):
                answer = strip_think(answer)
            answer = answer.strip()
            continue
        raise TraversalOutputError(answer)

    print(f"[Traversal] grammar=dynamic retry={retry_count} mode={mode}")

    if mode == "null":
        return None

    idx = int(answer)
    if not (0 <= idx < len(candidate_edges)):
        raise TraversalOutputError(answer)

    print(f"[Traversal] selected idx={idx}")
    return candidate_edges[idx]


###############################################################################
# Traversal algorithms
###############################################################################


def hoprag_traversal_algorithm(
    vj,
    graph: nx.DiGraph,
    query_text: str,
    visited_passages,
    server_configs,
    ccount,
    next_Cqueue,
    hop_log,
    state,
    traversal_prompt: str,
    hop: int = 0,
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,
    **kwargs,
):
    """Single HopRAG traversal step used by ``traverse_graph``."""

    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
        if (vj, vk, graph[vj][vk]["oq_id"], graph[vj][vk]["iq_id"]) not in state["Evisited"]
    ]
    if not candidates:
        return set()

    candidates.sort(key=lambda item: (item[1].get("oq_id", ""), item[0]))
    hop_log["candidate_edges"].extend(
        [
            (
                vj,
                vk,
                edge_data.get("oq_id"),
                edge_data.get("iq_id"),
            )
            for vk, edge_data in candidates
        ]
    )

    edge_model = (
        server_configs[1]["model"] if len(server_configs) > 1 else server_configs[0]["model"]
    )
    chosen = llm_choose_edge(
        query_text=query_text,
        candidate_edges=candidates,
        graph=graph,
        server_configs=server_configs,
        traversal_prompt=traversal_prompt,
        token_totals=token_totals,
        reason=is_r1_like(edge_model),
        seed=seed,
    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen
    state["Evisited"].add((vj, chosen_vk, chosen_edge["oq_id"], chosen_edge["iq_id"]))
    is_repeat = chosen_vk in visited_passages

    hop_log["edges_chosen"].append(
        {
            "from": vj,
            "to": chosen_vk,
            "oq_id": chosen_edge["oq_id"],
            "iq_id": chosen_edge["iq_id"],
            "repeat_visit": is_repeat,
        }
    )

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
        return set()

    hop_log["new_passages"].append(chosen_vk)
    next_Cqueue.append(chosen_vk)
    return {chosen_vk}


def enhanced_traversal_algorithm(
    vj,
    graph: nx.DiGraph,
    query_text: str,
    visited_passages,
    server_configs,
    ccount,
    next_Cqueue,
    hop_log,
    state,
    traversal_prompt: str,
    hop: int,
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,
    **kwargs,
):
    """Traversal step that biases by conditioned_score before LLM selection."""

    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
        if (vj, vk, graph[vj][vk]["oq_id"], graph[vj][vk]["iq_id"]) not in state["Evisited"]
    ]
    if not candidates:
        return set()

    candidates.sort(key=lambda it: (it[1].get("oq_id", ""), it[0]))
    reverse = hop == 0
    candidates.sort(
        key=lambda it: graph.nodes[it[0]].get("conditioned_score", 0.0),
        reverse=reverse,
    )

    hop_log["candidate_edges"].extend(
        [
            (
                vj,
                vk,
                edge_data.get("oq_id"),
                edge_data.get("iq_id"),
            )
            for vk, edge_data in candidates
        ]
    )

    edge_model = (
        server_configs[1]["model"] if len(server_configs) > 1 else server_configs[0]["model"]
    )
    chosen = llm_choose_edge(
        query_text=query_text,
        candidate_edges=candidates,
        graph=graph,
        server_configs=server_configs,
        traversal_prompt=traversal_prompt,
        token_totals=token_totals,
        reason=is_r1_like(edge_model),
        seed=seed,
    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen
    state["Evisited"].add((vj, chosen_vk, chosen_edge["oq_id"], chosen_edge["iq_id"]))
    is_repeat = chosen_vk in visited_passages

    hop_log["edges_chosen"].append(
        {
            "from": vj,
            "to": chosen_vk,
            "oq_id": chosen_edge["oq_id"],
            "iq_id": chosen_edge["iq_id"],
            "repeat_visit": is_repeat,
        }
    )

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
        return set()

    hop_log["new_passages"].append(chosen_vk)
    next_Cqueue.append(chosen_vk)
    return {chosen_vk}


def traverse_graph(
    graph: nx.DiGraph,
    query_text: str,
    query_emb: np.ndarray,
    passage_emb: np.ndarray,
    seed_passages: list,
    n_hops: int,
    server_configs: list,
    traversal_alg: Callable,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    traversal_prompt: str = "",
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,
):
    """Traverse the graph while recording query similarity for visited passages."""

    query_keywords = set(extract_keywords(query_text))
    query_vec = np.asarray(query_emb).reshape(-1)
    if query_keywords:
        node_metadata = [data for _, data in graph.nodes(data=True)]
        idf, avgdl, doc_count, _ = compute_bm25_stats(
            query_keywords,
            node_metadata,
            keyword_field="keywords_passage",
        )
    else:
        idf = {}
        avgdl = 0.0
        doc_count = 0

    def _update_query_sim(pid: str) -> None:
        node = graph.nodes.get(pid)
        if not node:
            return
        vec_id = node.get("vec_id")
        sim_cos = 0.0
        if vec_id is not None and 0 <= vec_id < len(passage_emb):
            sim_cos = float(np.dot(query_vec, passage_emb[vec_id]))
        if query_keywords and doc_count:
            sim_bm25 = bm25_score(
                query_keywords,
                set(node.get("keywords_passage", [])),
                idf,
                avgdl,
            )
        else:
            sim_bm25 = 0.0
        sim_hybrid = alpha * sim_cos + (1 - alpha) * sim_bm25
        node["query_sim"] = round(sim_hybrid, 4)

    Cqueue = seed_passages[:]
    for pid in Cqueue:
        _update_query_sim(pid)
    visited_passages = set()
    ccount = {pid: 1 for pid in Cqueue}
    hop_trace = []
    state = {
        "Evisited": set(),
        "none_count": 0,
        "repeat_visit_count": 0,
    }

    for hop in range(n_hops):
        next_Cqueue = []
        hop_log = {
            "hop": hop,
            "expanded_from": list(Cqueue),
            "new_passages": [],
            "edges_chosen": [],
            "candidate_edges": [],
            "none_count": 0,
            "repeat_visit_count": 0,
        }

        for vj in Cqueue:
            if vj not in graph:
                continue

            visited_passages.add(vj)

            new_nodes = traversal_alg(
                vj=vj,
                graph=graph,
                query_text=query_text,
                visited_passages=visited_passages,
                server_configs=server_configs,
                ccount=ccount,
                next_Cqueue=next_Cqueue,
                hop_log=hop_log,
                state=state,
                traversal_prompt=traversal_prompt,
                hop=hop,
                token_totals=token_totals,
                seed=seed,
            )
            for new_pid in new_nodes:
                _update_query_sim(new_pid)
            visited_passages.update(new_nodes)

        hop_trace.append(hop_log)
        Cqueue = next_Cqueue

    visited_passages.update(seed_passages)

    return list(visited_passages), ccount, hop_trace, {
        "none_count": state["none_count"],
        "repeat_visit_count": state["repeat_visit_count"],
    }


##############################################################################
# Result saving
##############################################################################



def save_traversal_result(
    question_id,
    gold_passages,
    visited_passages,
    ccount,
    hop_trace,
    traversal_alg,
    helpful_passages,
    hits_at_k,
    recall_at_k,
    *,
    dataset: str,
    split: str,
    variant: str,
    retriever_name: str,
    traverser_model: str,
    reader_model: str | None = None,
    traversal_wall_time_sec: float | None = None,
    output_path="dev_results.jsonl",
    token_usage: Optional[Dict[str, int]] = None,
    seed: int | None = None,
):
    """Save a complete traversal + metric result for a single query."""

    hop_trace_with_metrics, final_metrics = compute_hop_metrics(hop_trace, gold_passages)
    gold_counts, gold_attention_ratio = compute_gold_attention(ccount, gold_passages)

    result_entry = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "retriever_name": retriever_name,
        "traverser_model": traverser_model,
        "reader_model": reader_model,
        "question_id": question_id,
        "gold_passages": gold_passages,
        "visited_passages": list(visited_passages),
        "visit_counts": dict(ccount),
        "gold_visit_counts": gold_counts,
        "gold_attention_ratio": round(gold_attention_ratio, 4),
        "hop_trace": hop_trace_with_metrics,
        "final_metrics": final_metrics,
        "traversal_algorithm": traversal_alg.__name__,
        "helpful_passages": [
            {"passage_id": pid, "score": round(score, 4)}
            for pid, score in helpful_passages
        ],
        "hits_at_k": hits_at_k,
        "recall_at_k": recall_at_k,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if seed is not None and "seed" not in result_entry:
        result_entry["seed"] = seed

    if token_usage is not None:
        result_entry["trav_prompt_tokens"] = token_usage.get("trav_prompt_tokens", 0)
        result_entry["trav_output_tokens"] = token_usage.get("trav_output_tokens", 0)
        result_entry["trav_tokens_total"] = token_usage.get(
            "trav_tokens_total",
            token_usage.get("trav_prompt_tokens", 0)
            + token_usage.get("trav_output_tokens", 0),
        )
        result_entry["n_traversal_calls"] = token_usage.get("n_traversal_calls", 0)
        result_entry["query_latency_ms"] = token_usage.get(
            "query_latency_ms", token_usage.get("t_traversal_ms", 0)
        )
        result_entry["call_latency_ms"] = token_usage.get("call_latency_ms", 0)

    if traversal_wall_time_sec is not None:
        result_entry["traversal_wall_time_sec"] = round(traversal_wall_time_sec, 4)

    append_jsonl(str(output_path), result_entry)
