"""

A lightweight, rule-based discourse-aware extension to extractive evidence selection, designed to preserve cohesion and entailment across sentence boundaries.


Benchmarking
- Baselines = EXIT, dense rag, sparse rag, hybrid rag 
- 1 = EXIT + discourse aware sentence expansion 
- 2 = EXIT + discourse aware chunking
- 2 = EXIT + discourse aware sentence expansion and chunking 




1. Retrieve top-k chunks (fixed k, e.g. 20)
2. Split chunks into sentences
3. EXIT extractor scores sentences
4. EXIT adaptively selects sentences
   (ranked + halting / token budget)
5. ↓ YOUR ADDITION ↓
6. Discourse-aware span expansion
   - pronoun dependency → include antecedent sentence
   - sentence-initial connectives → include previous sentence
   - colon / list introducer → include following sentence
7. Repack expanded spans under the same token budget
8. Reader generates answer







Offline:
Gold datasets
 → build sentence-level labels
 → LoRA fine-tuning (binary extractor)

Inference:
Query
 → retrieve chunks (passages)
 → split chunks into sentences
 → score sentences with trained LoRA extractor 
 → expand sentences with discourse-aware rules
 → select top relevant sentences with trained LoRa extractor under token budget
 → send extracted sentences to reader
 → reader generates answer




"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.a3_representations import (
    dataset_rep_paths,
    get_embedding_model,
    load_faiss_index,
)
from src.a3_representations.sparse_representations import SPACY_MODEL
from src.a2_indexing.retrieval_indexing import _retrieve_chunk_indices
from src.b2_reranking.useful_sentence_extractor_lora import (
    DEFAULT_MAX_LEN,
    encode_batch,
    select_ranked_sentences,
)
from src.b1_retrieval import DEFAULT_HYBRID_ALPHA
from src.c1_reasoning.sentence_reasoning import (
    _build_sentence_candidates,
    _select_top_sentences,
    compute_passage_hits_recall_at_k,
)
from src.c2_generation.answer_gen import ask_llm_with_passages
from src.d1_evaluation.answer_metrics import aggregate_answer_scores, evaluate_answers
from src.d1_evaluation.timing_metrics import compute_throughput_stats, wall_time
from src.utils.__utils__ import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    load_jsonl,
    processed_dataset_paths,
    save_jsonl,
)

__all__ = ["run_da_exit", "main"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Sentence usefulness cutoff (set from best_metrics.json of chosen LoRA model)
TAU_LOW = 0.58
# Total token budget for selected sentences (adjust for reader context size).

DATASETS = ["musique", "hotpotqa", "2wikimultihopqa", "natural_questions"]
SPLITS = ["train"]
RETRIEVER_CONFIG = {
    "hybrid": True,
    "dense": False,
    "sparse": False,
}

# Chunking baseline.
CHUNKING_MODE = "standard_chunks"  # "standard_chunks" | "discourse_aware_chunks"
# Sentence mode controls post-ranking expansion for the reader context.
# "standard_sentences" = no expansion; "discourse_aware_sentences" = expand after ranking.
SENTENCE_MODE = "discourse_aware_sentences"  # "standard_sentences" | "discourse_aware_sentences"


SENTENCE_TOKEN_BUDGET = 512
TOP_K_CHUNK_SWEEP = [1, 5, 10, 20]
HYBRID_ALPHA = DEFAULT_HYBRID_ALPHA ## ????

IN_PROCESS_READER_LOCAL_FILES_ONLY = True
SENTENCE_GRADER_MODEL = (
    "data/models/useful_sentence_lora/grid/"
    "run4_hn6_rn1_pw2.0_lr3e-05_ep8_r8_a16_d0.05_bs16/checkpoint-epoch8"
)
SENTENCE_LORA_BATCH_SIZE = 32
SENTENCE_LORA_MAX_LEN = DEFAULT_MAX_LEN
SENTENCE_LORA_LOCAL_FILES_ONLY = True

SEEDS = [1] #[1, 2, 3]
RESUME = True

# Debug logging of ranked candidates (can be large).
LOG_RANKED_CANDIDATES = True
RANKED_CANDIDATES_LIMIT: int | None = 200

USE_IN_PROCESS_READER = False
SERVER_URL = "http://localhost:8005"
READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_chunk_representations(
    rep_paths: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Any]:
    """Load precomputed chunk metadata and FAISS index from dataset representations."""
    jsonl_path = Path(rep_paths["passages_jsonl"])
    index_path = Path(rep_paths["passages_index"])
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing representations JSONL: {jsonl_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    metadata = list(load_jsonl(str(jsonl_path)))
    index = load_faiss_index(str(index_path))
    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size mismatch: "
            f"index has {index.ntotal} vectors but metadata lists {len(metadata)} chunks"
        )
    return metadata, index


def _is_lora_checkpoint(model_name: str) -> bool:
    """Return True if ``model_name`` points to a LoRA checkpoint directory."""
    path = Path(model_name)
    return path.is_dir() and (path / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def _load_sentence_lora(checkpoint_dir: str):
    """Load a LoRA sentence classifier and its tokenizer once per process."""
    ckpt = Path(checkpoint_dir)
    config_path = ckpt / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {config_path}")
    with open(config_path, "rt", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"Missing base_model_name_or_path in {config_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt), local_files_only=SENTENCE_LORA_LOCAL_FILES_ONLY
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        use_safetensors=None,
        local_files_only=SENTENCE_LORA_LOCAL_FILES_ONLY,
    )
    model = PeftModel.from_pretrained(base, str(ckpt))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _score_sentences_lora(
    question: str,
    sentences: List[Dict[str, Any]],
    *,
    checkpoint_dir: Path,
    batch_size: int,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Score sentence candidates with the LoRA classifier and return token stats."""
    totals = {
        "sentence_prompt_tokens": 0,
        "sentence_output_tokens": 0,
        "sentence_total_tokens": 0,
        "n_sentence_calls": 0,
    }
    if not sentences:
        return [], totals

    model, tokenizer, device = _load_sentence_lora(str(checkpoint_dir))

    scored: List[Dict[str, Any]] = []
    for start in range(0, len(sentences), batch_size):
        batch_items = sentences[start : start + batch_size]
        batch = [
            {
                "query": question,
                "context": str(item.get("context", item.get("text", ""))),
                "sentence": str(item.get("text", "")),
                "label": 0,
            }
            for item in batch_items
        ]
        tokens, _ = encode_batch(
            tokenizer, batch, max_length=max_length
        )
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            totals["sentence_prompt_tokens"] += int(attention_mask.sum().item())
        else:
            totals["sentence_prompt_tokens"] += int(tokens["input_ids"].numel())

        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = model(**tokens).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        prob_list = probs.detach().cpu().tolist()

        for item, prob in zip(batch_items, prob_list):
            score = float(prob)
            scored.append(
                {
                    **item,
                    "score": score,
                    "score_normalized": score,
                    "invalid": False,
                }
            )

        totals["n_sentence_calls"] += len(batch_items)

    totals["sentence_total_tokens"] = totals["sentence_prompt_tokens"]
    return scored, totals


def run_da_exit(
    dataset: str,
    split: str,
    retriever: str,
    *,
    reader_model: str,
    sentence_model: str,
    server_url: str | None,
    top_k_chunks: int,
    chunking_mode: str,
    seed: int | None,
    resume: bool,
) -> Dict[str, Any]:
    """Run DA_EXIT for a dataset/split/retriever and return summary metrics."""
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

    if not _is_lora_checkpoint(sentence_model):
        raise ValueError(
            "SENTENCE_GRADER_MODEL must be a LoRA checkpoint path with adapter_config.json. "
            f"Got: {sentence_model}"
        )
    if not USE_IN_PROCESS_READER and server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    processed_paths = processed_dataset_paths(dataset, split)
    if chunking_mode == "standard_chunks":
        passage_source = "full_passages_chunks"
    elif chunking_mode == "discourse_aware_chunks":
        passage_source = "full_passages_chunks_discourse_aware"
    else:
        raise ValueError(
            "chunking_mode must be 'standard_chunks' or 'discourse_aware_chunks'"
        )

    rep_paths = dataset_rep_paths(dataset, split, passage_source=passage_source)
    metadata, index = _load_chunk_representations(rep_paths)
    encoder = get_embedding_model() if retriever in {"dense", "hybrid"} else None

    questions_path = processed_paths["questions"]
    questions = list(load_jsonl(str(questions_path)))

    variant = f"da_exit_{chunking_mode}_{SENTENCE_MODE}_{retriever}_k{top_k_chunks}"
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
    gold: Dict[str, List[str]] = {}
    wall_times: List[float] = []

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

    for q in tqdm(questions, desc=f"{dataset}/{split}/{retriever}"):
        q_id = q["question_id"]
        if resume and q_id in done_ids:
            continue
        q_text = q.get("question", "")
        gold[q_id] = [q.get("gold_answer", "")]

        def _run_query():
            """Execute retrieval, sentence scoring, and reader inference for one query."""
            query_vec = None
            if retriever in {"dense", "hybrid"} and encoder is not None:
                query_vec = encoder.encode([q_text], normalize_embeddings=False)
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
            expand_after = SENTENCE_MODE == "discourse_aware_sentences"
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
                ranked_sentences: Optional[List[Dict[str, Any]]] = None,
                ranked_truncated: Optional[bool] = None,
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
                    "chunking_mode": chunking_mode,
                    "sentence_mode": SENTENCE_MODE,
                    "expand_after_rerank": expand_after,
                    "reader_model": reader_model,
                    "question_id": q_id,
                    "question": q_text,
                    "tau_low": TAU_LOW,
                    "token_budget": SENTENCE_TOKEN_BUDGET,
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
                if LOG_RANKED_CANDIDATES and ranked_sentences is not None:
                    payload["ranked_sentences"] = [
                        _pack_sentence(s) for s in ranked_sentences
                    ]
                    payload["ranked_truncated"] = bool(ranked_truncated)
                    payload["ranked_limit"] = RANKED_CANDIDATES_LIMIT
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
                }

            """Scoring: LoRA checkpoint sentence grader."""
            scored, sentence_tokens = _score_sentences_lora(
                q_text,
                sentences,
                checkpoint_dir=Path(sentence_model),
                batch_size=SENTENCE_LORA_BATCH_SIZE,
                max_length=SENTENCE_LORA_MAX_LEN,
            )

            """Ranking: _select_top_sentences(scored, top_k=len(scored)) sorts by score (highest first)."""
            ranked = _select_top_sentences(scored, top_k=len(scored))

            ranked_for_debug: Optional[List[Dict[str, Any]]] = None
            ranked_truncated = None
            if LOG_RANKED_CANDIDATES:
                if RANKED_CANDIDATES_LIMIT is None:
                    ranked_for_debug = ranked
                    ranked_truncated = False
                else:
                    ranked_for_debug = ranked[:RANKED_CANDIDATES_LIMIT]
                    ranked_truncated = len(ranked) > RANKED_CANDIDATES_LIMIT

            """Repackaging with tau + budget: select_ranked_sentences(..., tau_low=TAU_LOW, token_budget=SENTENCE_TOKEN_BUDGET) keeps only above tau and within the budget, preserving rank order."""
            filtered = select_ranked_sentences(
                ranked,
                token_budget=SENTENCE_TOKEN_BUDGET,
                tau_low=TAU_LOW,
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
                    token_budget=SENTENCE_TOKEN_BUDGET,
                    tau_low=None,
                )
                selected = expanded_selected
            if not selected:
                _write_reader_debug(
                    [],
                    reason="no_selected",
                    n_candidates=len(sentences),
                    n_ranked=len(ranked),
                    n_filtered=len(filtered),
                    ranked_sentences=ranked_for_debug,
                    ranked_truncated=ranked_truncated,
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
                }

            sentence_lookup = {s["sentence_id"]: s["text"] for s in selected}
            sentence_ids = list(sentence_lookup.keys())

            """Sent to reader: the ordered selected list is passed through ask_llm_with_passages in that same order."""
            _write_reader_debug(
                selected,
                n_candidates=len(sentences),
                n_ranked=len(ranked),
                n_filtered=len(filtered),
                ranked_sentences=ranked_for_debug,
                ranked_truncated=ranked_truncated,
            )
            answer = ask_llm_with_passages(
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

            return {
                "selected_sentences": selected,
                "answer": answer,
                "sentence_tokens": sentence_tokens,
            }

        result, query_wall_sec = wall_time(_run_query)
        wall_times.append(query_wall_sec)

        selected_sentences = result["selected_sentences"]
        answer = result["answer"]
        sentence_tokens = result["sentence_tokens"]

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

        append_jsonl(
            str(paths["answers"]),
            {
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "retriever": retriever,
                "chunking_mode": chunking_mode,
                "reader_model": reader_model,
                "sentence_model": sentence_model,
                "question_id": q_id,
                "question": q_text,
                "raw_answer": answer.get("raw_answer", ""),
                "normalised_answer": answer.get("normalised_answer", ""),
                "used_sentence_ids": [s["sentence_id"] for s in selected_sentences],
                "selected_sentences": [
                    {
                        "sentence_id": s["sentence_id"],
                        "chunk_id": s["chunk_id"],
                        "sent_idx": s["sent_idx"],
                        "span_start": s.get("span_start", s["sent_idx"]),
                        "span_end": s.get("span_end", s["sent_idx"]),
                        "expanded": s.get("expanded", False),
                        "score": s.get("score", 0),
                        "score_normalized": s.get("score_normalized", 0.0),
                        "text": s["text"],
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
                "wall_time_sec": round(query_wall_sec, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        predictions[q_id] = answer.get("normalised_answer", "")

    if not predictions:
        print("No new queries processed.")
        return {}

    per_query = evaluate_answers(predictions, gold)
    agg_scores = aggregate_answer_scores(predictions, gold)

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
    if resume and paths["answer_metrics"].exists():
        for rec in metric_records:
            append_jsonl(str(paths["answer_metrics"]), rec)
    else:
        save_jsonl(str(paths["answer_metrics"]), metric_records)

    wall_time_sec_total = sum(wall_times)
    wall_time_sec_mean = (
        wall_time_sec_total / len(wall_times) if wall_times else 0.0
    )
    wall_time_sec_median = median(wall_times) if wall_times else 0.0

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
            "reader_model": reader_model,
            "sentence_model": sentence_model,
            "chunking_mode": chunking_mode,
            "sentence_mode": SENTENCE_MODE,
            "queries_total": queries_total,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "accuracy": {
            "mean_em": agg_scores["mean_em"],
            "mean_f1": agg_scores["mean_f1"],
        },
        "latency": {
            "wall_time_sec_total": round(wall_time_sec_total, 4),
            "wall_time_sec_mean": round(wall_time_sec_mean, 4),
            "wall_time_sec_median": round(wall_time_sec_median, 4),
        },
        "cost": {
            "tokens_total": tokens_total,
            "tokens_per_query_mean": round(tokens_per_query_mean, 4),
            "reader_prompt_tokens_total": token_totals["reader_prompt_tokens"],
            "reader_output_tokens_total": token_totals["reader_output_tokens"],
            "reader_tokens_total": token_totals["reader_total_tokens"],
            "reader_calls_total": token_totals["n_reader_calls"],
            "sentence_extractor_prompt_tokens_total": token_totals[
                "sentence_prompt_tokens"
            ],
            "sentence_extractor_output_tokens_total": token_totals[
                "sentence_output_tokens"
            ],
            "sentence_extractor_tokens_total": token_totals["sentence_total_tokens"],
            "sentence_extractor_calls_total": token_totals["n_sentence_calls"],
            "reader_prompt_tokens_per_query_mean": round(
                token_totals["reader_prompt_tokens"] / queries_total, 4
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
    if seed is not None:
        summary["meta"]["seed"] = seed
    if hits_at_k_scores:
        summary["retrieval"] = {
            "mean_hits_at_k_ratio": round(
                sum(hits_at_k_scores) / len(hits_at_k_scores), 4
            ),
            "mean_recall_at_k_ratio": round(
                sum(recall_at_k_scores) / len(recall_at_k_scores), 4
            ),
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
    """Entry point for the DA_EXIT orchestrator.

    Expects processed datasets under data/processed_datasets and precomputed
    chunk representations under data/representations/datasets.
    """
    if RETRIEVER_CONFIG.get("hybrid") or RETRIEVER_CONFIG.get("sparse"):
        print(f"[spaCy] Using: {SPACY_MODEL}")

    for dataset in DATASETS:
        for split in SPLITS:
            for retriever, enabled in RETRIEVER_CONFIG.items():
                if not enabled:
                    continue
                for top_k_chunks in TOP_K_CHUNK_SWEEP:
                    for seed in SEEDS:
                        print(
                            f"\n[DA_EXIT] dataset={dataset} split={split} retriever={retriever} "
                            f"chunking={CHUNKING_MODE} sentence_mode={SENTENCE_MODE} "
                            f"top_k_chunks={top_k_chunks} seed={seed}"
                        )
                        run_da_exit(
                            dataset=dataset,
                            split=split,
                            retriever=retriever,
                            reader_model=READER_MODEL,
                            sentence_model=SENTENCE_GRADER_MODEL,
                            server_url=SERVER_URL,
                            top_k_chunks=top_k_chunks,
                            chunking_mode=CHUNKING_MODE,
                            seed=seed,
                            resume=RESUME,
                        )


if __name__ == "__main__":
    main()
