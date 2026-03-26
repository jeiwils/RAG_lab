"""Answer generation utilities for passage-based QA."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.d1_evaluation.answer_metrics import normalise_answer
from src.utils.token_budgeting import (
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MAX_PASSAGE_TOKENS,
    ReaderBudgetPolicy,
    compute_reader_budget,
    estimate_tokens,
    pack_passages_to_budget,
)
from src.utils.x_config import MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)
from src.utils.z_llm_utils import is_r1_like, query_llm, strip_think

__all__ = [
    "ask_llm_with_passages",
    "extract_final_answer",
    "post_process_answer",
    "BASE_PROMPT_TEMPLATE",
    "compute_max_context_tokens",
]

### DEFAULTS
DEFAULT_MAX_CONTEXT_TOKENS = DEFAULT_CONTEXT_WINDOW_TOKENS
BASE_PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "-----------------------\n"
    "{context}\n"
    "-----------------------\n"
    "Given the context information and not prior knowledge, answer the query. "
    "Do not provide any explanation.\n"
    "Query: {query_text}\n"
    "Answer:"
)

def compute_max_context_tokens(
    query_text: str,
    model_name: str,
    context_budget_tokens: int,
    max_output_tokens: int,
) -> int:
    """Compute a max_context_tokens that yields the desired passage budget."""
    if context_budget_tokens <= 0:
        return 0
    base_prompt = BASE_PROMPT_TEMPLATE.format(context="", query_text=query_text)
    base_tokens = estimate_tokens(base_prompt, model_name)
    reserved_for_output = min(max_output_tokens, 256)
    return (
        context_budget_tokens
        + base_tokens
        + reserved_for_output
        + DEFAULT_CONTEXT_SAFETY_MARGIN
    )

def extract_final_answer(raw: str) -> str:
    """Extract the last stated answer from raw LLM output."""
    pattern = re.compile(
        r"(?:final answer|answer(?: is)?)\s*:?\s*(.+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = pattern.findall(raw)
    if matches:
        return matches[-1].strip()
    return raw.strip()


def clean_tokens(tokens: list[str]) -> list[str]:
    """Remove duplicate tokens while keeping order."""
    drop_unknown = any(t != "unknown" for t in tokens)
    seen: set[str] = set()
    cleaned: list[str] = []
    for t in tokens:
        if t == "unknown" and drop_unknown:
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


def first_fragment(text: str) -> str:
    """Return the first non-empty fragment split by period or newline."""
    for part in re.split(r"[.\n]", text):
        frag = part.strip()
        if frag:
            return frag
    return ""


def post_process_answer(
    text: str,
    max_words: int = 40,
    repeat_threshold: int = 3,
) -> Optional[str]:
    """Validate and possibly reject an LLM answer."""
    tokens = text.split()
    if len(tokens) > max_words:
        return None

    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > repeat_threshold:
            return None

    return text

def ask_llm_with_passages(
    query_text: str,
    passage_ids: List[str],
    graph: Optional[nx.DiGraph],
    server_url: str | None,
    max_tokens: int = MAX_TOKENS["answer_generation"],
    passage_lookup: Optional[Dict[str, str]] = None,
    model_name: str = "",
    top_k_answer_passages: int = 20,
    seed: int | None = None,
    max_context_tokens: int | None = None,
    max_passage_tokens: int | None = None,
    *,
    in_process: bool = False,
    local_files_only: bool = True,
) -> Dict[str, Any]:
    """Generate an answer from top passages using an LLM server."""
    if not query_text.strip() or not passage_ids:
        logger.warning("Empty query or passage_ids; returning empty answer")
        return {
            "raw_answer": "",
            "raw_clean": "",
            "normalised_answer": "",
            "prompt_len": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    passage_texts = []
    for i, pid in enumerate(passage_ids[:top_k_answer_passages], start=1):
        passage = None
        if graph:
            passage = graph.nodes[pid].get("text", "")
        elif passage_lookup:
            passage = passage_lookup.get(pid, "")
        
        if not passage or not passage.strip():
            logger.warning(f"Skipping missing or empty passage {pid}")
            continue
        passage_texts.append(passage)

    base_prompt = BASE_PROMPT_TEMPLATE.format(
        context="{context}",
        query_text=query_text,
    )

    policy = ReaderBudgetPolicy(
        context_window_tokens=max_context_tokens or DEFAULT_MAX_CONTEXT_TOKENS,
        safety_margin_tokens=DEFAULT_CONTEXT_SAFETY_MARGIN,
        max_passage_tokens=(
        DEFAULT_MAX_PASSAGE_TOKENS if max_passage_tokens is None else max_passage_tokens
        ),
    )
    budget_info = compute_reader_budget(
        policy=policy,
        base_prompt_tokens=estimate_tokens(base_prompt.format(context=""), model_name),
        max_output_tokens=max_tokens,
    )

    selected_passages, packing_stats = pack_passages_to_budget(
        passage_texts,
        model_name=model_name,
        context_budget_tokens=budget_info["context_budget_tokens"],
        max_passage_tokens=policy.max_passage_tokens,
        separator_tokens=policy.separator_tokens,
    )

    prompt = base_prompt.format(context="\n\n".join(selected_passages))

    raw = ""
    raw_clean = ""
    usage: Dict[str, int] = {}
    max_attempts = 2

    stop_sequences: list[str] | None = ["\n", ".", "Answer:", "Final answer:"]
    if is_r1_like(model_name):
        stop_sequences = None

    for attempt in range(max_attempts):
        try:
            if in_process:
                raw, usage = _query_llm_in_process(
                    prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE.get("answer_generation", 0.0),
                    stop=stop_sequences,
                    seed=seed,
                    local_files_only=local_files_only,
                )
            else:
                if not server_url:
                    raise ValueError("server_url must be set when in_process=False")
                raw, usage = query_llm(
                    prompt,
                    server_url=server_url,
                    max_tokens=max_tokens,
                    model_name=model_name,
                    stop=stop_sequences,
                    temperature=TEMPERATURE.get("answer_generation", 0.0),
                    phase="answer_generation",
                    seed=seed,
                )

            if is_r1_like(model_name):
                raw = strip_think(raw)

            raw_clean = first_fragment(extract_final_answer(raw))
            if post_process_answer(raw_clean) is not None:
                break
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                break
    else:
        logger.warning("All attempts failed; returning 'unknown'")
        raw = raw_clean = "unknown"

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    tokens = clean_tokens(normalise_answer(raw_clean).split())
    norm = " ".join(tokens)

    return {
        "raw_answer": raw,
        "raw_clean": raw_clean,
        "normalised_answer": norm,
        "prompt_len": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "budget_context_window_tokens": budget_info["context_window_tokens"],
        "budget_reserved_output_tokens": budget_info["reserved_for_output_tokens"],
        "budget_safety_margin_tokens": budget_info["safety_margin_tokens"],
        "budget_prompt_tokens": budget_info["prompt_budget_tokens"],
        "budget_base_prompt_tokens": budget_info["base_prompt_tokens"],
        "budget_context_tokens": budget_info["context_budget_tokens"],
        "budget_max_passage_tokens": (
            -1 if policy.max_passage_tokens is None else int(policy.max_passage_tokens)
        ),
        "budget_n_candidate_passages": packing_stats["n_candidate_passages"],
        "budget_n_selected_passages": packing_stats["n_selected_passages"],
        "budget_n_dropped_passages": packing_stats["n_dropped_passages"],
        "budget_n_truncated_passages": packing_stats["n_truncated_passages"],
        "budget_used_context_tokens": packing_stats["used_context_tokens"],
    }



@lru_cache(maxsize=1)
def _load_in_process_reader(model_name: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _query_llm_in_process(
    prompt: str,
    *,
    model_name: str,
    max_tokens: int,
    temperature: float | None,
    stop: list[str] | None,
    seed: int | None,
    local_files_only: bool,
) -> Tuple[str, Dict[str, int]]:
    model, tokenizer, device = _load_in_process_reader(
        model_name, local_files_only=local_files_only
    )
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    generated_ids = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if stop:
        cut = None
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        if cut is not None:
            text = text[:cut]

    prompt_tokens = int(input_ids.numel())
    completion_tokens = len(
        tokenizer.encode(text, add_special_tokens=False)
    )
    total_tokens = prompt_tokens + completion_tokens

    return text, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
