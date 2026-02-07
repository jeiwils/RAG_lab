"""Answer generation utilities for passage-based QA."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore

from src.d1_evaluation.answer_metrics import normalise_answer
from src.utils.x_config import MAX_TOKENS, TEMPERATURE
from src.utils.z_llm_utils import is_r1_like, query_llm, strip_think

__all__ = [
    "ask_llm_with_passages",
    "extract_final_answer",
    "post_process_answer",
    "BASE_PROMPT_TEMPLATE",
    "compute_max_context_tokens",
]

### DEFAULTS
DEFAULT_MAX_CONTEXT_TOKENS = int(os.environ.get("LLM_CONTEXT_WINDOW", "4096"))
DEFAULT_CONTEXT_SAFETY_MARGIN = int(os.environ.get("LLM_CONTEXT_SAFETY_MARGIN", "128"))
DEFAULT_MAX_PASSAGE_TOKENS = int(os.environ.get("MAX_PASSAGE_TOKENS", "512"))
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


@lru_cache(maxsize=4)
def _get_encoding(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str, model_name: str) -> int:
    if not text:
        return 0
    enc = _get_encoding(model_name)
    if enc is not None:
        return len(enc.encode(text))
    # Rough fallback: inflate word count to reduce risk of exceeding context.
    return int(len(text.split()) * 1.3) + 1


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
    base_tokens = _estimate_tokens(base_prompt, model_name)
    reserved_for_output = min(max_output_tokens, 256)
    return (
        context_budget_tokens
        + base_tokens
        + reserved_for_output
        + DEFAULT_CONTEXT_SAFETY_MARGIN
    )


def _truncate_to_token_budget(text: str, budget: int, model_name: str) -> str:
    if budget <= 0 or not text:
        return ""
    enc = _get_encoding(model_name)
    if enc is not None:
        tokens = enc.encode(text)
        if len(tokens) <= budget:
            return text
        return enc.decode(tokens[:budget]).strip()
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[:budget]).strip()


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
) -> Dict[str, str]:
    """Generate an answer from top passages using an LLM server."""
    passage_texts = []

    for i, pid in enumerate(passage_ids[:top_k_answer_passages], start=1):
        if graph:
            passage = graph.nodes[pid].get("text", "")
        elif passage_lookup:
            passage = passage_lookup.get(pid, "")
        else:
            passage = "[missing passage]"

        passage_texts.append(passage)

    base_prompt = BASE_PROMPT_TEMPLATE.format(
        context="{context}",
        query_text=query_text,
    )

    context_window = max_context_tokens or DEFAULT_MAX_CONTEXT_TOKENS
    passage_token_cap = (
        DEFAULT_MAX_PASSAGE_TOKENS if max_passage_tokens is None else max_passage_tokens
    )
    reserved_for_output = min(max_tokens, 256)
    prompt_budget = max(
        0, context_window - reserved_for_output - DEFAULT_CONTEXT_SAFETY_MARGIN
    )
    base_tokens = _estimate_tokens(base_prompt.format(context=""), model_name)
    context_budget = max(prompt_budget - base_tokens, 0)

    selected_passages: List[str] = []
    used_tokens = 0
    for passage in passage_texts:
        text = passage.strip()
        if not text:
            continue
        if passage_token_cap is not None and passage_token_cap > 0:
            text = _truncate_to_token_budget(text, passage_token_cap, model_name)
        est_tokens = _estimate_tokens(text, model_name)
        sep_tokens = 2
        if used_tokens + est_tokens + sep_tokens > context_budget:
            remaining = context_budget - used_tokens - sep_tokens
            if not selected_passages and remaining > 0:
                trimmed = _truncate_to_token_budget(text, remaining, model_name)
                if trimmed:
                    selected_passages.append(trimmed)
            break
        selected_passages.append(text)
        used_tokens += est_tokens + sep_tokens

    prompt = base_prompt.format(context="\n\n".join(selected_passages))

    raw = ""
    raw_clean = ""
    usage: Dict[str, int] = {}
    max_attempts = 2

    stop_sequences: list[str] | None = ["\n", ".", "Answer:", "Final answer:"]
    if is_r1_like(model_name):
        stop_sequences = None

    for _ in range(max_attempts):
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
    else:
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
