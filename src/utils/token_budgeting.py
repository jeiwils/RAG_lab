"""Shared token budgeting helpers used by selection and reader packing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

try:  # optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore

DEFAULT_CONTEXT_WINDOW_TOKENS = int(os.environ.get("LLM_CONTEXT_WINDOW", "4096"))
DEFAULT_CONTEXT_SAFETY_MARGIN = int(os.environ.get("LLM_CONTEXT_SAFETY_MARGIN", "128"))
DEFAULT_MAX_PASSAGE_TOKENS = int(os.environ.get("MAX_PASSAGE_TOKENS", "512"))


@lru_cache(maxsize=4)
def get_encoding(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model_name: str) -> int:
    if not text:
        return 0
    enc = get_encoding(model_name)
    if enc is not None:
        return len(enc.encode(text))
    # Rough fallback: inflate word count to reduce risk of context overflow.
    return int(len(text.split()) * 1.3) + 1


def truncate_to_token_budget(text: str, budget: int, model_name: str) -> str:
    if budget <= 0 or not text:
        return ""
    enc = get_encoding(model_name)
    if enc is not None:
        tokens = enc.encode(text)
        if len(tokens) <= budget:
            return text
        return enc.decode(tokens[:budget]).strip()
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[:budget]).strip()


@dataclass(frozen=True)
class ReaderBudgetPolicy:
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN
    max_passage_tokens: int | None = DEFAULT_MAX_PASSAGE_TOKENS
    max_output_reserve_tokens: int = 256
    separator_tokens: int = 2


def compute_reader_budget(
    *,
    policy: ReaderBudgetPolicy,
    base_prompt_tokens: int,
    max_output_tokens: int,
) -> Dict[str, int]:
    reserved_for_output = min(max_output_tokens, policy.max_output_reserve_tokens)
    prompt_budget = max(
        0,
        policy.context_window_tokens - reserved_for_output - policy.safety_margin_tokens,
    )
    context_budget = max(prompt_budget - base_prompt_tokens, 0)
    return {
        "context_window_tokens": policy.context_window_tokens,
        "reserved_for_output_tokens": reserved_for_output,
        "safety_margin_tokens": policy.safety_margin_tokens,
        "prompt_budget_tokens": prompt_budget,
        "base_prompt_tokens": base_prompt_tokens,
        "context_budget_tokens": context_budget,
    }


def pack_passages_to_budget(
    passages: Sequence[str],
    *,
    model_name: str,
    context_budget_tokens: int,
    max_passage_tokens: int | None,
    separator_tokens: int = 2,
) -> Tuple[List[str], Dict[str, int]]:
    selected_passages: List[str] = []
    used_tokens = 0
    n_truncated = 0

    for passage in passages:
        original = passage.strip()
        if not original:
            continue

        text = original
        was_truncated = False

        if max_passage_tokens is not None and max_passage_tokens > 0:
            capped = truncate_to_token_budget(text, max_passage_tokens, model_name)
            if capped != text:
                was_truncated = True
            text = capped

        est_tokens = estimate_tokens(text, model_name)
        if used_tokens + est_tokens + separator_tokens > context_budget_tokens:
            remaining = context_budget_tokens - used_tokens - separator_tokens
            if not selected_passages and remaining > 0:
                trimmed = truncate_to_token_budget(text, remaining, model_name)
                if trimmed:
                    if trimmed != text:
                        was_truncated = True
                    selected_passages.append(trimmed)
                    used_tokens += estimate_tokens(trimmed, model_name) + separator_tokens
                    if was_truncated:
                        n_truncated += 1
            break

        selected_passages.append(text)
        used_tokens += est_tokens + separator_tokens
        if was_truncated:
            n_truncated += 1

    n_candidates = len([p for p in passages if p and p.strip()])
    n_selected = len(selected_passages)
    return selected_passages, {
        "n_candidate_passages": n_candidates,
        "n_selected_passages": n_selected,
        "n_dropped_passages": max(0, n_candidates - n_selected),
        "n_truncated_passages": n_truncated,
        "used_context_tokens": used_tokens,
    }
