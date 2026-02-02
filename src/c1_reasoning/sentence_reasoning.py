"""Sentence-level reasoning and evaluation helpers for DA_EXIT."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.utils.algorithms.discourse_aware import (
    _expand_sentence_text,
    _has_anaphora_marker,
    _has_cataphora_marker,
    iter_sentence_spans,
)
from src.utils.x_config import LLM_DEFAULTS, MAX_TOKENS, TEMPERATURE
from src.utils.z_llm_utils import is_r1_like, query_llm, strip_think

__all__ = [
    "DEFAULT_SENTENCE_EVAL_PROMPT",
    "build_sentence_evaluation_prompt",
    "evaluate_sentence_usefulness",
    "split_chunk_sentences",
    "_score_sentences",
    "_build_sentence_candidates",
    "_select_top_sentences",
    "_filter_sentences_by_threshold",
]


DEFAULT_SENTENCE_EVAL_PROMPT = (
    "You are grading how useful a single sentence is for answering a question.\n"
    "Use the integer scale {MIN_SCORE} to {MAX_SCORE}.\n"
    "{MIN_SCORE} = unrelated or incorrect.\n"
    "{MAX_SCORE} = fully answers the question.\n"
    "Question: {QUESTION}\n"
    "Sentence: {SENTENCE}\n"
    "Return only the integer."
)


def build_sentence_evaluation_prompt(
    question: str,
    sentence: str,
    *,
    scale: Tuple[int, int] = (0, 5),
    prompt_template: str = DEFAULT_SENTENCE_EVAL_PROMPT,
) -> str:
    """Build the prompt used for sentence usefulness grading."""
    min_score, max_score = _validate_scale(scale)
    return prompt_template.format(
        QUESTION=question.strip(),
        SENTENCE=sentence.strip(),
        MIN_SCORE=min_score,
        MAX_SCORE=max_score,
    )


def split_chunk_sentences(text: str) -> List[str]:
    """Split a chunk into individual sentences using discourse-aware heuristics."""
    sentences: List[str] = []
    for _, _, sentence in iter_sentence_spans(text):
        cleaned = sentence.strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _validate_scale(scale: Tuple[int, int]) -> Tuple[int, int]:
    min_score, max_score = scale
    if min_score > max_score:
        raise ValueError("scale must be (min_score, max_score)")
    return min_score, max_score


def _score_grammar(scale: Tuple[int, int]) -> str:
    min_score, max_score = _validate_scale(scale)
    choices = " | ".join(f'"{i}"' for i in range(min_score, max_score + 1))
    return f"root ::= {choices}\n"


def _extract_score(text: str, scale: Tuple[int, int]) -> Tuple[int | None, bool]:
    """Return (score, invalid_flag)."""
    min_score, max_score = _validate_scale(scale)
    cleaned = text.strip()
    if cleaned.isdigit():
        val = int(cleaned)
        if min_score <= val <= max_score:
            return val, False
    match = re.search(r"-?\d+", cleaned)
    if match:
        val = int(match.group())
        val = min(max(val, min_score), max_score)
        return val, True
    return None, True


def _score_to_unit(score: int, scale: Tuple[int, int]) -> float:
    min_score, max_score = _validate_scale(scale)
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def evaluate_sentence_usefulness(
    question: str,
    sentence: str,
    server_url: str,
    *,
    model_name: str = "",
    scale: Tuple[int, int] = (0, 5),
    prompt_template: str = DEFAULT_SENTENCE_EVAL_PROMPT,
    max_attempts: int = 2,
    raise_on_invalid: bool = False,
    seed: int | None = None,
) -> Dict[str, object]:
    """Score how useful a sentence is for answering a question.

    Returns a dict with the raw model output, integer score, normalized score,
    and token usage stats.
    """
    min_score, max_score = _validate_scale(scale)
    if not question.strip() or not sentence.strip():
        return {
            "raw": "",
            "score": min_score,
            "score_normalized": _score_to_unit(min_score, scale),
            "invalid": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    prompt = build_sentence_evaluation_prompt(
        question,
        sentence,
        scale=scale,
        prompt_template=prompt_template,
    )
    grammar = _score_grammar(scale)
    reason = is_r1_like(model_name)

    last_raw = ""
    last_usage: Dict[str, int] = {}
    for _ in range(max_attempts):
        raw, usage = query_llm(
            prompt,
            server_url=server_url,
            max_tokens=MAX_TOKENS.get("cs", 128),  # TODO: add a dedicated setting
            temperature=TEMPERATURE.get("cs", 0.0),
            model_name=model_name,
            phase="cs",
            grammar=grammar,
            reason=reason,
            seed=seed,
            **LLM_DEFAULTS,
        )
        if is_r1_like(model_name):
            raw = strip_think(raw)
        last_raw = raw
        last_usage = usage

        score, invalid = _extract_score(raw, scale)
        if score is not None:
            return {
                "raw": raw,
                "score": score,
                "score_normalized": _score_to_unit(score, scale),
                "invalid": invalid,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get(
                    "total_tokens",
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                ),
            }

    if raise_on_invalid:
        raise ValueError(f"Invalid usefulness score: {last_raw!r}")

    return {
        "raw": last_raw,
        "score": min_score,
        "score_normalized": _score_to_unit(min_score, scale),
        "invalid": True,
        "prompt_tokens": last_usage.get("prompt_tokens", 0),
        "completion_tokens": last_usage.get("completion_tokens", 0),
        "total_tokens": last_usage.get(
            "total_tokens",
            last_usage.get("prompt_tokens", 0) + last_usage.get("completion_tokens", 0),
        ),
    }


def _score_sentences(
    question: str,
    sentences: Sequence[Dict[str, Any]],
    *,
    server_url: str,
    model_name: str,
    seed: int | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    totals = {
        "sentence_prompt_tokens": 0,
        "sentence_output_tokens": 0,
        "sentence_total_tokens": 0,
        "n_sentence_calls": 0,
    }
    scored: List[Dict[str, Any]] = []
    for item in sentences:
        result = evaluate_sentence_usefulness(
            question,
            item["text"],
            server_url=server_url,
            model_name=model_name,
            seed=seed,
        )
        scored_item = {
            **item,
            "score": result.get("score", 0),
            "score_normalized": result.get("score_normalized", 0.0),
            "invalid": result.get("invalid", True),
        }
        scored.append(scored_item)

        totals["sentence_prompt_tokens"] += result.get("prompt_tokens", 0)
        totals["sentence_output_tokens"] += result.get("completion_tokens", 0)
        totals["sentence_total_tokens"] += result.get("total_tokens", 0)
        totals["n_sentence_calls"] += 1

    return scored, totals


def _build_sentence_candidates(
    chunk_text: str,
    chunk_id: str,
    *,
    mode: str,
    extension: int,
    include_reformulation: bool,
    include_parenthetical: bool,
    require_forward_punct: bool,
) -> List[Dict[str, Any]]:
    sentences = split_chunk_sentences(chunk_text)
    candidates: List[Dict[str, Any]] = []
    if not sentences:
        return candidates

    if mode == "standard":
        for sent_idx, sent in enumerate(sentences):
            if not sent.strip():
                continue
            candidates.append(
                {
                    "sentence_id": f"{chunk_id}__sent{sent_idx}",
                    "chunk_id": chunk_id,
                    "sent_idx": sent_idx,
                    "span_start": sent_idx,
                    "span_end": sent_idx,
                    "expanded": False,
                    "text": sent,
                }
            )
        return candidates

    if mode != "discourse_aware":
        raise ValueError(f"Unknown SENTENCE_MODE: {mode}")

    for sent_idx, sent in enumerate(sentences):
        if not sent.strip():
            continue
        has_anaphora = _has_anaphora_marker(
            sent,
            include_reformulation=include_reformulation,
            include_parenthetical=include_parenthetical,
        )
        has_cataphora = _has_cataphora_marker(
            sent,
            include_reformulation=include_reformulation,
            require_forward_punct=require_forward_punct,
        )
        expanded_text, span_start, span_end = _expand_sentence_text(
            sentences,
            sent_idx,
            extension=extension,
            has_anaphora=has_anaphora,
            has_cataphora=has_cataphora,
        )
        candidates.append(
            {
                "sentence_id": f"{chunk_id}__sent{sent_idx}",
                "chunk_id": chunk_id,
                "sent_idx": sent_idx,
                "span_start": span_start,
                "span_end": span_end,
                "expanded": span_start != span_end,
                "text": expanded_text,
            }
        )
    return candidates


def _select_top_sentences(
    scored_sentences: Sequence[Dict[str, Any]],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        scored_sentences,
        key=lambda x: (x.get("score_normalized", 0.0), x.get("score", 0)),
        reverse=True,
    )
    return list(ranked[:top_k])


def _filter_sentences_by_threshold(
    scored_sentences: Sequence[Dict[str, Any]],
    *,
    threshold: float | None,
) -> List[Dict[str, Any]]:
    if threshold is None:
        return list(scored_sentences)
    return [
        s for s in scored_sentences if float(s.get("score_normalized", 0.0)) >= threshold
    ]
