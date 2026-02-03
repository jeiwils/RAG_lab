"""Simple passage chunking utilities for ingestion-time preprocessing."""

from __future__ import annotations

import re
from typing import List

from src.utils.algorithms.discourse_aware import (
    _has_anaphora_marker,
    _has_cataphora_marker,
    iter_sentence_spans,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_OVERLAP",
    "DEFAULT_THRESHOLD",
    "count_tokens",
    "split_long_passage",
    "split_long_passage_discourse_aware",
    "split_long_passage_discourse_aware_with_flags",
]

DEFAULT_THRESHOLD = 512 #### repetition? redundancy???
DEFAULT_MIN_TOKENS = 0
DEFAULT_MAX_TOKENS = 512 #### repetition? redundancy???
DEFAULT_OVERLAP = 0

_TOKEN_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace-delimited tokens."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def split_long_passage(
    text: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split a long passage into sentence-aligned sub-passages.

    If the passage length exceeds ``threshold`` characters, return chunks that
    each end on a sentence boundary and are closest to ``max_tokens``
    characters. Otherwise, return the original passage as a single-item list.
    """
    if not text:
        return []

    cleaned = text.strip()
    if len(cleaned) <= threshold:
        return [cleaned]

    sentences = [s.strip() for _, _, s in iter_sentence_spans(cleaned)]
    if not sentences:
        return [cleaned]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def _flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
        current = []
        current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if not current:
            current.append(sent)
            current_len = sent_len
            if current_len >= max_tokens:
                _flush_current()
            continue

        new_len = current_len + 1 + sent_len
        if new_len < max_tokens:
            current.append(sent)
            current_len = new_len
            continue

        dist_before = max_tokens - current_len
        dist_after = new_len - max_tokens
        if dist_before <= dist_after:
            _flush_current()
            current.append(sent)
            current_len = sent_len
            if current_len >= max_tokens:
                _flush_current()
        else:
            current.append(sent)
            current_len = new_len
            _flush_current()

    if current:
        _flush_current()

    return chunks


def _split_long_passage_discourse_aware_with_flags(
    text: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extension: int = 1,
    include_reformulation: bool = True,
    include_parenthetical: bool = True,
    require_forward_punct: bool = True,
) -> tuple[List[str], List[bool]]:
    """Split a long passage into discourse-aware, sentence-aligned sub-passages.

    Uses the same sentence splitting as :func:`split_long_passage`, but applies
    backward/forward expansion only at chunk boundaries based on discourse markers.
    """
    if not text:
        return [], []

    cleaned = text.strip()
    if len(cleaned) <= threshold:
        return [cleaned], [False]

    sentences = [s.strip() for _, _, s in iter_sentence_spans(cleaned)]
    if not sentences:
        return [cleaned], [False]

    chunks: List[str] = []
    extended_flags: List[bool] = []
    n = len(sentences)
    start = 0
    pending_extend_next = False
    while start < n:
        current_len = len(sentences[start])
        if current_len >= max_tokens:
            chunk_len = 1
            chunks.append(sentences[start].strip())
            extended_flags.append(pending_extend_next and chunk_len > 1)
            pending_extend_next = False
            start += 1
            continue

        i = start + 1
        while i < n:
            sent_len = len(sentences[i])
            new_len = current_len + 1 + sent_len
            if new_len < max_tokens:
                current_len = new_len
                i += 1
                continue

            dist_before = max_tokens - current_len
            dist_after = new_len - max_tokens
            if dist_before <= dist_after:
                split_idx = i
            else:
                current_len = new_len
                i += 1
                split_idx = i

            last_idx = split_idx - 1
            next_idx = split_idx if split_idx < n else None
            extended_current = False
            extended_next = False

            if next_idx is not None:
                last_sent = sentences[last_idx]
                next_sent = sentences[next_idx]
                if _has_cataphora_marker(
                    last_sent,
                    include_reformulation=include_reformulation,
                    require_forward_punct=require_forward_punct,
                ):
                    shift = min(extension, n - split_idx)
                    if shift > 0:
                        split_idx += shift
                        extended_current = True
                elif _has_anaphora_marker(
                    next_sent,
                    include_reformulation=include_reformulation,
                    include_parenthetical=include_parenthetical,
                ):
                    max_back = split_idx - start - 1
                    if max_back > 0:
                        shift = min(extension, max_back)
                        if shift > 0:
                            split_idx -= shift
                            extended_next = True

            split_idx = max(start + 1, min(split_idx, n))
            chunk_len = split_idx - start
            chunks.append(" ".join(sentences[start:split_idx]).strip())
            extended_flags.append((extended_current or pending_extend_next) and chunk_len > 1)
            pending_extend_next = extended_next
            start = split_idx
            break
        else:
            chunk_len = n - start
            chunks.append(" ".join(sentences[start:n]).strip())
            extended_flags.append(pending_extend_next and chunk_len > 1)
            pending_extend_next = False
            break

    return chunks, extended_flags


def split_long_passage_discourse_aware(
    text: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extension: int = 1,
    include_reformulation: bool = True,
    include_parenthetical: bool = True,
    require_forward_punct: bool = True,
) -> List[str]:
    chunks, _ = _split_long_passage_discourse_aware_with_flags(
        text,
        threshold=threshold,
        max_tokens=max_tokens,
        extension=extension,
        include_reformulation=include_reformulation,
        include_parenthetical=include_parenthetical,
        require_forward_punct=require_forward_punct,
    )
    return chunks


def split_long_passage_discourse_aware_with_flags(
    text: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extension: int = 1,
    include_reformulation: bool = True,
    include_parenthetical: bool = True,
    require_forward_punct: bool = True,
) -> tuple[List[str], List[bool]]:
    return _split_long_passage_discourse_aware_with_flags(
        text,
        threshold=threshold,
        max_tokens=max_tokens,
        extension=extension,
        include_reformulation=include_reformulation,
        include_parenthetical=include_parenthetical,
        require_forward_punct=require_forward_punct,
    )
