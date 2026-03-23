"""Discourse-aware sentence chunking with boundary extension near linking markers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from src.a2_indexing.chunking import (
    Chunk,
    _chunk_long_unit,
    _length_fn,
    _normalise_length_unit,
    _overlap_units,
    _reindex,
    _trim_span,
    make_chunk_id,
)
from src.utils.__utils__ import append_jsonl, existing_ids, load_jsonl
from src.utils.algorithms.discourse_aware import (
    find_anaphora_markers,
    find_cataphora_markers,
    find_temporal_markers,
    iter_sentence_spans,
)

__all__ = [
    "Chunk",
    "DiscourseAwareChunkingConfig",
    "chunk_jsonl",
    "chunk_record",
    "chunk_records",
    "chunk_text",
    "make_chunk_id",
]


@dataclass(frozen=True)
class DiscourseAwareChunkingConfig:
    """Configuration for discourse-aware sentence chunking.

    Parameters
    ----------
    max_length:
        Maximum size of a chunk measured in ``length_unit``.
    overlap:
        Overlap between consecutive chunks measured in ``length_unit``.
        Overlap is applied at the sentence level, so it is approximate.
    min_length:
        Minimum size of the final chunk. Smaller tails are merged into the
        previous chunk when possible.
    length_unit:
        Measurement unit for chunk length; supported values are
        ``"tokens"`` or ``"chars"``.
    strip_whitespace:
        Whether to trim leading/trailing whitespace from sentences and chunks.
    extension_sentences:
        Number of sentences to extend when a chunk boundary starts or ends
        on a discourse marker.
    include_reformulation:
        Whether to include reformulation markers (Category E).
    include_parenthetical:
        Whether to include parenthetical markers (Category B).
    include_temporal:
        Whether to include temporal markers (Category F).
    require_forward_punct:
        Whether to require forward punctuation for cataphora markers.
    """

    max_length: int = 200
    overlap: int = 30
    min_length: int = 50
    length_unit: str = "tokens"
    strip_whitespace: bool = True
    extension_sentences: int = 1
    include_reformulation: bool = True
    include_parenthetical: bool = True
    include_temporal: bool = True
    require_forward_punct: bool = False


@dataclass(frozen=True)
class _SentenceUnit:
    text: str
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class _ChunkSpan:
    sent_start: int | None
    sent_end: int | None
    start_aligned: bool
    end_aligned: bool


def _validate_config(config: DiscourseAwareChunkingConfig) -> DiscourseAwareChunkingConfig:
    if config.max_length <= 0:
        raise ValueError("max_length must be positive.")
    if config.overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if config.overlap >= config.max_length:
        raise ValueError("overlap must be smaller than max_length.")
    if config.min_length < 0:
        raise ValueError("min_length cannot be negative.")
    if config.min_length > config.max_length:
        raise ValueError("min_length cannot exceed max_length.")
    if config.extension_sentences < 0:
        raise ValueError("extension_sentences cannot be negative.")
    _normalise_length_unit(config.length_unit)
    return config


def _build_sentence_units(
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> List[_SentenceUnit]:
    units: List[_SentenceUnit] = []
    for start, end, _ in iter_sentence_spans(text):
        start, end, span = _trim_span(text, start, end, config.strip_whitespace)
        if not span:
            continue
        units.append(_SentenceUnit(span, start, end, length_fn(span)))
    return units


def _units_to_chunk(
    units: Sequence[_SentenceUnit],
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> Chunk:
    start = units[0].start
    end = units[-1].end
    start, end, chunk_text = _trim_span(text, start, end, config.strip_whitespace)
    length = length_fn(chunk_text)
    return Chunk(text=chunk_text, index=-1, start=start, end=end, length=length)


def _chunk_sentence_units(
    units: Sequence[_SentenceUnit],
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> Tuple[List[Chunk], List[_ChunkSpan]]:
    chunks: List[Chunk] = []
    spans: List[_ChunkSpan] = []
    current: List[_SentenceUnit] = []
    current_len = 0
    current_start_idx: int | None = None

    for idx, unit in enumerate(units):
        if unit.length > config.max_length:
            if current:
                chunks.append(_units_to_chunk(current, text, config, length_fn))
                spans.append(
                    _ChunkSpan(
                        sent_start=current_start_idx,
                        sent_end=idx - 1,
                        start_aligned=True,
                        end_aligned=True,
                    )
                )
                current = []
                current_len = 0
                current_start_idx = None
            for sub_chunk in _chunk_long_unit(unit, text, config, length_fn):
                spans.append(
                    _ChunkSpan(
                        sent_start=idx,
                        sent_end=idx,
                        start_aligned=sub_chunk.start == unit.start,
                        end_aligned=sub_chunk.end == unit.end,
                    )
                )
                chunks.append(sub_chunk)
            continue

        if not current:
            current_start_idx = idx

        if current_len + unit.length <= config.max_length:
            current.append(unit)
            current_len += unit.length
            continue

        chunks.append(_units_to_chunk(current, text, config, length_fn))
        spans.append(
            _ChunkSpan(
                sent_start=current_start_idx,
                sent_end=idx - 1,
                start_aligned=True,
                end_aligned=True,
            )
        )

        overlap_units = _overlap_units(current, config.overlap)
        current = list(overlap_units)
        current_len = sum(u.length for u in current)
        if current:
            current_start_idx = idx - len(current)
        else:
            current_start_idx = None

        if current and current_len + unit.length > config.max_length:
            chunks.append(_units_to_chunk(current, text, config, length_fn))
            spans.append(
                _ChunkSpan(
                    sent_start=current_start_idx,
                    sent_end=idx - 1,
                    start_aligned=True,
                    end_aligned=True,
                )
            )
            current = []
            current_len = 0
            current_start_idx = None

        if not current:
            current_start_idx = idx
        current.append(unit)
        current_len += unit.length

    if current:
        end_idx = (
            current_start_idx + len(current) - 1 if current_start_idx is not None else None
        )
        chunks.append(_units_to_chunk(current, text, config, length_fn))
        spans.append(
            _ChunkSpan(
                sent_start=current_start_idx,
                sent_end=end_idx,
                start_aligned=True,
                end_aligned=True,
            )
        )

    return chunks, spans


def _merge_short_tail(
    chunks: List[Chunk],
    spans: List[_ChunkSpan],
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> Tuple[List[Chunk], List[_ChunkSpan]]:
    if config.min_length <= 0 or len(chunks) < 2:
        return chunks, spans
    tail = chunks[-1]
    if tail.length >= config.min_length:
        return chunks, spans
    head = chunks[-2]
    start = head.start
    end = tail.end
    start, end, span = _trim_span(text, start, end, config.strip_whitespace)
    merged = Chunk(
        text=span,
        index=-1,
        start=start,
        end=end,
        length=length_fn(span),
    )
    merged_span = _ChunkSpan(
        sent_start=spans[-2].sent_start,
        sent_end=spans[-1].sent_end,
        start_aligned=spans[-2].start_aligned,
        end_aligned=spans[-1].end_aligned,
    )
    return chunks[:-2] + [merged], spans[:-2] + [merged_span]


def _marker_spans(text: str, config: DiscourseAwareChunkingConfig) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    spans.extend(
        (hit.start, hit.end)
        for hit in find_anaphora_markers(
            text,
            include_reformulation=config.include_reformulation,
            include_parenthetical=config.include_parenthetical,
        )
    )
    spans.extend(
        (hit.start, hit.end)
        for hit in find_cataphora_markers(
            text,
            include_reformulation=config.include_reformulation,
            require_forward_punct=config.require_forward_punct,
        )
    )
    if config.include_temporal:
        spans.extend((hit.start, hit.end) for hit in find_temporal_markers(text))
    if not spans:
        return []
    return sorted(set(spans))


def _sentence_marker_flags(
    text: str,
    units: Sequence[_SentenceUnit],
    config: DiscourseAwareChunkingConfig,
) -> List[bool]:
    flags = [False] * len(units)
    if not units:
        return flags
    spans = _marker_spans(text, config)
    if not spans:
        return flags
    span_idx = 0
    for i, unit in enumerate(units):
        start = unit.start
        end = unit.end
        while span_idx < len(spans) and spans[span_idx][1] <= start:
            span_idx += 1
        if span_idx >= len(spans):
            break
        if spans[span_idx][0] < end:
            flags[i] = True
    return flags


def _extend_chunks_by_markers(
    chunks: List[Chunk],
    spans: Sequence[_ChunkSpan],
    units: Sequence[_SentenceUnit],
    sentence_has_marker: Sequence[bool],
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> List[Chunk]:
    if config.extension_sentences <= 0 or not units:
        return chunks
    max_idx = len(units) - 1
    extended: List[Chunk] = []
    for chunk, meta in zip(chunks, spans):
        new_start = chunk.start
        new_end = chunk.end
        if (
            meta.sent_start is not None
            and meta.start_aligned
            and meta.sent_start > 0
            and sentence_has_marker[meta.sent_start]
        ):
            start_idx = max(0, meta.sent_start - config.extension_sentences)
            new_start = units[start_idx].start
        if (
            meta.sent_end is not None
            and meta.end_aligned
            and meta.sent_end < max_idx
            and sentence_has_marker[meta.sent_end]
        ):
            end_idx = min(max_idx, meta.sent_end + config.extension_sentences)
            new_end = units[end_idx].end
        if new_start != chunk.start or new_end != chunk.end:
            new_start, new_end, span = _trim_span(
                text, new_start, new_end, config.strip_whitespace
            )
            chunk = Chunk(
                text=span,
                index=-1,
                start=new_start,
                end=new_end,
                length=length_fn(span),
            )
        extended.append(chunk)
    return extended


def chunk_text(text: str, config: DiscourseAwareChunkingConfig) -> List[Chunk]:
    """Split ``text`` into discourse-aware chunks."""
    _validate_config(config)
    if not text:
        return []

    length_fn = _length_fn(config)
    units = _build_sentence_units(text, config, length_fn)
    if not units:
        return []

    chunks, spans = _chunk_sentence_units(units, text, config, length_fn)
    chunks, spans = _merge_short_tail(chunks, spans, text, config, length_fn)

    if config.extension_sentences > 0:
        sentence_has_marker = _sentence_marker_flags(text, units, config)
        chunks = _extend_chunks_by_markers(
            chunks,
            spans,
            units,
            sentence_has_marker,
            text,
            config,
            length_fn,
        )

    return _reindex(chunks)


def chunk_record(
    record: Dict,
    config: DiscourseAwareChunkingConfig,
    *,
    text_field: str = "text",
    id_field: str = "passage_id",
    title_field: str | None = "title",
    output_id_field: str = "chunk_id",
    parent_id_field: str = "source_id",
    copy_fields: Sequence[str] = (),
) -> List[Dict]:
    """Chunk a single record into chunk dictionaries."""
    from src.a2_indexing.chunking import _chunk_record_generic

    return _chunk_record_generic(
        record,
        config,
        chunk_text,
        text_field=text_field,
        id_field=id_field,
        title_field=title_field,
        output_id_field=output_id_field,
        parent_id_field=parent_id_field,
        copy_fields=copy_fields,
    )


def chunk_records(
    records: Iterable[Dict],
    config: DiscourseAwareChunkingConfig,
    *,
    text_field: str = "text",
    id_field: str = "passage_id",
    title_field: str | None = "title",
    output_id_field: str = "chunk_id",
    parent_id_field: str = "source_id",
    copy_fields: Sequence[str] = (),
) -> Iterator[Dict]:
    """Yield chunk dictionaries for an iterable of records."""
    from src.a2_indexing.chunking import _chunk_records_generic

    return _chunk_records_generic(
        records,
        config,
        chunk_text,
        text_field=text_field,
        id_field=id_field,
        title_field=title_field,
        output_id_field=output_id_field,
        parent_id_field=parent_id_field,
        copy_fields=copy_fields,
    )


def chunk_jsonl(
    input_path: str,
    output_path: str,
    config: DiscourseAwareChunkingConfig,
    *,
    text_field: str = "text",
    id_field: str = "passage_id",
    title_field: str | None = "title",
    output_id_field: str = "chunk_id",
    parent_id_field: str = "source_id",
    copy_fields: Sequence[str] = (),
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    """Chunk a JSONL file and write chunked records to ``output_path``."""
    from src.a2_indexing.chunking import _chunk_jsonl_generic

    return _chunk_jsonl_generic(
        input_path,
        output_path,
        config,
        chunk_text,
        text_field=text_field,
        id_field=id_field,
        title_field=title_field,
        output_id_field=output_id_field,
        parent_id_field=parent_id_field,
        copy_fields=copy_fields,
        resume=resume,
        overwrite=overwrite,
    )
