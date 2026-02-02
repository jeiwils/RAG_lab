"""Discourse-aware sentence chunking with boundary extension near linking markers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from src.a2_indexing.chunking import Chunk, make_chunk_id
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


_TOKEN_RE = re.compile(r"\S+")


def _normalise_length_unit(length_unit: str) -> str:
    value = length_unit.strip().lower()
    if value in {"token", "tokens"}:
        return "tokens"
    if value in {"char", "chars", "character", "characters"}:
        return "chars"
    raise ValueError(f"Unsupported length_unit value: {length_unit}")


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


def _length_fn(config: DiscourseAwareChunkingConfig):
    if _normalise_length_unit(config.length_unit) == "chars":
        return len
    return lambda s: len(_TOKEN_RE.findall(s))


def _trim_span(text: str, start: int, end: int, strip_whitespace: bool) -> Tuple[int, int, str]:
    if not strip_whitespace:
        return start, end, text[start:end]
    span = text[start:end]
    if not span:
        return start, end, ""
    left_trim = len(span) - len(span.lstrip())
    right_trim = len(span) - len(span.rstrip())
    start += left_trim
    end -= right_trim
    if start >= end:
        return start, end, ""
    return start, end, text[start:end]


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


def _overlap_units(units: Sequence[_SentenceUnit], overlap: int) -> List[_SentenceUnit]:
    if overlap <= 0 or not units:
        return []
    total = 0
    suffix: List[_SentenceUnit] = []
    for unit in reversed(units):
        if suffix and total >= overlap:
            break
        suffix.append(unit)
        total += unit.length
        if total >= overlap:
            break
    return list(reversed(suffix))


def _chunk_long_unit(
    unit: _SentenceUnit,
    text: str,
    config: DiscourseAwareChunkingConfig,
    length_fn,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    max_len = config.max_length
    overlap = config.overlap
    step = max_len - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_length.")

    if _normalise_length_unit(config.length_unit) == "chars":
        start = unit.start
        while start < unit.end:
            end = min(start + max_len, unit.end)
            start, end, span = _trim_span(text, start, end, config.strip_whitespace)
            if span:
                chunks.append(
                    Chunk(
                        text=span,
                        index=-1,
                        start=start,
                        end=end,
                        length=length_fn(span),
                    )
                )
            if end >= unit.end:
                break
            start = end - overlap
        return chunks

    tokens = list(_TOKEN_RE.finditer(unit.text))
    if not tokens:
        return chunks
    for i in range(0, len(tokens), step):
        window = tokens[i : i + max_len]
        if not window:
            break
        start = unit.start + window[0].start()
        end = unit.start + window[-1].end()
        start, end, span = _trim_span(text, start, end, config.strip_whitespace)
        if span:
            chunks.append(
                Chunk(
                    text=span,
                    index=-1,
                    start=start,
                    end=end,
                    length=length_fn(span),
                )
            )
        if i + max_len >= len(tokens):
            break
    return chunks


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


def _reindex(chunks: List[Chunk]) -> List[Chunk]:
    return [
        Chunk(
            text=chunk.text,
            index=i,
            start=chunk.start,
            end=chunk.end,
            length=chunk.length,
        )
        for i, chunk in enumerate(chunks)
    ]


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
    if text_field not in record:
        raise KeyError(f"Missing {text_field} in record.")
    if id_field not in record:
        raise KeyError(f"Missing {id_field} in record.")

    text = record[text_field]
    source_id = record[id_field]
    chunks = chunk_text(text, config)

    output: List[Dict] = []
    for chunk in chunks:
        chunk_id = make_chunk_id(str(source_id), chunk.index)
        item = {
            output_id_field: chunk_id,
            parent_id_field: source_id,
            "chunk_index": chunk.index,
            "text": chunk.text,
            "char_start": chunk.start,
            "char_end": chunk.end,
        }
        if title_field and title_field in record:
            item[title_field] = record[title_field]
        for field in copy_fields:
            if field in record:
                item[field] = record[field]
        output.append(item)
    return output


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
    for record in records:
        for chunk in chunk_record(
            record,
            config,
            text_field=text_field,
            id_field=id_field,
            title_field=title_field,
            output_id_field=output_id_field,
            parent_id_field=parent_id_field,
            copy_fields=copy_fields,
        ):
            yield chunk


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
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        elif not resume:
            raise FileExistsError(
                f"{output_path} exists. Use overwrite=True or resume=True."
            )

    done_sources = set()
    if resume and os.path.exists(output_path):
        done_sources = existing_ids(output_path, id_field=parent_id_field)

    for record in load_jsonl(input_path):
        source_id = record.get(id_field)
        if resume and source_id in done_sources:
            continue
        for chunk in chunk_record(
            record,
            config,
            text_field=text_field,
            id_field=id_field,
            title_field=title_field,
            output_id_field=output_id_field,
            parent_id_field=parent_id_field,
            copy_fields=copy_fields,
        ):
            append_jsonl(output_path, chunk)

    return output_path
