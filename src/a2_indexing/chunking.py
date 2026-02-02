"""Chunking utilities for splitting text into fixed-length segments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, TYPE_CHECKING

from src.a2_indexing.chunking_paths import _chunk_paths
from src.utils.__utils__ import append_jsonl, existing_ids, load_jsonl, processed_dataset_paths

if TYPE_CHECKING:
    from src.a2_indexing.discourse_aware_chunking import DiscourseAwareChunkingConfig

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "chunk_jsonl",
    "chunk_record",
    "chunk_records",
    "chunk_text",
    "make_chunk_id",
    "_ensure_chunked_passages",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text chunking.

    Parameters
    ----------
    max_length:
        Maximum size of a chunk measured in ``length_unit``.
    overlap:
        Overlap between consecutive chunks measured in ``length_unit``.
        Overlap is applied at the unit level (sentence/paragraph/token),
        so it is approximate unless ``split_on="token"``.
    min_length:
        Minimum size of the final chunk. Smaller tails are merged into the
        previous chunk when possible.
    length_unit:
        Measurement unit for chunk length; supported values are
        ``"tokens"`` or ``"chars"``.
    split_on:
        Unit boundary used before chunk assembly; supported values are
        ``"sentence"``, ``"paragraph"``, ``"line"``, or ``"token"``.
    strip_whitespace:
        Whether to trim leading/trailing whitespace from units and chunks.
    """

    max_length: int = 200
    overlap: int = 30
    min_length: int = 50
    length_unit: str = "tokens"
    split_on: str = "sentence"
    strip_whitespace: bool = True


@dataclass(frozen=True)
class Chunk:
    """A single chunk and its position within the source text."""

    text: str
    index: int
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class _Unit:
    text: str
    start: int
    end: int
    length: int


_TOKEN_RE = re.compile(r"\S+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"(?:\r?\n)\s*(?:\r?\n)+")
_LINE_BOUNDARY_RE = re.compile(r"\r?\n+")


def make_chunk_id(source_id: str, chunk_index: int, *, sep: str = "__chunk") -> str:
    """Return a deterministic chunk identifier for a source id."""

    return f"{source_id}{sep}{chunk_index}"


def _normalise_split_on(split_on: str) -> str:
    value = split_on.strip().lower()
    if value in {"sentence", "sentences"}:
        return "sentence"
    if value in {"paragraph", "paragraphs"}:
        return "paragraph"
    if value in {"line", "lines"}:
        return "line"
    if value in {"token", "tokens", "word", "words"}:
        return "token"
    raise ValueError(f"Unsupported split_on value: {split_on}")


def _normalise_length_unit(length_unit: str) -> str:
    value = length_unit.strip().lower()
    if value in {"token", "tokens"}:
        return "tokens"
    if value in {"char", "chars", "character", "characters"}:
        return "chars"
    raise ValueError(f"Unsupported length_unit value: {length_unit}")


def _validate_config(config: ChunkingConfig) -> ChunkingConfig:
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
    _normalise_split_on(config.split_on)
    _normalise_length_unit(config.length_unit)
    return config


def _length_fn(config: ChunkingConfig):
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


def _iter_spans(text: str, split_on: str) -> Iterator[Tuple[int, int]]:
    mode = _normalise_split_on(split_on)
    if mode == "token":
        for m in _TOKEN_RE.finditer(text):
            yield m.start(), m.end()
        return
    if mode == "sentence":
        start = 0
        for m in _SENTENCE_BOUNDARY_RE.finditer(text):
            end = m.start()
            if start < end:
                yield start, end
            start = m.end()
        if start < len(text):
            yield start, len(text)
        return
    if mode == "paragraph":
        start = 0
        for m in _PARAGRAPH_BOUNDARY_RE.finditer(text):
            end = m.start()
            if start < end:
                yield start, end
            start = m.end()
        if start < len(text):
            yield start, len(text)
        return
    if mode == "line":
        start = 0
        for m in _LINE_BOUNDARY_RE.finditer(text):
            end = m.start()
            if start < end:
                yield start, end
            start = m.end()
        if start < len(text):
            yield start, len(text)
        return
    raise ValueError(f"Unsupported split_on mode: {split_on}")


def _build_units(
    text: str,
    config: ChunkingConfig,
    length_fn,
) -> List[_Unit]:
    units: List[_Unit] = []
    for start, end in _iter_spans(text, config.split_on):
        start, end, span = _trim_span(text, start, end, config.strip_whitespace)
        if not span:
            continue
        units.append(_Unit(span, start, end, length_fn(span)))
    return units


def _units_to_chunk(
    units: Sequence[_Unit],
    text: str,
    config: ChunkingConfig,
    length_fn,
) -> Chunk:
    start = units[0].start
    end = units[-1].end
    start, end, chunk_text = _trim_span(text, start, end, config.strip_whitespace)
    length = length_fn(chunk_text)
    return Chunk(text=chunk_text, index=-1, start=start, end=end, length=length)


def _overlap_units(units: Sequence[_Unit], overlap: int) -> List[_Unit]:
    if overlap <= 0 or not units:
        return []
    total = 0
    suffix: List[_Unit] = []
    for unit in reversed(units):
        if suffix and total >= overlap:
            break
        suffix.append(unit)
        total += unit.length
        if total >= overlap:
            break
    return list(reversed(suffix))


def _chunk_long_unit(
    unit: _Unit,
    text: str,
    config: ChunkingConfig,
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


def _merge_short_tail(
    chunks: List[Chunk],
    text: str,
    config: ChunkingConfig,
    length_fn,
) -> List[Chunk]:
    if config.min_length <= 0 or len(chunks) < 2:
        return chunks
    tail = chunks[-1]
    if tail.length >= config.min_length:
        return chunks
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
    return chunks[:-2] + [merged]


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


def chunk_text(text: str, config: ChunkingConfig) -> List[Chunk]:
    """Split ``text`` into chunks according to ``config``."""

    _validate_config(config)
    if not text:
        return []

    length_fn = _length_fn(config)
    units = _build_units(text, config, length_fn)
    if not units:
        return []

    chunks: List[Chunk] = []
    current: List[_Unit] = []
    current_len = 0

    for unit in units:
        if unit.length > config.max_length:
            if current:
                chunks.append(_units_to_chunk(current, text, config, length_fn))
                current = []
                current_len = 0
            chunks.extend(_chunk_long_unit(unit, text, config, length_fn))
            continue

        if not current or current_len + unit.length <= config.max_length:
            current.append(unit)
            current_len += unit.length
            continue

        chunks.append(_units_to_chunk(current, text, config, length_fn))
        overlap_units = _overlap_units(current, config.overlap)
        current = list(overlap_units)
        current_len = sum(u.length for u in current)

        if current and current_len + unit.length > config.max_length:
            chunks.append(_units_to_chunk(current, text, config, length_fn))
            current = []
            current_len = 0

        current.append(unit)
        current_len += unit.length

    if current:
        chunks.append(_units_to_chunk(current, text, config, length_fn))

    chunks = _merge_short_tail(chunks, text, config, length_fn)
    return _reindex(chunks)


def chunk_record(
    record: Dict,
    config: ChunkingConfig,
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
    config: ChunkingConfig,
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
    config: ChunkingConfig,
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


def _ensure_chunked_passages(
    dataset: str,
    split: str,
    *,
    chunking_mode: str,
    resume: bool,
    run_chunking: bool,
    standard_config: ChunkingConfig,
    discourse_config: "DiscourseAwareChunkingConfig",
    passages_path: Path | str | None = None,
) -> Dict[str, Path | str]:
    if chunking_mode not in {"standard", "discourse_aware"}:
        raise ValueError(f"Unknown chunking_mode: {chunking_mode}")

    chunk_paths = _chunk_paths(dataset, split, chunking_mode)
    chunk_paths["base"].mkdir(parents=True, exist_ok=True)

    if not run_chunking and not chunk_paths["chunks_jsonl"].exists():
        raise FileNotFoundError(
            f"Chunked passages missing: {chunk_paths['chunks_jsonl']}"
        )

    if run_chunking:
        if passages_path is None:
            src_path = processed_dataset_paths(dataset, split)["passages"]
        else:
            src_path = Path(passages_path)
        if chunking_mode == "discourse_aware":
            from src.a2_indexing.discourse_aware_chunking import (
                chunk_jsonl as discourse_aware_chunk_jsonl,
            )

            config = discourse_config
            chunk_fn = discourse_aware_chunk_jsonl
        else:
            config = standard_config
            chunk_fn = chunk_jsonl
        chunk_fn(
            input_path=str(src_path),
            output_path=str(chunk_paths["chunks_jsonl"]),
            config=config,
            resume=resume,
            overwrite=not resume,
        )

    return chunk_paths
