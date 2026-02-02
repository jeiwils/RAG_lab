"""Indexing utilities for preparing chunked passages."""

from src.a2_indexing.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_jsonl,
    chunk_record,
    chunk_records,
    chunk_text,
    make_chunk_id,
)
from src.a2_indexing.discourse_aware_chunking import (
    DiscourseAwareChunkingConfig,
    chunk_jsonl as discourse_aware_chunk_jsonl,
    chunk_record as discourse_aware_chunk_record,
    chunk_records as discourse_aware_chunk_records,
    chunk_text as discourse_aware_chunk_text,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "DiscourseAwareChunkingConfig",
    "chunk_jsonl",
    "chunk_record",
    "chunk_records",
    "chunk_text",
    "discourse_aware_chunk_jsonl",
    "discourse_aware_chunk_record",
    "discourse_aware_chunk_records",
    "discourse_aware_chunk_text",
    "make_chunk_id",
]
