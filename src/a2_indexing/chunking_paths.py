"""Path helpers for chunked passage artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

__all__ = ["_chunk_paths"]


def _chunk_paths(dataset: str, split: str, chunking_tag: str) -> Dict[str, Path | str]:
    base = Path(f"data/representations/chunks/{dataset}/{split}/{chunking_tag}")
    dataset_tag = f"{dataset}_{chunking_tag}"
    return {
        "base": base,
        "dataset_tag": dataset_tag,
        "chunks_jsonl": base / f"{dataset_tag}_chunks.jsonl",
        "chunks_meta": base / f"{dataset_tag}_chunks_meta.jsonl",
        "chunks_emb": base / f"{dataset_tag}_chunks_emb.npy",
        "chunks_index": base / f"{dataset_tag}_faiss_passages.faiss",
    }
