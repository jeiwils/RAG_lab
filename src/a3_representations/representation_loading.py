"""Helpers for loading passage metadata and FAISS indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.a3_representations.dense_representations import load_faiss_index
from src.utils.__utils__ import load_jsonl

__all__ = ["load_chunk_representations"]


def load_chunk_representations(
    rep_paths: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Any]:
    """Load precomputed chunk metadata and FAISS index from dataset representations."""
    jsonl_path = Path(rep_paths["passages_jsonl"])
    index_path = Path(rep_paths["passages_index"])
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing representations JSONL: {jsonl_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    metadata = list(load_jsonl(str(jsonl_path)))
    index = load_faiss_index(str(index_path))
    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size mismatch: "
            f"index has {index.ntotal} vectors but metadata lists {len(metadata)} chunks"
        )
    return metadata, index
