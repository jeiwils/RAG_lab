"""Representation utilities for dense/sparse embeddings and indexing."""

from src.a3_representations.dense_representations import (
    build_and_save_faiss_index,
    embed_and_save,
    get_embedding_model,
    load_faiss_index,
)
from src.a3_representations.sparse_representations import (
    add_keywords_to_passages_jsonl,
    extract_keywords,
)
from src.a3_representations.representations_paths import dataset_rep_paths

__all__ = [
    "add_keywords_to_passages_jsonl",
    "build_and_save_faiss_index",
    "dataset_rep_paths",
    "embed_and_save",
    "extract_keywords",
    "get_embedding_model",
    "load_faiss_index",
]
