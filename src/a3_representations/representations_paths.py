"""Path helpers for representation artifacts."""

from __future__ import annotations

import os
from typing import Dict


def dataset_rep_paths(dataset: str, split: str) -> Dict[str, str]:
    """Return paths for model-agnostic dataset-level passage representations."""

    base = os.path.join("data", "representations", "datasets", dataset, split)
    return {
        "passages_jsonl": os.path.join(base, f"{dataset}_passages.jsonl"),
        "passages_emb": os.path.join(base, f"{dataset}_passages_emb.npy"),
        "passages_index": os.path.join(base, f"{dataset}_faiss_passages.faiss"),
    }
