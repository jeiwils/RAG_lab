"""Path helpers for representation artifacts."""

from __future__ import annotations

import os
from typing import Dict


def dataset_rep_paths(
    dataset: str,
    split: str,
    *,
    passage_source: str = "passages",
) -> Dict[str, str]:
    """Return paths for model-agnostic dataset-level passage representations."""

    base = os.path.join("data", "representations", "datasets", dataset, split)
    suffix = ""
    if passage_source and passage_source != "passages":
        safe_source = passage_source.replace(os.sep, "_").replace("/", "_")
        base = os.path.join(base, safe_source)
        suffix = f"_{safe_source}"
    return {
        "passages_jsonl": os.path.join(base, f"{dataset}{suffix}_passages.jsonl"),
        "passages_emb": os.path.join(base, f"{dataset}{suffix}_passages_emb.npy"),
        "passages_index": os.path.join(
            base, f"{dataset}{suffix}_faiss_passages.faiss"
        ),
    }
