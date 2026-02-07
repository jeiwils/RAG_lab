"""Orchestrator for building dense/sparse passage representations."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.a3_representations.dense_representations import (
    build_and_save_faiss_index,
    embed_and_save,
    get_embedding_model,
)
from src.a3_representations.sparse_representations import (
    SPACY_MODEL,
    add_keywords_to_passages_jsonl,
)
from src.a3_representations.representations_paths import dataset_rep_paths
from src.utils.__utils__ import (
    compute_resume_sets,
    load_jsonl,
    processed_dataset_paths,
)

RESUME = True

# Config
DATASETS = ["musique", "hotpotqa", "2wikimultihopqa", "natural_questions"]
SPLITS = ["dev"] #["train_sub", "val", "dev"]

PASSAGE_SOURCES = ["passages", "full_passages_chunks"]



def main() -> None:
    print(f"[spaCy] Using: {SPACY_MODEL}")

    bge_model = get_embedding_model()



    # -------------------------------
    # Phase A: Passages (Dataset-only)
    # -------------------------------
    for dataset in DATASETS:
        for split in SPLITS:
            for passage_source in PASSAGE_SOURCES:
                print(f"\n=== DATASET: {dataset} ({split}) [{passage_source}] ===")

                # File paths
                pass_paths = dataset_rep_paths(
                    dataset, split, passage_source=passage_source
                )
                dataset_dir = Path(os.path.dirname(pass_paths["passages_jsonl"]))
                os.makedirs(dataset_dir, exist_ok=True)

                source_paths = processed_dataset_paths(dataset, split)
                if passage_source not in source_paths:
                    raise KeyError(
                        f"Unknown PASSAGE_SOURCE '{passage_source}'. "
                        f"Available keys: {sorted(source_paths.keys())}"
                    )
                passages_jsonl_src = str(source_paths[passage_source])
                if not os.path.exists(passages_jsonl_src):
                    print(
                        f"[skip] missing {dataset}/{split} {passage_source}: "
                        f"{passages_jsonl_src}"
                    )
                    continue
                passages_jsonl = pass_paths["passages_jsonl"]
                passages_npy = pass_paths["passages_emb"]

                # === PASSAGE EMBEDDINGS ===
                if os.path.exists(passages_npy) and not RESUME:
                    passages_emb = np.load(passages_npy).astype("float32")
                    print(f"[skip] {passages_npy} exists; loaded.")
                    if not os.path.exists(pass_paths["passages_index"]):
                        build_and_save_faiss_index(
                            embeddings=passages_emb,
                            dataset_name=dataset,
                            index_type="passages",
                            output_dir=str(dataset_dir),
                        )
                else:
                    pass_items = load_jsonl(passages_jsonl_src)
                    done_ids, shard_ids = compute_resume_sets(
                        resume=RESUME,
                        out_path=str(passages_jsonl),
                        items=pass_items,
                        get_id=lambda x, i: x["passage_id"],
                        phase_label="passage embeddings",
                        required_field="vec_id",
                    )
                    new_ids = shard_ids - done_ids
                    passages_emb, new_pass_embs = embed_and_save(
                        input_jsonl=passages_jsonl_src,
                        output_npy=str(passages_npy),
                        output_jsonl=str(passages_jsonl),
                        model=bge_model,
                        text_key="text",
                        id_field="passage_id",
                        done_ids=done_ids,
                    )
                    if new_pass_embs.size > 0:
                        add_keywords_to_passages_jsonl(
                            str(passages_jsonl),
                            only_ids=new_ids,
                        )
                        build_and_save_faiss_index(
                            embeddings=passages_emb,
                            dataset_name=dataset,
                            index_type="passages",
                            output_dir=str(dataset_dir),
                            new_vectors=new_pass_embs,
                        )
                    elif not os.path.exists(pass_paths["passages_index"]):
                        build_and_save_faiss_index(
                            embeddings=passages_emb,
                            dataset_name=dataset,
                            index_type="passages",
                            output_dir=str(dataset_dir),
                        )

        # Dataset-level phase only handles passage embeddings and indexing.


if __name__ == "__main__":
    main()
