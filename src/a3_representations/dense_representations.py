"""Dense representation utilities (embeddings + FAISS indexing)."""

from __future__ import annotations

import json
import os
from typing import Set

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.__utils__ import load_jsonl

_bge_model = None


"""################## Embedding model + encoding ##################"""


def get_embedding_model():
    """Load and cache the BGE embedding model."""

    global _bge_model

    if _bge_model is None:
        model_name = os.environ.get("BGE_MODEL", "BAAI/bge-base-en-v1.5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bge_model = SentenceTransformer(model_name, device=device)
        print(f"[BGE] Loaded {model_name} on {device}")

    return _bge_model


def embed_and_save(
    input_jsonl,
    output_npy,
    output_jsonl,
    model,
    text_key,
    *,
    id_field="passage_id",
    done_ids: Set[str] | None = None,
    output_jsonl_input: str | None = None,
):
    """Embed texts from ``input_jsonl`` and save results.

    Parameters
    ----------
    input_jsonl: str
        Path to the JSONL file containing the text used for embedding.
    output_npy: str
        Destination path for the NumPy embedding array.
    output_jsonl: str
        Destination JSONL path where entries with ``vec_id`` are written.
    model: SentenceTransformer
        The embedding model.
    text_key: str
        Key in each JSON record that contains the text to be embedded.
    id_field: str, optional
        Field holding the unique identifier for each entry.
    done_ids: set[str], optional
        Set of IDs that already have embeddings and should be skipped.
    output_jsonl_input: str, optional
        Path to the JSONL file providing the metadata to write to ``output_jsonl``.
        If ``None``, ``input_jsonl`` is used.
    """

    if not text_key:
        raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")

    # If a separate source for the output JSONL is provided (e.g., a cleaned
    # version of the data), build a lookup by ID so we can pair the embedding
    # text from ``input_jsonl`` with the cleaned metadata.
    if output_jsonl_input is None:
        output_jsonl_input = input_jsonl

    by_id = {}
    if output_jsonl_input != input_jsonl:
        with open(output_jsonl_input, "rt", encoding="utf-8") as f_clean:
            for line in f_clean:
                entry = json.loads(line)
                by_id[entry[id_field]] = entry

    data, texts = [], []
    # ``load_jsonl`` handles blank or malformed lines gracefully which ensures
    # we iterate over *every* passage in the source file. Some datasets include
    # passages whose identifiers start with ``2hop__``; these should be embedded
    # just like any other passage and must not be skipped.
    for entry in load_jsonl(input_jsonl):
        entry_id = entry.get(id_field)
        if done_ids and entry_id in done_ids:
            continue
        texts.append(entry[text_key])
        if by_id:
            if entry_id not in by_id:
                raise KeyError(
                    f"{id_field} {entry_id} from {input_jsonl} not found in {output_jsonl_input}"
                )
            data.append(by_id[entry_id])
        else:
            data.append(entry)

    existing_embs = None
    vec_offset = 0
    if os.path.exists(output_npy):
        existing_embs = np.load(output_npy).astype("float32")
        vec_offset = existing_embs.shape[0]
        if os.path.exists(output_jsonl):
            with open(output_jsonl, "rt", encoding="utf-8") as f_old:
                idx = -1
                for idx, line in enumerate(f_old):
                    if json.loads(line).get("vec_id") != idx:
                        raise AssertionError(
                            f"vec_id mismatch at line {idx} in {output_jsonl}"
                        )
                if vec_offset != idx + 1:
                    raise AssertionError(
                        f"Embedding count {vec_offset} does not match JSONL entries {idx + 1}"
                    )

    if not data:
        if existing_embs is not None:
            embs_all = existing_embs
        else:
            embs_all = np.empty(
                (0, model.get_sentence_embedding_dimension()), dtype="float32"
            )
        print(f"[Embeddings] No new items for {input_jsonl}; skipping.")
        return embs_all, np.empty(
            (0, embs_all.shape[1] if embs_all.size else 0), dtype="float32"
        )

    new_embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    for i, entry in enumerate(data):
        entry["vec_id"] = i + vec_offset

    dir_path = os.path.dirname(output_npy)
    os.makedirs(dir_path or ".", exist_ok=True)
    if existing_embs is not None:
        embs_all = np.vstack([existing_embs, new_embs])
    else:
        embs_all = new_embs
    np.save(output_npy, embs_all)

    mode = "a" if vec_offset > 0 else "w"
    dir_path = os.path.dirname(output_jsonl)
    os.makedirs(dir_path or ".", exist_ok=True)
    with open(output_jsonl, mode + "t", encoding="utf-8") as f_out:
        for d in data:
            # ``ensure_ascii=False`` preserves non-ASCII characters in the
            # stored metadata, keeping a 1:1 correspondence with the source
            # passages.
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(
        f"[Embeddings] Saved {len(data)} new vectors to {output_npy} and updated JSONL {output_jsonl}"
    )
    return embs_all, new_embs


"""################## FAISS indexing ##################"""


def build_and_save_faiss_index(
    embeddings: np.ndarray,
    dataset_name: str,
    index_type: str,
    output_dir: str = ".",
    new_vectors: np.ndarray | None = None,
):
    """Build or update a FAISS cosine-similarity index.

    If ``new_vectors`` is provided and an existing index file is found, the new
    vectors are appended to that index. Otherwise, a fresh index is built from
    ``embeddings``.
    """

    if index_type != "passages":
        raise ValueError("index_type must be provided and set to 'passages'.")

    # Ensure float32, contiguous layout and normalize vectors for cosine sim
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    if new_vectors is not None:
        new_vectors = np.ascontiguousarray(new_vectors, dtype=np.float32)
        faiss.normalize_L2(new_vectors)

    faiss_path = os.path.join(output_dir, f"{dataset_name}_faiss_{index_type}.faiss")

    if new_vectors is not None and os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        assert new_vectors.shape[1] == index.d, (
            f"Dimension mismatch: new_vectors.shape[1]={new_vectors.shape[1]}, index.d={index.d}"
        )
        index.add(new_vectors)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        assert embeddings.shape[1] == index.d, (
            f"Dimension mismatch: embeddings.shape[1]={embeddings.shape[1]}, index.d={index.d}"
        )
        index.add(embeddings)

    faiss.write_index(index, faiss_path)

    print(f"[FAISS] Saved {index_type} index to {faiss_path} with {index.ntotal} vectors.")


def load_faiss_index(path: str):
    """Load a FAISS index from ``path``."""

    index = faiss.read_index(path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {path}")
    return index
