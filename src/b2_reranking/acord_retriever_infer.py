"""Inference utilities for the ACORD near-miss-graph-aware retriever."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

__all__ = [
    "DEFAULT_TEXT_TEMPLATE",
    "DEFAULT_MAX_LEN",
    "LOCAL_FILES_ONLY",
    "encode_batch",
    "is_lora_checkpoint",
    "load_retriever_lora",
    "score_documents_lora",
]

DEFAULT_TEXT_TEMPLATE = (
    "Query:\n"
    "{query}\n"
    "Clause:\n"
    "{document}\n"
    "Is this clause the legally correct match for the query? "
    "Answer only \"Yes\" or \"No\"."
)

DEFAULT_MAX_LEN = 512
LOCAL_FILES_ONLY = True

_TOKEN_RE = re.compile(r"\S+")


def encode_batch(
    tokenizer,
    batch: Sequence[Dict[str, Any]],
    *,
    max_length: int = DEFAULT_MAX_LEN,
    text_template: str = DEFAULT_TEXT_TEMPLATE,
):
    """Tokenize a batch of query-document pairs."""
    texts: List[str] = []
    labels: List[int] = []
    for ex in batch:
        document = str(
            ex.get("document", ex.get("text", ex.get("doc_text", "")))
        )
        texts.append(
            text_template.format(
                query=str(ex.get("query", "")),
                document=document,
            )
        )
        labels.append(int(ex.get("label", 0)))

    tokens = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokens, torch.tensor(labels, dtype=torch.float32)


def is_lora_checkpoint(model_name: str) -> bool:
    """Return True if ``model_name`` points to a LoRA checkpoint directory."""
    path = Path(model_name)
    return path.is_dir() and (path / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def load_retriever_lora(checkpoint_dir: str, *, local_files_only: bool = True):
    """Load a LoRA retriever classifier and its tokenizer once per process."""
    ckpt = Path(checkpoint_dir)
    config_path = ckpt / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {config_path}")
    with open(config_path, "rt", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"Missing base_model_name_or_path in {config_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt), local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        use_safetensors=None,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base, str(ckpt))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def score_documents_lora(
    query: str,
    documents: List[Dict[str, Any]],
    *,
    checkpoint_dir: Path,
    batch_size: int,
    max_length: int,
    local_files_only: bool = LOCAL_FILES_ONLY,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Score document candidates with the LoRA retriever."""
    totals = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "n_calls": 0,
    }
    if not documents:
        return [], totals

    model, tokenizer, device = load_retriever_lora(
        str(checkpoint_dir), local_files_only=local_files_only
    )

    scored: List[Dict[str, Any]] = []
    for start in range(0, len(documents), batch_size):
        batch_items = documents[start : start + batch_size]
        batch = [
            {
                "query": query,
                "document": str(
                    item.get("document", item.get("text", item.get("doc_text", "")))
                ),
                "label": 0,
            }
            for item in batch_items
        ]
        tokens, _ = encode_batch(tokenizer, batch, max_length=max_length)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            totals["prompt_tokens"] += int(attention_mask.sum().item())
        else:
            totals["prompt_tokens"] += int(tokens["input_ids"].numel())

        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = model(**tokens).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        prob_list = probs.detach().cpu().tolist()

        for item, prob in zip(batch_items, prob_list):
            score = float(prob)
            scored.append(
                {
                    **item,
                    "score": score,
                    "score_normalized": score,
                    "invalid": False,
                }
            )

        totals["n_calls"] += len(batch_items)

    totals["total_tokens"] = totals["prompt_tokens"]
    return scored, totals
