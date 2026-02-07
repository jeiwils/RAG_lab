"""Inference utilities for sentence usefulness extraction."""

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
    "count_tokens",
    "encode_batch",
    "is_lora_checkpoint",
    "load_sentence_lora",
    "score_sentences_lora",
    "select_ranked_sentences",
]

DEFAULT_TEXT_TEMPLATE = (
    "Query:\n"
    "{query}\n"
    "Full context:\n"
    "{context}\n"
    "Sentence:\n"
    "{sentence}\n"
    "Is this sentence useful in answering the\n"
    "query? Answer only \"Yes\" or \"No\""
)

DEFAULT_MAX_LEN = 256
LOCAL_FILES_ONLY = True

_TOKEN_RE = re.compile(r"\S+")


def encode_batch(
    tokenizer,
    batch: Sequence[Dict[str, Any]],
    *,
    max_length: int = DEFAULT_MAX_LEN,
    text_template: str = DEFAULT_TEXT_TEMPLATE,
):
    """Tokenize a batch of question + sentence pairs."""
    texts: List[str] = []
    labels: List[int] = []
    for ex in batch:
        sentence = str(ex.get("sentence", ""))
        context = str(ex.get("context", sentence))
        texts.append(
            text_template.format(
                query=str(ex.get("query", "")),
                context=context,
                sentence=sentence,
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


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace-delimited tokens."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def select_ranked_sentences(
    ranked: Sequence[Dict[str, Any]],
    *,
    token_budget: int | None,
    tau_low: float | None = None,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> List[Dict[str, Any]]:
    """Select ranked items under an optional token budget with a low-score guardrail."""
    enforce_budget = token_budget is not None
    if enforce_budget and token_budget <= 0:
        return []

    if count_tokens_fn is None:
        count_tokens_fn = count_tokens

    selected: List[Dict[str, Any]] = []
    total_tokens = 0

    for item in ranked:
        text = item.get("text")
        if text is None:
            text = item.get("sentence", "")
        sent_tokens = count_tokens_fn(str(text)) if enforce_budget else 0

        if enforce_budget and sent_tokens > token_budget:
            continue

        raw_score = item.get("score")
        if raw_score is None:
            raw_score = item.get("score_normalized", 0.0)
        score = float(raw_score)
        if tau_low is not None and score < tau_low:
            continue

        if enforce_budget and total_tokens + sent_tokens > token_budget:
            continue

        selected.append(item)
        if enforce_budget:
            total_tokens += sent_tokens
            if total_tokens >= token_budget:
                break

    return selected


def is_lora_checkpoint(model_name: str) -> bool:
    """Return True if ``model_name`` points to a LoRA checkpoint directory."""
    path = Path(model_name)
    return path.is_dir() and (path / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def load_sentence_lora(checkpoint_dir: str, *, local_files_only: bool = True):
    """Load a LoRA sentence classifier and its tokenizer once per process."""
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


def score_sentences_lora(
    question: str,
    sentences: List[Dict[str, Any]],
    *,
    checkpoint_dir: Path,
    batch_size: int,
    max_length: int,
    local_files_only: bool = LOCAL_FILES_ONLY,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Score sentence candidates with the LoRA classifier and return token stats."""
    totals = {
        "sentence_prompt_tokens": 0,
        "sentence_output_tokens": 0,
        "sentence_total_tokens": 0,
        "n_sentence_calls": 0,
    }
    if not sentences:
        return [], totals

    model, tokenizer, device = load_sentence_lora(
        str(checkpoint_dir), local_files_only=local_files_only
    )

    scored: List[Dict[str, Any]] = []
    for start in range(0, len(sentences), batch_size):
        batch_items = sentences[start : start + batch_size]
        batch = [
            {
                "query": question,
                "context": str(item.get("context", item.get("text", ""))),
                "sentence": str(item.get("text", "")),
                "label": 0,
            }
            for item in batch_items
        ]
        tokens, _ = encode_batch(tokenizer, batch, max_length=max_length)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            totals["sentence_prompt_tokens"] += int(attention_mask.sum().item())
        else:
            totals["sentence_prompt_tokens"] += int(tokens["input_ids"].numel())

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

        totals["n_sentence_calls"] += len(batch_items)

    totals["sentence_total_tokens"] = totals["sentence_prompt_tokens"]
    return scored, totals
