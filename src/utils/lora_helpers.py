"""Reusable helpers for LoRA training and evaluation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypeVar

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from src.d1_evaluation.classification_metrics import binary_f1

__all__ = [
    "evaluate_and_choose_tau",
    "find_attention_projection_modules",
    "init_model_with_lora",
    "make_minibatches",
    "move_to_device",
    "sample_without_replacement",
    "save_checkpoint",
    "seed_everything",
    "trainable_parameters",
]

T = TypeVar("T")


def find_attention_projection_modules(model) -> List[str]:
    """Best-effort detection of attention projection module names."""
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    preferred_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["query_proj", "key_proj", "value_proj", "output_proj"],
        ["query_proj", "key_proj", "value_proj"],
        ["query", "key", "value", "out_proj"],
        ["in_proj", "out_proj"],
    ]
    for group in preferred_groups:
        found = [name for name in group if name in present]
        if len(found) >= 3:
            return found
    for group in preferred_groups:
        found = [name for name in group if name in present]
        if found:
            return found
    return preferred_groups[0]


def init_model_with_lora(
    base_model_name: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str] | None = None,
    num_labels: int = 1,
    task_type: Any | None = None,
    use_safetensors: bool = True,
    local_files_only: bool = False,
    torch_dtype: torch.dtype = torch.float32,
):
    """Load a base model and attach LoRA adapters."""
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    if target_modules is None:
        target_modules = find_attention_projection_modules(base)

    if task_type is None:
        task_type = TaskType.SEQ_CLS

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(base, lora_cfg)
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_minibatches(
    examples: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
    stratified: bool = False,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield minibatches of examples.

    When shuffling, attempt to ensure each batch contains at least one positive
    label if possible.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    items = list(examples)
    if not shuffle or not stratified:
        for idx in range(0, len(items), batch_size):
            yield items[idx : idx + batch_size]
        return

    rng = random.Random(seed)
    pos_items = [ex for ex in items if int(ex.get("label", 0)) == 1]
    neg_items = [ex for ex in items if int(ex.get("label", 0)) == 0]

    if not pos_items or not neg_items:
        rng.shuffle(items)
        for idx in range(0, len(items), batch_size):
            yield items[idx : idx + batch_size]
        return

    rng.shuffle(pos_items)
    rng.shuffle(neg_items)

    num_batches = math.ceil(len(items) / batch_size)
    pos_cursor = 0
    neg_cursor = 0
    for _ in range(num_batches):
        batch: List[Dict[str, Any]] = []
        if pos_cursor < len(pos_items):
            batch.append(pos_items[pos_cursor])
            pos_cursor += 1
        remaining = batch_size - len(batch)
        if remaining > 0:
            take = min(remaining, len(neg_items) - neg_cursor)
            if take > 0:
                batch.extend(neg_items[neg_cursor : neg_cursor + take])
                neg_cursor += take
        remaining = batch_size - len(batch)
        if remaining > 0:
            take = min(remaining, len(pos_items) - pos_cursor)
            if take > 0:
                batch.extend(pos_items[pos_cursor : pos_cursor + take])
                pos_cursor += take
        if batch:
            rng.shuffle(batch)
            yield batch


def sample_without_replacement(
    candidates: Sequence[T],
    n: int,
    rng: random.Random,
) -> List[T]:
    if n <= 0 or not candidates:
        return []
    if n >= len(candidates):
        return list(candidates)
    return rng.sample(list(candidates), n)


def save_checkpoint(
    model,
    tokenizer,
    metrics: Dict[str, Any],
    *,
    output_dir: str | Path,
    epoch: int | None = None,
) -> str:
    """Save LoRA adapter weights and metrics."""
    output_dir = Path(output_dir)
    if epoch is not None:
        output_dir = output_dir / f"checkpoint-epoch{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "wt", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return str(output_dir)


def evaluate_and_choose_tau(
    model,
    tokenizer,
    dev_examples: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device | None = None,
    encode_batch_fn=None,
    encode_kwargs: Dict[str, Any] | None = None,
    thresholds: torch.Tensor | None = None,
    include_confusion: bool = False,
):
    """Evaluate on dev set and choose the best decision threshold."""
    if encode_batch_fn is None:
        raise ValueError("encode_batch_fn is required")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if thresholds is None:
        thresholds = torch.linspace(0.1, 0.9, steps=81)

    encode_kwargs = encode_kwargs or {}

    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in make_minibatches(dev_examples, batch_size=batch_size, shuffle=False):
            tokens, labels = encode_batch_fn(tokenizer, batch, **encode_kwargs)
            tokens = move_to_device(tokens, device)
            logits = model(**tokens).logits.squeeze(-1)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_logits:
        return {"dev_f1": 0.0, "tau": 0.5, "dev_examples": 0}

    probs = torch.sigmoid(torch.cat(all_logits, dim=0))
    gold = torch.cat(all_labels, dim=0)

    best_tau = 0.5
    best_score = -1.0
    for tau in thresholds:
        preds = probs >= tau
        score = binary_f1(preds, gold)
        if score > best_score:
            best_score = score
            best_tau = float(tau.item())

    metrics = {
        "dev_f1": float(best_score),
        "tau": best_tau,
        "dev_examples": int(gold.numel()),
    }

    if include_confusion:
        preds = probs >= best_tau
        tp = int(((preds == 1) & (gold == 1)).sum().item())
        fp = int(((preds == 1) & (gold == 0)).sum().item())
        tn = int(((preds == 0) & (gold == 0)).sum().item())
        fn = int(((preds == 0) & (gold == 1)).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics.update(
            {
                "dev_precision": float(precision),
                "dev_recall": float(recall),
                "dev_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
        )

    return metrics
