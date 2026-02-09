"""LoRA training utilities for the ACORD near-miss-graph-aware retriever."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm import tqdm

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.a1_ingestion.dataset_preprocessing_functions import (
    ACORD_DEFAULT_HARD_THRESHOLD,
    ACORD_DEFAULT_NEG_THRESHOLD,
    ACORD_DEFAULT_POS_THRESHOLD,
)
from src.b2_reranking.acord_retriever_dataset import (
    DEFAULT_ACORD_RAW_DIR,
    build_acord_retriever_examples,
    load_acord_retriever_examples,
    split_examples_by_difficulty,
)
from src.b2_reranking.acord_retriever_infer import (
    DEFAULT_MAX_LEN,
    LOCAL_FILES_ONLY,
    encode_batch,
)
from src.utils.lora_helpers import (
    evaluate_and_choose_tau,
    find_attention_projection_modules,
    init_model_with_lora,
    make_minibatches,
    move_to_device as _move_to_device,
    save_checkpoint,
    seed_everything as _seed_everything,
    trainable_parameters as _trainable_parameters,
)
from src.utils.__utils__ import processed_dataset_paths

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"
DEFAULT_OUTPUT_DIR = "data/models/acord_retriever_lora"
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_WD = 0.0
DEFAULT_WARMUP = 200
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TRAIN_EXAMPLES_PATH = (
    processed_dataset_paths("acord", "train")["base"] / "retriever_examples.jsonl"
)
DEFAULT_DEV_EXAMPLES_PATH = (
    processed_dataset_paths("acord", "dev")["base"] / "retriever_examples.jsonl"
)

DEFAULT_SEED = 1

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "build_acord_retriever_examples",
    "encode_batch",
    "evaluate_and_choose_tau",
    "find_attention_projection_modules",
    "init_model_with_lora",
    "load_acord_retriever_examples",
    "train",
]


def _prepare_curriculum_sets(
    examples: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    pos, easy_neg, hard_neg = split_examples_by_difficulty(examples)
    return {
        "pos": pos,
        "easy": pos + easy_neg,
        "hard": pos + hard_neg,
        "mixed": pos + easy_neg + hard_neg,
    }


def _select_epoch_examples(
    curriculum_sets: Dict[str, List[Dict[str, Any]]],
    *,
    epoch: int,
    hard_start_epoch: int | None,
    mode: str,
) -> List[Dict[str, Any]]:
    if hard_start_epoch is None:
        return curriculum_sets["mixed"]
    if epoch < hard_start_epoch:
        return curriculum_sets["easy"]
    if mode == "mix":
        return curriculum_sets["mixed"]
    return curriculum_sets["hard"]


def _pos_weight_from_examples(examples: Sequence[Dict[str, Any]]) -> float:
    pos = sum(1 for ex in examples if int(ex.get("label", 0)) == 1)
    neg = max(len(examples) - pos, 0)
    return (neg / max(pos, 1)) if examples else 1.0


def train(
    *,
    base_model_name: str = DEFAULT_BASE_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    raw_dir: str | Path = DEFAULT_ACORD_RAW_DIR,
    train_split: str = "train",
    dev_split: str = "dev",
    train_examples_path: str | Path | None = None,
    dev_examples_path: str | Path | None = None,
    near_miss_path: str | Path | None = None,
    pos_threshold: int = ACORD_DEFAULT_POS_THRESHOLD,
    neg_threshold: int = ACORD_DEFAULT_NEG_THRESHOLD,
    hard_threshold: int | None = ACORD_DEFAULT_HARD_THRESHOLD,
    max_pos: int | None = None,
    max_easy_neg: int | None = None,
    max_hard_neg: int | None = None,
    max_ontology_neg: int | None = None,
    include_siblings: bool = True,
    include_near_miss: bool = True,
    drop_no_positives: bool = True,
    curriculum_hard_start_epoch: int | None = None,
    curriculum_mode: str = "replace",
    pos_weight: float | None = None,
    batch_size: int | None = None,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LEN,
    num_epochs: int | None = None,
    lr: float | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    weight_decay: float = DEFAULT_WD,
    warmup_steps: int = DEFAULT_WARMUP,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    seed: int | None = DEFAULT_SEED,
    local_files_only: bool = LOCAL_FILES_ONLY,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Train a LoRA retriever classifier."""
    get_scheduler = get_linear_schedule_with_warmup

    if seed is not None:
        _seed_everything(seed)

    if batch_size is None:
        batch_size = 16
    if num_epochs is None:
        num_epochs = 4
    if lr is None:
        lr = 2e-5
    if lora_r is None:
        lora_r = 8
    if lora_alpha is None:
        lora_alpha = 16
    if lora_dropout is None:
        lora_dropout = 0.05

    if train_examples_path is not None:
        train_examples = load_acord_retriever_examples(train_examples_path)
    else:
        train_examples, _ = build_acord_retriever_examples(
            split=train_split,
            raw_dir=raw_dir,
            near_miss_path=near_miss_path,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            hard_threshold=hard_threshold,
            max_pos=max_pos,
            max_easy_neg=max_easy_neg,
            max_hard_neg=max_hard_neg,
            max_ontology_neg=max_ontology_neg,
            seed=seed,
            include_siblings=include_siblings,
            include_near_miss=include_near_miss,
            drop_no_positives=drop_no_positives,
            require_near_miss=include_near_miss,
        )

    if dev_examples_path is not None:
        dev_examples = load_acord_retriever_examples(dev_examples_path)
    else:
        dev_examples, _ = build_acord_retriever_examples(
            split=dev_split,
            raw_dir=raw_dir,
            near_miss_path=near_miss_path,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            hard_threshold=hard_threshold,
            max_pos=max_pos,
            max_easy_neg=max_easy_neg,
            max_hard_neg=max_hard_neg,
            max_ontology_neg=max_ontology_neg,
            seed=seed,
            include_siblings=include_siblings,
            include_near_miss=include_near_miss,
            drop_no_positives=drop_no_positives,
            require_near_miss=include_near_miss,
        )

    if not train_examples:
        raise ValueError("No training examples were built.")
    if not dev_examples:
        raise ValueError("No dev examples were built.")

    curriculum_sets = _prepare_curriculum_sets(train_examples)

    if pos_weight is None:
        pos_weight = _pos_weight_from_examples(curriculum_sets["mixed"])

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        local_files_only=local_files_only,
        use_fast=not local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = init_model_with_lora(
        base_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        local_files_only=local_files_only,
    )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.to(device)

    optimiser = torch.optim.AdamW(
        params=_trainable_parameters(model),
        lr=lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_examples) / batch_size)
    total_steps = steps_per_epoch * max(num_epochs, 1)
    scheduler = get_scheduler(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_metrics: Dict[str, Any] = {
        "dev_f1": -1.0,
        "dev_loss": float("inf"),
        "epoch": 0,
        "tau": 0.5,
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        epoch_examples = _select_epoch_examples(
            curriculum_sets,
            epoch=epoch,
            hard_start_epoch=curriculum_hard_start_epoch,
            mode=curriculum_mode,
        )
        if not epoch_examples:
            raise ValueError(f"No examples available for epoch {epoch}.")

        for batch in tqdm(
            make_minibatches(
                epoch_examples,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            ),
            total=math.ceil(len(epoch_examples) / batch_size),
            desc=f"epoch {epoch}",
        ):
            tokens, labels = encode_batch(
                tokenizer,
                batch,
                max_length=max_length,
            )
            tokens = _move_to_device(tokens, device)
            labels = labels.to(device)

            logits = model(**tokens).logits.squeeze(-1)
            if not torch.isfinite(logits).all():
                print(f"[warn] non-finite logits at epoch {epoch} step {steps + 1}")
                optimiser.zero_grad()
                continue
            weight_tensor = torch.tensor([pos_weight], device=labels.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=weight_tensor
            )
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch} step {steps + 1}")
                optimiser.zero_grad()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                print(f"[warn] non-finite grad norm at epoch {epoch} step {steps + 1}")
                optimiser.zero_grad()
                continue
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            running_loss += float(loss.item())
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        metrics = evaluate_and_choose_tau(
            model,
            tokenizer,
            dev_examples,
            batch_size=eval_batch_size,
            device=device,
            encode_batch_fn=encode_batch,
            encode_kwargs={"max_length": max_length},
            pos_weight=pos_weight,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = avg_loss

        save_checkpoint(
            model,
            tokenizer,
            metrics,
            output_dir=output_dir,
            epoch=epoch,
        )

        print(
            f"[epoch {epoch}] train_loss={avg_loss:.4f} "
            f"dev_loss={metrics['dev_loss']:.4f} "
            f"dev_f1={metrics['dev_f1']:.4f} tau={metrics['tau']:.3f}"
        )

        if epoch == 1 or metrics["dev_loss"] < best_metrics["dev_loss"]:
            best_metrics = metrics
            best_with_confusion = evaluate_and_choose_tau(
                model,
                tokenizer,
                dev_examples,
                batch_size=eval_batch_size,
                device=device,
                encode_batch_fn=encode_batch,
                encode_kwargs={"max_length": max_length},
                pos_weight=pos_weight,
                include_confusion=True,
            )
            best_with_confusion["epoch"] = epoch
            best_with_confusion["train_loss"] = avg_loss
            best_metrics = best_with_confusion

    best_path = Path(output_dir) / "best_metrics.json"
    with open(best_path, "wt", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print(
        f"[best] epoch={best_metrics['epoch']} dev_loss={best_metrics['dev_loss']:.4f} "
        f"dev_f1={best_metrics['dev_f1']:.4f} tau={best_metrics['tau']:.3f}"
    )
    return best_metrics


if __name__ == "__main__":
    train(
        train_examples_path=DEFAULT_TRAIN_EXAMPLES_PATH,
        dev_examples_path=DEFAULT_DEV_EXAMPLES_PATH,
    )
