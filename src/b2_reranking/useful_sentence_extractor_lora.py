"""LoRA training utilities for sentence usefulness classification."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from tqdm import tqdm

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.utils.dataset_utils import build_training_examples, load_dataset_split
from src.utils.__utils__ import processed_dataset_paths
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
from src.utils.supervised_learning_utils import grid_search
from src.tests.plot_confusion_matrix import plot_confusion





# Script-level defaults when running this file directly.
DEFAULT_DATASET = "hotpotqa"
DEFAULT_TRAIN_SPLIT = "train_sub"
DEFAULT_DEV_SPLIT = "val"

# Toggle to train on discourse-aware passages (passages_discourse_aware.jsonl).
USE_DISCOURSE_AWARE_PASSAGES = False


USE_STRATIFIED_BATCHING = False
DEFAULT_SEED = 1
RUN_GRID_SEARCH = False



#### BASE MODEL SETTINGS

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

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"
DEFAULT_OUTPUT_DIR = "data/models/useful_sentence_lora"
DEFAULT_OUTPUT_DIR_DISCOURSE_AWARE = "data/models/useful_sentence_lora_discourse_aware"
DEFAULT_MAX_LEN = 256
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_WD = 0.0
DEFAULT_WARMUP = 200
DEFAULT_MAX_GRAD_NORM = 1.0



### LORA SETTINGS


DEFAULT_HARD_NEGATIVES = 8
DEFAULT_RANDOM_NEGATIVES = 1
DEFAULT_POS_WEIGHT = None
LOCAL_FILES_ONLY = True

DEFAULT_LR = 2e-5

DEFAULT_NUM_EPOCHS = 8

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05

DEFAULT_BATCH_SIZE = 16 





#### GRID SEARCH SETTINGS


BEST_GRID_RUN_KWARGS = {
    "hard_negatives": 8,
    "random_negatives": 2,
    "pos_weight": 1.0,
    "lr": 3e-5,
    "num_epochs": 2,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "batch_size": 16,
    "weight_decay": 0.01,
}

GRID_CONFIGS = [
    {
        "name": "transfer_balanced",
        "hard_negatives": 6,
        "random_negatives": 2,
        "pos_weight": 1.0,
        "lr": 2e-5,
        "num_epochs": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 16,
        "weight_decay": 0.05,
    },
    {
        "name": "transfer_precision",
        "hard_negatives": 6,
        "random_negatives": 2,
        "pos_weight": 0.5,
        "lr": 2e-5,
        "num_epochs": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 16,
        "weight_decay": 0.05,
    },
    {
        "name": "transfer_underfit",
        "hard_negatives": 8,
        "random_negatives": 2,
        "pos_weight": 1.0,
        "lr": 3e-5,
        "num_epochs": 2,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 16,
        "weight_decay": 0.01,
    },
]





__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TEXT_TEMPLATE",
    "build_training_examples",
    "count_tokens",
    "encode_batch",
    "evaluate_and_choose_tau",
    "find_attention_projection_modules",
    "init_model_with_lora",
    "load_dataset_split",
    "select_ranked_sentences",
    "train",
]







_TOKEN_RE = re.compile(r"\S+")



########


def _split_exists(dataset: str, split: str) -> bool:
    paths = processed_dataset_paths(dataset, split)
    return Path(paths["questions"]).exists() and Path(paths["passages"]).exists()





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


def _run_explicit_grid(
    train_fn: Callable[..., Dict[str, Any]],
    *,
    output_dir: str | Path,
    grid_configs: Sequence[Dict[str, Any]],
    train_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run explicit (non-cartesian) grid configurations."""
    train_kwargs = dict(train_kwargs or {})
    train_kwargs.pop("output_dir", None)

    best: Dict[str, Any] = {"dev_f1": -1.0}
    for run_idx, cfg in enumerate(grid_configs, start=1):
        params = dict(cfg)
        name = params.pop("name", None)
        suffix = (
            f"_hn{params.get('hard_negatives')}"
            f"_rn{params.get('random_negatives')}"
            f"_pw{params.get('pos_weight')}"
            f"_lr{params.get('lr')}"
            f"_ep{params.get('num_epochs')}"
            f"_r{params.get('lora_r')}"
            f"_a{params.get('lora_alpha')}"
            f"_d{params.get('lora_dropout')}"
            f"_bs{params.get('batch_size')}"
            f"_wd{params.get('weight_decay', DEFAULT_WD)}"
        )
        if name:
            suffix = f"_{name}{suffix}"
        run_dir = f"{output_dir}/grid/run{run_idx}{suffix}"

        metrics = train_fn(
            output_dir=run_dir,
            **train_kwargs,
            **params,
        )
        metrics["run_dir"] = run_dir
        metrics["run_idx"] = run_idx
        if metrics.get("dev_f1", -1.0) > best.get("dev_f1", -1.0):
            best = metrics
        print(
            f"[grid {run_idx}] dev_f1={metrics['dev_f1']:.4f} "
            f"tau={metrics['tau']:.3f} dir={run_dir}"
        )

    print(
        f"[grid best] dev_f1={best['dev_f1']:.4f} "
        f"tau={best['tau']:.3f} dir={best.get('run_dir')}"
    )
    return best






def train(
    *,
    base_model_name: str = DEFAULT_BASE_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset: str = "hotpotqa",
    train_split: str = "train",
    dev_split: str = "dev",
    train_path: str | Path | None = None,
    dev_path: str | Path | None = None,
    train_passages_path: str | Path | None = None,
    dev_passages_path: str | Path | None = None,
    question_id_col: str = "question_id",
    question_col: str = "question",
    gold_passages_col: str = "gold_passages",
    passage_id_col: str = "passage_id",
    passage_title_col: str = "title",
    passage_text_col: str = "text",
    passage_question_id_col: str | None = None,
    passage_id_split_token: str | None = "__",
    max_train_records: int | None = None,
    max_dev_records: int | None = None,
    hard_negatives: int = DEFAULT_HARD_NEGATIVES,
    random_negatives: int = DEFAULT_RANDOM_NEGATIVES,
    pos_weight: float | None = DEFAULT_POS_WEIGHT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LEN,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    lr: float = DEFAULT_LR,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    weight_decay: float = DEFAULT_WD,
    warmup_steps: int = DEFAULT_WARMUP,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    seed: int | None = DEFAULT_SEED,
    local_files_only: bool = LOCAL_FILES_ONLY,
    device: str | torch.device | None = None,
):
    """Train a LoRA sentence usefulness classifier.

    ``train_path``/``dev_path`` refer to processed question JSONL files.
    Use ``train_passages_path``/``dev_passages_path`` to override passages.
    """
    get_scheduler = get_linear_schedule_with_warmup

    if seed is not None:
        _seed_everything(seed)

    if not _split_exists(dataset, train_split):
        raise FileNotFoundError(
            f"Missing processed split for training: {dataset}/{train_split}"
        )
    if not _split_exists(dataset, dev_split):
        raise FileNotFoundError(
            f"Missing processed split for eval: {dataset}/{dev_split}"
        )

    train_data = load_dataset_split(
        train_split,
        dataset=dataset,
        questions_path=train_path,
        passages_path=train_passages_path,
        max_records=max_train_records,
        question_id_col=question_id_col,
        question_col=question_col,
        gold_passages_col=gold_passages_col,
        passage_id_col=passage_id_col,
        passage_title_col=passage_title_col,
        passage_text_col=passage_text_col,
        passage_question_id_col=passage_question_id_col,
        passage_id_split_token=passage_id_split_token,
    )
    dev_data = load_dataset_split(
        dev_split,
        dataset=dataset,
        questions_path=dev_path,
        passages_path=dev_passages_path,
        max_records=max_dev_records,
        question_id_col=question_id_col,
        question_col=question_col,
        gold_passages_col=gold_passages_col,
        passage_id_col=passage_id_col,
        passage_title_col=passage_title_col,
        passage_text_col=passage_text_col,
        passage_question_id_col=passage_question_id_col,
        passage_id_split_token=passage_id_split_token,
    )

    train_examples = build_training_examples(
        train_data,
        hard_negatives=hard_negatives,
        random_negatives=random_negatives,
        seed=seed,
        dataset=dataset,
        split=train_split,
    )
    dev_examples = build_training_examples(
        dev_data,
        hard_negatives=hard_negatives,
        random_negatives=random_negatives,
        seed=seed,
        dataset=dataset,
        split=dev_split,
    )

    if pos_weight is None:
        pos = sum(1 for ex in train_examples if int(ex.get("label", 0)) == 1)
        neg = max(len(train_examples) - pos, 0)
        pos_weight = (neg / max(pos, 1)) if train_examples else 1.0

    # Fast tokenizers trigger a hub lookup in some versions when offline.
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

    steps_per_epoch = math.ceil(len(train_examples) / batch_size) if train_examples else 0
    total_steps = steps_per_epoch * max(num_epochs, 1)
    scheduler = get_scheduler(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_metrics: Dict[str, Any] = {
        "dev_f1": -1.0, # why is this -1?
        "dev_loss": float("inf"),
        "epoch": 0,
        "tau": 0.5, # why is this .5? 
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in tqdm(
            make_minibatches(
                train_examples,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                stratified=USE_STRATIFIED_BATCHING,
            ),
            total=steps_per_epoch or None,
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
    try:
        plot_confusion(best_path, None)
    except ModuleNotFoundError as exc:
        if str(exc).startswith("No module named 'matplotlib'"):
            print("[warn] matplotlib not installed; skipping confusion plot")
        else:
            raise
    print(
        f"[best] epoch={best_metrics['epoch']} dev_loss={best_metrics['dev_loss']:.4f} "
        f"dev_f1={best_metrics['dev_f1']:.4f} tau={best_metrics['tau']:.3f}"
    )
    return best_metrics


if __name__ == "__main__":
    output_dir = (
        DEFAULT_OUTPUT_DIR_DISCOURSE_AWARE
        if USE_DISCOURSE_AWARE_PASSAGES
        else DEFAULT_OUTPUT_DIR
    )
    train_kwargs: Dict[str, Any] = {
        "dataset": DEFAULT_DATASET,
        "train_split": DEFAULT_TRAIN_SPLIT,
        "dev_split": DEFAULT_DEV_SPLIT,
    }
    if USE_DISCOURSE_AWARE_PASSAGES:
        train_kwargs.update(
            {
                "train_passages_path": (
                    f"data/processed_datasets/{DEFAULT_DATASET}/"
                    f"{DEFAULT_TRAIN_SPLIT}/passages_discourse_aware.jsonl"
                ),
                "dev_passages_path": (
                    f"data/processed_datasets/{DEFAULT_DATASET}/"
                    f"{DEFAULT_DEV_SPLIT}/passages_discourse_aware.jsonl"
                ),
            }
        )
    if RUN_GRID_SEARCH:
        _run_explicit_grid(
            train,
            output_dir=output_dir,
            train_kwargs=train_kwargs,
            grid_configs=GRID_CONFIGS,
        )
    else:
        train(output_dir=output_dir, **train_kwargs, **BEST_GRID_RUN_KWARGS)
