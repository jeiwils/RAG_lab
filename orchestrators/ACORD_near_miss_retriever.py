"""ACORD retrieval benchmarking orchestrator.

This runner focuses on **evaluation** of a trained retriever checkpoint.
Near-miss graph validation is handled in a separate module:
`orchestrators/a_index/acord_near_miss_validation.py`.

Upstream note:
- Dataset preprocessing (questions/passages JSONL) is handled in
  `orchestrators/a_index/dataset_preprocessing.py`.
- Retriever example building lives in
  `orchestrators/b_retrieve/build_acord_retriever_examples.py`.
"""

from __future__ import annotations

from src.d1_evaluation.acord_retriever_eval import evaluate_acord_retriever
from src.ontology.acord_near_miss_graph import DEFAULT_NEAR_MISS_PATH
from src.utils.__utils__ import raw_dataset_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = raw_dataset_dir("ACORD")

# Thresholds (YOUR INTERVENTION: decide what counts as positive)
# This threshold is used for evaluation only (nDCG + CCR), not inference.
POS_THRESHOLD = 3

# Near-miss graph file (manual or promoted auto): `data/raw_datasets/ACORD/near_miss.yaml`.
# Use `orchestrators/a_index/acord_near_miss_validation.py` to write the summary report.

# Step 2: Retrieval evaluation (manual checkpoint selection)
RUN_EVAL = True
# Use "dev" (ACORD valid.tsv) for tuning; switch to "test" for final benchmarks.
EVAL_SPLIT = "dev"
EVAL_MODEL_CHECKPOINT = (
    "data/models/acord_retriever_lora/"
    "checkpoint-epoch1"
)
EVAL_TOP_K = 10
EVAL_BATCH_SIZE = 32
EVAL_MAX_LEN = 512

# Step 3: Baseline evaluation (no LoRA)
RUN_BASELINE_EVAL = True
BASELINE_MODE = "token_overlap"  # options: "token_overlap", "random"
BASELINE_SEED = 1
BASELINE_RUN_TAG = "baseline_token_overlap"

# Optional: run the same evaluation on the official test split.
RUN_TEST_EVAL = False
TEST_SPLIT = "test"




def main() -> None:
    """Run ACORD retrieval prep steps in order.

    Step 1 (optional): Evaluate a manually selected retriever checkpoint.
    Step 2 (optional): Evaluate a baseline (no LoRA) scorer.
    Step 3 (optional): Repeat evaluation on the test split.
    """
    def _run_eval_for_split(split: str, tag_suffix: str | None = None) -> None:
        suffix = f"_{tag_suffix}" if tag_suffix else ""
        if RUN_EVAL:
            if not EVAL_MODEL_CHECKPOINT:
                raise ValueError(
                    "EVAL_MODEL_CHECKPOINT must be set to a LoRA checkpoint directory."
                )
            evaluate_acord_retriever(
                model_checkpoint=EVAL_MODEL_CHECKPOINT,
                split=split,
                raw_dir=RAW_DIR,
                near_miss_path=DEFAULT_NEAR_MISS_PATH,
                pos_threshold=POS_THRESHOLD,
                top_k=EVAL_TOP_K,
                batch_size=EVAL_BATCH_SIZE,
                max_length=EVAL_MAX_LEN,
                scoring_mode="lora",
                run_tag=f"lora{suffix}",
            )

        if RUN_BASELINE_EVAL:
            evaluate_acord_retriever(
                model_checkpoint=None,
                split=split,
                raw_dir=RAW_DIR,
                near_miss_path=DEFAULT_NEAR_MISS_PATH,
                pos_threshold=POS_THRESHOLD,
                top_k=EVAL_TOP_K,
                batch_size=EVAL_BATCH_SIZE,
                max_length=EVAL_MAX_LEN,
                scoring_mode=BASELINE_MODE,
                run_tag=f"{BASELINE_RUN_TAG}{suffix}",
                baseline_seed=BASELINE_SEED,
            )

    _run_eval_for_split(EVAL_SPLIT)
    if RUN_TEST_EVAL:
        _run_eval_for_split(TEST_SPLIT, tag_suffix="test")


if __name__ == "__main__":
    main()
