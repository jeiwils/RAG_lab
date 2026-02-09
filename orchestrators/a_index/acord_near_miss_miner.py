"""Orchestrator for auto-generating ACORD near-miss pairs."""

from __future__ import annotations

from pathlib import Path

from src.ontology.acord_near_miss_miner import write_near_miss_yaml
from src.utils.__utils__ import raw_dataset_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = raw_dataset_dir("ACORD")

# Output file (keep auto output separate from the hand-curated near_miss.yaml)
OUTPUT_PATH = RAW_DIR / "near_miss_auto.yaml"

MIN_SCORE = 0.35
MAX_CANDIDATES_PER_QUERY = 6

INCLUDE_CROSS_CATEGORY = True
CROSS_CATEGORY_MIN_SCORE = 0.55
CROSS_CATEGORY_MAX_PER_QUERY = 2

USE_LLM = True
# Keep in sync with orchestrators/DA_EXIT.py.
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_SERVER_URL = "http://localhost:8005"
LLM_MAX_KEEP = 3

SEED = 1


def main() -> None:
    report = write_near_miss_yaml(
        output_path=OUTPUT_PATH,
        raw_dir=RAW_DIR,
        min_score=MIN_SCORE,
        max_candidates_per_query=MAX_CANDIDATES_PER_QUERY,
        include_cross_category=INCLUDE_CROSS_CATEGORY,
        cross_category_min_score=CROSS_CATEGORY_MIN_SCORE,
        cross_category_max_per_query=CROSS_CATEGORY_MAX_PER_QUERY,
        use_llm=USE_LLM,
        llm_model_name=LLM_MODEL_NAME,
        llm_server_url=LLM_SERVER_URL,
        llm_max_keep=LLM_MAX_KEEP,
        seed=SEED,
    )
    print(
        f"[near_miss_auto] pairs={len(report.get('pairs', []))} "
        f"output={OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
