"""Orchestrator for mining ACORD trick queries from confusion reports."""

from __future__ import annotations

from pathlib import Path

from src.d1_evaluation.acord_trick_query_miner import write_trick_queries

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFUSION_REPORT = Path(
    "data/results/acord_retriever/dev/confusion_report_lora.json"
)
OUTPUT_DIR = Path("data/results/acord_retriever/dev/assurance")

N_PAIRS = 10
N_QUERIES_PER_PAIR = 3

# Keep in sync with orchestrators/DA_EXIT.py.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SERVER_URL = "http://localhost:8005"

USE_LLM = True
SEED = 1


def main() -> None:
    write_trick_queries(
        confusion_report_path=CONFUSION_REPORT,
        output_dir=OUTPUT_DIR,
        n_pairs=N_PAIRS,
        n_queries_per_pair=N_QUERIES_PER_PAIR,
        model_name=MODEL_NAME,
        server_url=SERVER_URL,
        seed=SEED,
        use_llm=USE_LLM,
    )


if __name__ == "__main__":
    main()
