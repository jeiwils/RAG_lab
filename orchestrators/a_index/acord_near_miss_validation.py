"""Validate ACORD near-miss graph inputs and write a summary report.

This writes `ontology_summary.json` to confirm near-miss IDs align with
the ACORD query catalog. It does **not** create the near-miss graph itself.
"""

from __future__ import annotations

from src.ontology.acord_near_miss_graph import (
    DEFAULT_NEAR_MISS_PATH,
    write_acord_ontology_summary,
)
from src.utils.__utils__ import processed_dataset_root, raw_dataset_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = raw_dataset_dir("ACORD")
NEAR_MISS_PATH = DEFAULT_NEAR_MISS_PATH
ONTOLOGY_SUMMARY_PATH = processed_dataset_root("acord") / "ontology_summary.json"


def main() -> None:
    write_acord_ontology_summary(
        raw_dir=RAW_DIR,
        near_miss_path=NEAR_MISS_PATH,
        out_path=ONTOLOGY_SUMMARY_PATH,
        require_near_miss=True,
    )


if __name__ == "__main__":
    main()
