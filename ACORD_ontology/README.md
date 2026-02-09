# ACORD Near-Miss Graph-Aware Retriever

This project is a two-step process designed to harden a retriever against legal confusions by mining near-miss relationships and then attacking the model with adversarial probes.
Stage 1 mines a near-miss pair list (confusable clause labels) and trains a LoRA retriever on those hard negatives. 
Stage 2 benchmarks the retriever, inspects the confusion report for weaknesses, and uses adversarial AI to generate trick queries from those failures to further harden the model.

## What this adds on top of ACORD
- A `near_miss.yaml` near-miss pair list that links confusable query labels (manual or auto).
- Hard-negative sampling that uses those near-miss links (and category siblings).
- Evaluation with nDCG and Concept Confusion Rate (CCR) to measure near-miss errors.
- Failure discovery via trick queries and an iterative hardening loop.

## How the near-miss pairs are made
For each query, the mining script builds a candidate list using lexical similarity (weighted Jaccard + token overlap + sequence ratio). If LLM filtering is enabled, it asks the LLM to select a small set of confusable-but-distinct labels, returning strict JSON that lists the chosen IDs; otherwise it keeps the similarity-based candidates.
- Passed to the model (when enabled): the query text plus a numbered list of candidate IDs with their text and category. The list is capped to up to 6 items.
- Asked to (when enabled): choose up to 3 candidates that are easy to confuse with the query but still semantically distinct, and return JSON only: `{"near_miss_ids": ["..."]}`.
- Output used: the `near_miss_ids` list when LLM filtering is enabled; otherwise the similarity-based candidates. Either way, the selected IDs become near-miss pairs in the auto YAML.

## How trick queries are made
For each high-confusion pair, the trick-query generator asks the LLM to draft short adversarial queries that are ambiguous between the two labels but still indicate which label is intended. The output is parsed into labelled items and saved as `trick_queries.jsonl` (with a template fallback if parsing fails).
- Passed to the model: Query A text and Query B text, plus the desired number of trick queries.
- Asked to: produce distinct short queries that are easy to confuse between A and B but still specify one target, prefixing each with `A:` or `B:`, and return JSON only under `"Question List"`.
- Output used: the list of labelled trick queries, each mapped to a `target_query_id` for downstream training data.

## Workflow (end-to-end)
1. Create the near-miss pairs (choose one: auto-mine via `orchestrators/a_index/acord_near_miss_miner.py` -> `data/raw_datasets/ACORD/near_miss_auto.yaml`, OR manual authoring of `data/raw_datasets/ACORD/near_miss.yaml`).
2. If auto-mined, promote to the active file: replace or copy to `data/raw_datasets/ACORD/near_miss.yaml`.
3. Validate near-miss pairs: `orchestrators/a_index/acord_near_miss_validation.py`.
4. Build retriever examples (hard negatives): `orchestrators/b_retrieve/build_acord_retriever_examples.py`.
5. Train LoRA retriever: `src/b2_reranking/acord_retriever_train.py`.
6. Benchmark (baseline vs LoRA): `orchestrators/ACORD_near_miss_retriever.py` (produces confusion report). 
7. Iterate: use the confusion report to generate trick queries (`orchestrators/d_evaluate/acord_trick_query_miner.py`), rebuild examples with trick queries, retrain, then re-benchmark. Note: the trick-query script defaults to `data/results/acord_retriever/dev/confusion_report_lora.json`; update `CONFUSION_REPORT` for other runs.

## Repository entry points
- Near-miss validation report: `orchestrators/a_index/acord_near_miss_validation.py`
- Auto near-miss mining: `orchestrators/a_index/acord_near_miss_miner.py`
- Build retriever examples: `orchestrators/b_retrieve/build_acord_retriever_examples.py`
- Train LoRA retriever: `src/b2_reranking/acord_retriever_train.py`
- Evaluate / benchmark: `orchestrators/ACORD_near_miss_retriever.py` (baseline runs by default; LoRA eval requires `RUN_EVAL=True`)
- Failure discovery (trick queries): `orchestrators/d_evaluate/acord_trick_query_miner.py`

## Files you edit
- `data/raw_datasets/ACORD/near_miss.yaml` (manual near-miss pairs or promoted auto output)

## Expected outputs
- `data/processed_datasets/acord/ontology_summary.json`
- `data/processed_datasets/acord/*/retriever_examples*.jsonl`
- `data/models/acord_retriever_lora/` (LoRA checkpoints)
- `data/results/acord_retriever/*/confusion_report*.json`
- `data/results/acord_retriever/*/assurance/trick_queries.jsonl`
- `data/raw_datasets/ACORD/near_miss_auto.report.json` (auto-miner report)
- `data/results/acord_retriever/*/assurance/assurance_report.json`
- `data/results/acord_retriever/*/assurance/AI_Assurance_Report.md`
