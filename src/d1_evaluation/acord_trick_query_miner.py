"""Generate adversarial ("trick") queries from ACORD confusion reports.

This module mines the top near-miss confusions and uses an LLM (or a fallback
template) to draft trick queries that stress the retriever's weaknesses.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.a1_ingestion.dataset_preprocessing_functions import load_acord_queries
from src.utils.__utils__ import raw_dataset_dir, save_jsonl
from src.utils.z_llm_utils import (
    build_prompt,
    is_r1_like,
    query_llm,
    question_list_grammar,
    strip_think,
)

__all__ = [
    "mine_trick_queries",
    "write_trick_queries",
]


def _load_confusion_report(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing confusion report: {path}")
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _select_pairs(report: Dict[str, Any], n_pairs: int) -> List[Dict[str, Any]]:
    pairs = list(report.get("pair_confusion", []))
    if not pairs:
        raise ValueError("confusion_report has no pair_confusion entries.")
    pairs.sort(key=lambda x: x.get("hits", 0), reverse=True)
    return pairs[:n_pairs] if n_pairs > 0 else pairs


_TRICK_LABEL_RE = re.compile(
    r"^\s*[\(\[]?\s*([AB])\s*[\)\]]?\s*[:\-\|]\s*(.+)$",
    flags=re.I,
)


def _parse_labelled_trick(text: str) -> Dict[str, str] | None:
    text = str(text).strip()
    if not text:
        return None
    match = _TRICK_LABEL_RE.match(text)
    if not match:
        return None
    target = match.group(1).upper()
    body = match.group(2).strip()
    if not body:
        return None
    return {"text": body, "target": target}


def _fallback_trick_queries(a: str, b: str, n: int) -> List[Dict[str, str]]:
    templates = [
        ("A", "Clause that addresses {a} but explicitly not {b}."),
        ("B", "Clause that addresses {b} but explicitly not {a}."),
        ("A", "Provision focused on {a}, excluding any {b} language."),
        ("B", "Provision focused on {b}, excluding any {a} language."),
        ("A", "Only {a}; do not include {b}."),
        ("B", "Only {b}; do not include {a}."),
        ("A", "Clause about {a} (not {b})."),
        ("B", "Clause about {b} (not {a})."),
    ]
    out: List[Dict[str, str]] = []
    for i in range(n):
        target, tmpl = templates[i % len(templates)]
        out.append({"text": tmpl.format(a=a, b=b), "target": target})
    return out


def _generate_trick_queries_llm(
    *,
    query_a: str,
    query_b: str,
    n_queries: int,
    model_name: str,
    server_url: str,
    seed: int | None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    system = (
        "You are a legal drafting assistant. Create adversarial query labels that "
        "stress-test a clause retriever by exploiting confusions between two clause types."
    )
    user = (
        f"Query A: {query_a}\n"
        f"Query B: {query_b}\n\n"
        f"Generate {n_queries} short, distinct 'trick' queries that would be easy to "
        "confuse between A and B but still specify A or B clearly. "
        "Prefix each query with \"A:\" or \"B:\" to indicate the intended target. "
        "Return only JSON with the key \"Question List\"."
    )
    prompt = build_prompt(model_name, system, user)
    grammar = question_list_grammar(n_queries, n_queries)
    raw, usage = query_llm(
        prompt,
        server_url=server_url,
        max_tokens=128,
        temperature=0.4,
        grammar=grammar,
        model_name=model_name,
        phase="answer_generation",
        seed=seed,
    )
    if is_r1_like(model_name):
        raw = strip_think(raw)
    try:
        payload = json.loads(raw)
        items = payload.get("Question List", [])
        if not isinstance(items, list):
            raise ValueError("Question List is not a list")
        cleaned = [str(x).strip() for x in items if str(x).strip()]
        labelled: List[Dict[str, str]] = []
        for item in cleaned:
            parsed = _parse_labelled_trick(item)
            if parsed:
                labelled.append(parsed)
        if len(labelled) < n_queries:
            labelled.extend(
                _fallback_trick_queries(query_a, query_b, n_queries - len(labelled))
            )
        return labelled[:n_queries], usage
    except Exception:
        return _fallback_trick_queries(query_a, query_b, n_queries), usage


def mine_trick_queries(
    *,
    confusion_report_path: str | Path,
    raw_dir: str | Path = raw_dataset_dir("ACORD"),
    n_pairs: int = 10,
    n_queries_per_pair: int = 3,
    model_name: str,
    server_url: str,
    seed: int | None = 1,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Return mined trick queries and a failure report payload."""
    report = _load_confusion_report(confusion_report_path)
    pairs = _select_pairs(report, n_pairs)
    queries = load_acord_queries(Path(raw_dir) / "queries.jsonl")

    rng = random.Random(seed) if seed is not None else None
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    outputs: List[Dict[str, Any]] = []
    for item in pairs:
        qid = item.get("query_id", "")
        nm_id = item.get("near_miss_id", "")
        query_a = queries.get(qid, {}).get("text", qid)
        query_b = queries.get(nm_id, {}).get("text", nm_id)

        if use_llm:
            tricks, usage = _generate_trick_queries_llm(
                query_a=query_a,
                query_b=query_b,
                n_queries=n_queries_per_pair,
                model_name=model_name,
                server_url=server_url,
                seed=seed,
            )
            for k in total_usage:
                total_usage[k] += int(usage.get(k, 0))
        else:
            tricks = _fallback_trick_queries(query_a, query_b, n_queries_per_pair)
            if rng:
                rng.shuffle(tricks)
        enriched: List[Dict[str, Any]] = []
        for trick in tricks:
            if isinstance(trick, dict):
                text = str(trick.get("text", "")).strip()
                target = str(trick.get("target", "")).strip().upper()
            else:
                parsed = _parse_labelled_trick(str(trick))
                if not parsed:
                    continue
                text = parsed["text"]
                target = parsed["target"]
            if not text or target not in {"A", "B"}:
                continue
            target_query_id = qid if target == "A" else nm_id
            target_query_text = query_a if target == "A" else query_b
            enriched.append(
                {
                    "text": text,
                    "target": target,
                    "target_query_id": target_query_id,
                    "target_query_text": target_query_text,
                }
            )

        outputs.append(
            {
                "query_id": qid,
                "near_miss_id": nm_id,
                "query_text": query_a,
                "near_miss_text": query_b,
                "hits": int(item.get("hits", 0)),
                "hit_rate": float(item.get("hit_rate", 0.0)),
                "trick_queries": enriched,
            }
        )

    failure_report = {
        "meta": {
            "confusion_report": str(confusion_report_path),
            "model_name": model_name,
            "server_url": server_url,
            "n_pairs": n_pairs,
            "n_queries_per_pair": n_queries_per_pair,
            "use_llm": use_llm,
            "seed": seed,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "usage": total_usage,
        "failure_modes": outputs,
    }

    return {
        "trick_queries": outputs,
        "failure_report": failure_report,
    }


def write_trick_queries(
    *,
    confusion_report_path: str | Path,
    output_dir: str | Path,
    raw_dir: str | Path = raw_dataset_dir("ACORD"),
    n_pairs: int = 10,
    n_queries_per_pair: int = 3,
    model_name: str,
    server_url: str,
    seed: int | None = 1,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Write trick query artifacts and return the failure report."""
    payload = mine_trick_queries(
        confusion_report_path=confusion_report_path,
        raw_dir=raw_dir,
        n_pairs=n_pairs,
        n_queries_per_pair=n_queries_per_pair,
        model_name=model_name,
        server_url=server_url,
        seed=seed,
        use_llm=use_llm,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trick_path = output_dir / "trick_queries.jsonl"
    report_path = output_dir / "assurance_report.json"
    md_path = output_dir / "AI_Assurance_Report.md"

    save_jsonl(str(trick_path), payload["trick_queries"])
    with open(report_path, "wt", encoding="utf-8") as f:
        json.dump(payload["failure_report"], f, indent=2)

    _write_markdown_report(md_path, payload["failure_report"])
    return payload["failure_report"]


def _write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# AI Assurance Report (Failure Discovery)")
    lines.append("")
    lines.append("This report lists top confusion pairs and generated trick queries.")
    lines.append("")
    meta = report.get("meta", {})
    lines.append(f"- Source confusion report: `{meta.get('confusion_report', '')}`")
    lines.append(f"- Model: `{meta.get('model_name', '')}`")
    lines.append(f"- Timestamp: `{meta.get('timestamp', '')}`")
    lines.append("")

    for idx, item in enumerate(report.get("failure_modes", []), start=1):
        lines.append(f"## Failure Mode {idx}")
        lines.append(f"- Query A: `{item.get('query_text', '')}`")
        lines.append(f"- Query B: `{item.get('near_miss_text', '')}`")
        lines.append(
            f"- Confused hits: `{item.get('hits', 0)}` "
            f"(rate {item.get('hit_rate', 0.0):.3f})"
        )
        lines.append("- Trick queries:")
        for q in item.get("trick_queries", []):
            if isinstance(q, dict):
                label = q.get("target", "?")
                text = q.get("text", "")
                lines.append(f"  - [{label}] {text}")
            else:
                lines.append(f"  - {q}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
