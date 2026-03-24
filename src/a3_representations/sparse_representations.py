"""Keyword extraction utilities for sparse representations."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Set

import spacy
from tqdm import tqdm

from src.utils.__utils__ import clean_text

### SPACY
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

# OPTIMIZED: Excluded unused pipes (including lemmatizer) to save RAM and initialization time
nlp = spacy.load(SPACY_MODEL, exclude=["parser", "textcat", "lemmatizer"])
logger = logging.getLogger(__name__)

### NORMALIZATION
ALIAS = {
    # United States variants
    "u": "united_states",
    "us": "united_states",
    "u_s": "united_states",
    "u_s_a": "united_states",
    "united_states_of_america": "united_states",
    "the_united_states": "united_states",
    "u_s_navy": "united_states_navy",
    "u_s_air_force": "united_states_air_force",
    "u_s_army": "united_states_army",
    "u_s_marine_corps": "united_states_marine_corps",
    # United Kingdom / UN
    "uk": "united_kingdom",
    "u_k": "united_kingdom",
    "great_britain": "united_kingdom",
    "un": "united_nations",
    "u_n": "united_nations",
    # Map junk to None (drop)
    "what_year": None,
    "the_years": None,
    "year": None,
}

_NOISE_PATTERNS = [
    re.compile(r"^\d{4}$"),  # bare years
    re.compile(r"^\d+$"),  # numbers
    re.compile(
        r"^(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)$"
    ),
]


def canonicalize_keyword(kw: str) -> str | None:
    # alias (single step)
    kw = ALIAS.get(kw, kw)
    if kw is None:
        return None
    # noise drop
    for pat in _NOISE_PATTERNS:
        if pat.match(kw):
            return None
    return kw


def strip_accents(t: str) -> str:
    """Transliterate ``t`` to ASCII by stripping accents and special letters."""

    t = unicodedata.normalize("NFKD", t)
    replacements = {"\u00c3\u0178": "ss", "\u00c3\u00a6": "ae", "\u00c5\u201c": "oe"}
    for src, tgt in replacements.items():
        t = t.replace(src, tgt)
    return t.encode("ascii", "ignore").decode("ascii")


def normalise_text(s: str) -> str:
    """Return a normalised keyword string suitable for comparisons."""

    if not s:
        return ""
    t = clean_text(s).lower()
    t = t.replace("\u00e2\u20ac\u2122", "'")  # unify curly quote
    t = strip_accents(t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\W+", "_", t.strip())
    t = re.sub(r"_s_", "_", t)  # drop possessive
    t = re.sub(r"_s$", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


### ENTITY FILTERS
# Keep only these named-entity types
KEEP_ENTS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "EVENT",
    "LAW",  # drop: DATE, TIME, CARDINAL, ORDINAL, LANGUAGE, NORP
}


def extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    doc = nlp(text)
    out = set()
    for ent in doc.ents:
        if ent.label_ in KEEP_ENTS and ent.text.strip():
            norm = normalise_text(ent.text)
            if norm:
                canon = canonicalize_keyword(norm)
                if canon:
                    out.add(canon)
    return sorted(out)


def add_keywords_to_passages_jsonl(
    passages_jsonl: str,
    only_ids: Set[str] | None = None,
    batch_size: int = 50,  # Lowered from 1000 for better multiprocess stability
):
    """Add keywords to passages in a JSONL file.

    Parameters
    ----------
    passages_jsonl: str
        Path to the JSONL file containing passages.
    only_ids: set[str], optional
        Set of passage IDs to process. If None, all passages are processed.
    batch_size: int, optional
        Batch size for spaCy's nlp.pipe.
    """
    try:
        with open(passages_jsonl, "rt", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
    except Exception as e:
        logger.error(f"Failed to read {passages_jsonl}: {e}")
        raise

    if only_ids:
        targets = [r for r in rows if r.get("passage_id") in only_ids]
    else:
        targets = rows

    if not targets:
        logger.info("No targets to process.")
        return

    texts = [r.get("text", "") for r in targets]
    
    # OPTIMIZED: Initialize the pipe with 24 processes and stream it
    generator = nlp.pipe(texts, batch_size=batch_size, n_process=2)

    # OPTIMIZED: Wrap the zipping of targets and the generator in tqdm
    for r, doc in tqdm(zip(targets, generator), total=len(texts), desc="Extracting Keywords (24 Cores)"):
        kws = set()
        for ent in doc.ents:
            if ent.label_ in KEEP_ENTS and ent.text.strip():
                norm = normalise_text(ent.text)
                if norm:
                    canon = canonicalize_keyword(norm)
                    if canon:
                        kws.add(canon)
        r["keywords_passage"] = sorted(kws)

    try:
        with open(passages_jsonl, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to {passages_jsonl}: {e}")
        raise