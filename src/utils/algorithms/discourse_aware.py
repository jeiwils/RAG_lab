"""Regex-only discourse marker detection for anaphora, cataphora, and parentheticals.

Lists and patterns follow the Category A-G markers supplied in the prompt.
Heuristics are intentionally lightweight and rely only on regular expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern, Sequence, Tuple

###########################################################################
# Marker categories (from the prompt)
###########################################################################

CATEGORY_A_STRONG_BACKWARD_INITIAL = [
    "however",
    "nevertheless",
    "nonetheless",
    "still",
    "instead",
    "otherwise",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "moreover",
    "furthermore",
]

CATEGORY_B_BACKWARD_PARENTHESES = [
    "however",
    "yet",
    "whereas",
    "therefore",
    "thus",
]

CATEGORY_D_EXAMPLE_FORWARD = [
    "for example",
    "for instance",
    "such as",
    "namely",
    "specifically",
    "in particular",
]

CATEGORY_E_REFORMULATION = [
    "in other words",
    "i.e.",
    "put differently",
]

CATEGORY_F_TEMPORAL = [
    "then",
    "meanwhile",
    "afterwards",
    "previously",
    "subsequently",
]

CATEGORY_G_CONDITIONAL = [
    "otherwise",
    "in that case",
]

###########################################################################
# Combined lists and configuration
###########################################################################

ANAPHORA_MARKERS_STRONG = CATEGORY_A_STRONG_BACKWARD_INITIAL
ANAPHORA_MARKERS_WEAK = sorted(set(CATEGORY_B_BACKWARD_PARENTHESES + CATEGORY_G_CONDITIONAL))
CATAPHORA_MARKERS = CATEGORY_D_EXAMPLE_FORWARD
REFORMULATION_MARKERS = CATEGORY_E_REFORMULATION
TEMPORAL_MARKERS = CATEGORY_F_TEMPORAL
PARENTHETICAL_MARKERS = sorted(set(CATEGORY_B_BACKWARD_PARENTHESES))

SOFT_PREFIXES = ("and", "but", "so")

PHRASE_OVERRIDES = {
    "i.e.": r"i\.?\s*e\.?",
}

_EXCLUDED_ANAPHORA_PATTERNS = [
    re.compile(r"\beven\s+though\b", flags=re.I),
]

###########################################################################
# Regex helpers
###########################################################################


def _phrase_to_pattern(phrase: str) -> str:
    phrase = phrase.strip().lower()
    override = PHRASE_OVERRIDES.get(phrase)
    if override:
        return override
    escaped = re.escape(phrase)
    return escaped.replace(r"\ ", r"\s+")


def _marker_pattern(phrases: Sequence[str]) -> str:
    if not phrases:
        return r"(?!x)"
    parts = [_phrase_to_pattern(p) for p in phrases]
    body = "|".join(parts)
    return rf"(?<!\w)(?:{body})(?!\w)"


def _compile_phrase_regex(phrases: Sequence[str]) -> Pattern[str]:
    return re.compile(_marker_pattern(phrases), flags=re.I)


_SOFT_PREFIX_GROUP = ""
if SOFT_PREFIXES:
    _SOFT_PREFIX_RE = "|".join(re.escape(p) for p in SOFT_PREFIXES)
    _SOFT_PREFIX_GROUP = rf"(?:(?:{_SOFT_PREFIX_RE})\s*,?\s+)?"

SENTENCE_INITIAL_PREFIX = rf"^\s*(?:[\"'(\[]\s*)*{_SOFT_PREFIX_GROUP}"


def _compile_sentence_initial_re(phrases: Sequence[str]) -> Pattern[str]:
    marker = _marker_pattern(phrases)
    return re.compile(SENTENCE_INITIAL_PREFIX + rf"(?P<marker>{marker})", flags=re.I)

###########################################################################
# Sentence splitting & masking
###########################################################################

SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|\n+|$)")

_SENTENCE_DOT_MASK = "\x1f"
_TITLE_ABBREV_RE = re.compile(r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st)\.", flags=re.I)
_MULTI_DOT_ABBREV_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_INITIAL_RE = re.compile(r"\b[A-Z]\.(?=\s+[A-Z])")
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_COMMON_ABBREV_FOLLOW_RE = re.compile(
    r"\b(?:etc|vs|cf|al|eq|fig|figs|ref|refs|sec|secs|no|vol|pp|dept|inc|ltd|co|corp|univ)"
    r"\.(?=\s+(?:[a-z0-9]))",
    flags=re.I,
)
_MONTH_ABBREV_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.(?=\s+\d)",
    flags=re.I,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DOMAIN_RE = re.compile(
    r"\b[\w.-]+\.(?:com|org|net|edu|gov|io|ai|co|uk|us|de|fr|jp|cn|in|au|ca)\b",
    re.I,
)


def _mask_sentence_dots(text: str) -> str:
    """Mask dots that should not end sentences; preserves string length."""

    def _mask(match: re.Match) -> str:
        return match.group(0).replace(".", _SENTENCE_DOT_MASK)

    text = _TITLE_ABBREV_RE.sub(_mask, text)
    text = _MULTI_DOT_ABBREV_RE.sub(_mask, text)
    text = _INITIAL_RE.sub(_mask, text)
    text = _DECIMAL_RE.sub(_SENTENCE_DOT_MASK, text)
    text = _COMMON_ABBREV_FOLLOW_RE.sub(_mask, text)
    text = _MONTH_ABBREV_RE.sub(_mask, text)
    text = _EMAIL_RE.sub(_mask, text)
    text = _DOMAIN_RE.sub(_mask, text)
    return text

###########################################################################
# Compiled category regexes
###########################################################################

CATEGORY_A_INITIAL_RE = _compile_sentence_initial_re(CATEGORY_A_STRONG_BACKWARD_INITIAL)
CATEGORY_B_INLINE_RE = _compile_phrase_regex(CATEGORY_B_BACKWARD_PARENTHESES)
CATEGORY_D_INLINE_RE = _compile_phrase_regex(CATEGORY_D_EXAMPLE_FORWARD)
CATEGORY_D_INITIAL_RE = _compile_sentence_initial_re(CATEGORY_D_EXAMPLE_FORWARD)
CATEGORY_E_INLINE_RE = _compile_phrase_regex(CATEGORY_E_REFORMULATION)
CATEGORY_F_INLINE_RE = _compile_phrase_regex(CATEGORY_F_TEMPORAL)
CATEGORY_G_INLINE_RE = _compile_phrase_regex(CATEGORY_G_CONDITIONAL)

_PAREN_PATTERN = _marker_pattern(PARENTHETICAL_MARKERS)
PARENTHETICAL_COMMA_RE = re.compile(
    rf",\s*(?P<marker>{_PAREN_PATTERN})\s*,",
    flags=re.I,
)
PARENTHETICAL_PAREN_RE = re.compile(
    rf"\(\s*(?P<marker>{_PAREN_PATTERN})\s*\)",
    flags=re.I,
)

PARENTHESIS_RE = re.compile(r"\([^)]*\)")
FORWARD_FOLLOW_RE = re.compile(r"^\s*[:;]")

###########################################################################
# Data model
###########################################################################


@dataclass(frozen=True)
class MarkerHit:
    """Single discourse marker match."""

    marker: str
    start: int
    end: int
    category: str
    direction: str
    kind: str

###########################################################################
# Core utilities
###########################################################################


def iter_sentence_spans(text: str) -> Iterable[Tuple[int, int, str]]:
    """Yield (start, end, sentence_text) spans using a lightweight regex splitter."""
    masked = _mask_sentence_dots(text)
    for match in SENTENCE_RE.finditer(masked):
        sentence = text[match.start() : match.end()]
        if not sentence.strip():
            continue
        yield match.start(), match.end(), sentence


def _dedupe_by_span(hits: list[MarkerHit]) -> list[MarkerHit]:
    seen: set[Tuple[int, int]] = set()
    deduped: list[MarkerHit] = []
    for hit in hits:
        key = (hit.start, hit.end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def _filter_excluded_anaphora(text: str, hits: list[MarkerHit]) -> list[MarkerHit]:
    if not hits:
        return hits
    excluded_spans: list[Tuple[int, int]] = []
    for pattern in _EXCLUDED_ANAPHORA_PATTERNS:
        excluded_spans.extend((m.start(), m.end()) for m in pattern.finditer(text))
    if not excluded_spans:
        return hits
    filtered: list[MarkerHit] = []
    for hit in hits:
        if any(start <= hit.start and end >= hit.end for start, end in excluded_spans):
            continue
        filtered.append(hit)
    return filtered


def _second_comma_index(sentence: str) -> int | None:
    comma_count = 0
    for idx, ch in enumerate(sentence):
        if ch == ",":
            comma_count += 1
            if comma_count == 2:
                return idx
    return None


def _filter_hits_before_second_comma(text: str, hits: list[MarkerHit]) -> list[MarkerHit]:
    if not hits:
        return hits
    spans = list(iter_sentence_spans(text))
    if not spans:
        return hits
    span_meta: list[Tuple[int, int, int | None]] = []
    for start, end, sentence in spans:
        second_idx = _second_comma_index(sentence)
        second_abs = start + second_idx if second_idx is not None else None
        span_meta.append((start, end, second_abs))
    filtered: list[MarkerHit] = []
    for hit in hits:
        if hit.kind not in {"inline", "parenthetical"}:
            filtered.append(hit)
            continue
        for start, end, second_abs in span_meta:
            if start <= hit.start < end:
                if second_abs is None or hit.start < second_abs:
                    filtered.append(hit)
                break
        else:
            filtered.append(hit)
    return filtered


def _forward_context(text: str, end: int, *, window: int = 80) -> bool:
    return bool(FORWARD_FOLLOW_RE.search(text[end : end + window]))

###########################################################################
# Detection helpers
###########################################################################


def find_sentence_initial_markers(
    text: str,
    regex: Pattern[str],
    *,
    category: str,
    direction: str,
    kind: str = "sentence_initial",
) -> list[MarkerHit]:
    hits: list[MarkerHit] = []
    for sent_start, _, sentence in iter_sentence_spans(text):
        for match in regex.finditer(sentence):
            hits.append(
                MarkerHit(
                    marker=match.group("marker"),
                    start=sent_start + match.start("marker"),
                    end=sent_start + match.end("marker"),
                    category=category,
                    direction=direction,
                    kind=kind,
                )
            )
    return hits


def find_inline_markers(
    text: str,
    regex: Pattern[str],
    *,
    category: str,
    direction: str,
    kind: str = "inline",
) -> list[MarkerHit]:
    return [
        MarkerHit(
            marker=match.group(0),
            start=match.start(),
            end=match.end(),
            category=category,
            direction=direction,
            kind=kind,
        )
        for match in regex.finditer(text)
    ]


def find_parenthetical_markers(
    text: str,
    markers: Sequence[str] | None = None,
    *,
    category: str = "B",
    direction: str = "backward",
) -> list[MarkerHit]:
    """Detect markers used parenthetically via commas or parentheses."""
    if markers is None:
        comma_re = PARENTHETICAL_COMMA_RE
        paren_re = PARENTHETICAL_PAREN_RE
    else:
        pattern = _marker_pattern(markers)
        comma_re = re.compile(rf",\s*(?P<marker>{pattern})\s*,", flags=re.I)
        paren_re = re.compile(rf"\(\s*(?P<marker>{pattern})\s*\)", flags=re.I)

    hits: list[MarkerHit] = []
    for regex in (comma_re, paren_re):
        for match in regex.finditer(text):
            hits.append(
                MarkerHit(
                    marker=match.group("marker"),
                    start=match.start("marker"),
                    end=match.end("marker"),
                    category=category,
                    direction=direction,
                    kind="parenthetical",
                )
            )
    return hits


def find_parentheses(text: str) -> list[Tuple[int, int, str]]:
    """Return raw parenthesis spans, regardless of marker type."""
    return [
        (match.start(), match.end(), match.group(0))
        for match in PARENTHESIS_RE.finditer(text)
    ]

###########################################################################
# Public detection API
###########################################################################


def find_anaphora_markers(
    text: str,
    *,
    include_reformulation: bool = True,
    include_parenthetical: bool = True,
) -> list[MarkerHit]:
    """Find backward-dependent linking markers."""
    hits: list[MarkerHit] = []
    hits.extend(
        find_sentence_initial_markers(
            text,
            CATEGORY_A_INITIAL_RE,
            category="A",
            direction="backward",
        )
    )
    if include_parenthetical:
        hits.extend(find_parenthetical_markers(text))
    hits.extend(
        find_inline_markers(
            text,
            CATEGORY_B_INLINE_RE,
            category="B",
            direction="backward",
        )
    )
    hits.extend(
        find_inline_markers(
            text,
            CATEGORY_G_INLINE_RE,
            category="G",
            direction="backward",
        )
    )
    if include_reformulation:
        hits.extend(
            find_inline_markers(
                text,
                CATEGORY_E_INLINE_RE,
                category="E",
                direction="both",
            )
        )
    hits = _dedupe_by_span(hits)
    hits = _filter_excluded_anaphora(text, hits)
    return _filter_hits_before_second_comma(text, hits)


def find_cataphora_markers(
    text: str,
    *,
    include_reformulation: bool = True,
    require_forward_punct: bool = True,
) -> list[MarkerHit]:
    """Find forward-looking example/elaboration markers."""
    # Cataphora expansion is disabled; keep API for compatibility.
    return []


def find_parenthetical_discourse_markers(text: str) -> list[MarkerHit]:
    """Convenience wrapper for parenthetical discourse markers."""
    return find_parenthetical_markers(text)


def find_temporal_markers(text: str) -> list[MarkerHit]:
    """Find local temporal/sequencing markers (Category F)."""
    return find_inline_markers(
        text,
        CATEGORY_F_INLINE_RE,
        category="F",
        direction="local",
    )

###########################################################################
# Sentence expansion helpers
###########################################################################


def _has_anaphora_marker(
    text: str,
    *,
    include_reformulation: bool = True,
    include_parenthetical: bool = True,
) -> bool:
    return bool(
        find_anaphora_markers(
            text,
            include_reformulation=include_reformulation,
            include_parenthetical=include_parenthetical,
        )
    )


def _has_cataphora_marker(
    text: str,
    *,
    include_reformulation: bool = True,
    require_forward_punct: bool = True,
) -> bool:
    return False


def _expand_sentence_text(
    sentences: Sequence[str],
    idx: int,
    *,
    extension: int,
    has_anaphora: bool,
    has_cataphora: bool,
) -> Tuple[str, int, int]:
    start = idx
    end = idx
    if has_anaphora:
        start = max(0, idx - extension)
    if has_cataphora:
        end = min(len(sentences) - 1, idx + extension)
    return " ".join(sentences[start : end + 1]), start, end
