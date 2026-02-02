"""Default configuration values for local LLM calls."""

from __future__ import annotations

MAX_TOKENS = {
    "cs": 200,
    "iqoq_generation": 192,
    "edge_selection": 5,
    "answer_generation": 512,
}

TEMPERATURE = {
    "cs": 0.0,
    "iqoq_generation": 0.6,
    "edge_selection": 0.7,
    "answer_generation": 0.6,
}

LLM_DEFAULTS = {
    "top_p": 0.95,
    "top_k": 0,
    "mirostat": 0,
    "repeat_penalty": 1.0,
}

__all__ = ["LLM_DEFAULTS", "MAX_TOKENS", "TEMPERATURE"]
