"""LLM helper utilities shared across pipeline stages."""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore


from src.utils.x_config import LLM_DEFAULTS, MAX_TOKENS, TEMPERATURE


logger = logging.getLogger(__name__)

TOKEN_TOTALS = {"prompt": 0, "completion": 0, "total": 0}

_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    max_retries=Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504]),
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def _post(url: str, json: dict[str, Any], timeout: int = 300):
    return _session.post(url, json=json, timeout=timeout, headers={"Connection": "keep-alive"})


def build_prompt(model_name: str, system: str, user: str):
    """Return an LLM prompt formatted for the given model."""
    name = model_name.lower()
    if "llama" in name:
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user.strip()})
        return messages
    return user.strip()


def question_list_grammar(min_n: int, max_n: int) -> str:
    """Enforce a JSON object {"Question List": ["..."]} with bounds."""
    rep_min = max(0, min_n - 1)
    rep_max = max(0, max_n - 1)
    return (
        "root    ::= ws \"{\" ws \"\\\"Question List\\\"\" ws \":\" ws \"[\" ws string"
        + " (ws \",\" ws string){" + str(rep_min) + "," + str(rep_max) + "} ws \"]\" ws \"}\" ws\n"
        "string  ::= \"\\\"\" chars \"\\\"\"\n"
        "chars   ::= (escape | char)*\n"
        "char    ::= [^\"\\\\\\x00-\\x1F]\n"
        "escape  ::= \"\\\\\" ([\"\\\\/bfnrt] | \"u\" hex hex hex hex)\n"
        "hex     ::= [0-9a-fA-F]\n"
        "ws      ::= ([ \\t\\r\\n])*\n"
    )


THINK_BLOCK_RE = re.compile(r"^<think>.*?</think>\s*", flags=re.S | re.I)


def strip_think(text: str) -> str:
    """Remove a leading reasoning block delimited by ``<think>`` tags."""
    text = THINK_BLOCK_RE.sub("", text)
    if text.lstrip().lower().startswith("<think>"):
        text = re.sub(r"^<think>\s*", "", text, flags=re.I)
    return text.strip()


def is_r1_like(model_name: str) -> bool:
    name = model_name.lower()
    return ("deepseek" in name and ("r1" in name or "distill" in name)) or "r1" in name


def _wrap_for_deepseek_user(prompt: str, task: str, reason: bool = True) -> str:
    """Embed DeepSeek-R1 guidance in the *user* message."""
    if reason and task in {"answer_generation"}:
        preface = (
            "All instructions are provided in this single user message (no system prompt). "
            "Begin your output with '<think>\\n' and use that block for your reasoning. "
            "After the think block, produce the structured output exactly as requested."
        )
        return f"{preface}\n\n{prompt}"
    return prompt


def query_llm(
    prompt,
    server_url,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    grammar: str | None = None,
    response_format: dict | None = None,
    model_name: str = "",
    phase: str | None = None,
    reason: bool = True,
    top_p: float = LLM_DEFAULTS["top_p"],
    top_k: int = LLM_DEFAULTS["top_k"],
    mirostat: int = LLM_DEFAULTS["mirostat"],
    repeat_penalty: float = LLM_DEFAULTS["repeat_penalty"],
    seed: int | None = None,
):
    """Query a local LLM server.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokens to generate. If ``None`` and ``phase`` is
        provided, the value is pulled from :data:`src.config.MAX_TOKENS`.
        Defaults to ``128`` when no phase is available.
    temperature:
        Sampling temperature. If ``None`` and ``phase`` is provided, the value
        is pulled from :data:`src.config.TEMPERATURE`. Defaults to ``0.2`` when
        no phase is available.
    """
    if response_format is not None and grammar is not None:
        raise ValueError("response_format and grammar cannot both be set")

    is_deepseek = "deepseek" in model_name.lower()
    use_chat = isinstance(prompt, list) or is_deepseek or response_format is not None

    prompt_text = "".join(m.get("content", "") for m in prompt) if isinstance(prompt, list) else prompt

    if max_tokens is None:
        max_tokens = MAX_TOKENS.get(phase, 128) if phase else 128
    if temperature is None:
        temperature = TEMPERATURE.get(phase, 0.2) if phase else 0.6

    if use_chat:
        endpoint = "/v1/chat/completions"
        if isinstance(prompt, list):
            messages = prompt
        else:
            content = _wrap_for_deepseek_user(prompt, phase or "", reason) if is_deepseek else prompt
            messages = [{"role": "user", "content": content}]
        payload: dict[str, Any] = {
            "model": "local",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "mirostat": mirostat,
            "repeat_penalty": repeat_penalty,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if grammar:
            payload["grammar"] = grammar
    else:
        if grammar:
            endpoint = "/v1/completions"
            payload = {
                "model": "local",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "grammar": grammar,
                "top_p": top_p,
                "top_k": top_k,
                "mirostat": mirostat,
                "repeat_penalty": repeat_penalty,
            }
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop
        else:
            endpoint = "/completion"
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "mirostat": mirostat,
                "repeat_penalty": repeat_penalty,
            }
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop

    r = _post(f"{server_url}{endpoint}", json=payload)
    r.raise_for_status()
    data = r.json()
    if use_chat:
        content = data["choices"][0]["message"]["content"]
    elif grammar:
        content = data.get("choices", [{}])[0].get("text", "")
    else:
        content = data.get("content", data.get("message", ""))

    usage = data.get("usage") if isinstance(data, dict) else None
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    elif tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(prompt_text))
        completion_tokens = len(enc.encode(content))
    else:
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(content.split())
        logger.debug("tiktoken not available; using word counts as approximation")
    total_tokens = prompt_tokens + completion_tokens

    logger.debug(
        f"{phase or 'query'} tokens - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}"
    )
    TOKEN_TOTALS["prompt"] += prompt_tokens
    TOKEN_TOTALS["completion"] += completion_tokens
    TOKEN_TOTALS["total"] += total_tokens
    logger.debug(
        f"Cumulative tokens - prompt: {TOKEN_TOTALS['prompt']}, "
        f"completion: {TOKEN_TOTALS['completion']}, total: {TOKEN_TOTALS['total']}"
    )

    return content, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


__all__ = [
    "build_prompt",
    "question_list_grammar",
    "strip_think",
    "is_r1_like",
    "_wrap_for_deepseek_user",
    "query_llm",
]
