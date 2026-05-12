"""Pluggable model-call client for LLMEval-Logic.

The released `eval.py` is **endpoint-agnostic**. This file is the single place
where you wire up the model API(s) you want to evaluate against.

Two built-in dispatch paths are provided:

* ``"openai"``    — any OpenAI-compatible Chat Completions endpoint
                    (OpenAI, DeepInfra, Fireworks, OpenRouter, vLLM, SGLang,
                    Ollama in OpenAI-compat mode, ...). This is the default
                    path used when nothing else is configured.
* ``"anthropic"`` — Anthropic Messages API (Claude family).

There are three ways to add a model:

1.  **Use the built-in OpenAI-compatible default with no edits.**
    If you only need one OpenAI-compatible endpoint (e.g. an internal vLLM /
    OpenAI / OpenRouter proxy), just set ``OPENAI_BASE_URL`` and
    ``OPENAI_API_KEY`` in `.env` and pass ``--models <model-name>`` on the
    command line. The model name is forwarded as-is to the API.

2.  **Edit ``MODEL_CONFIGS``** below to register a stable ``--models`` key
    (per-model concurrency cap, request kwargs, dispatch backend, ...).

3.  **Register a fully custom callable** at runtime:
    ``register_model("my-model", lambda prompt, **kw: ModelResult(...))``.

The eval driver only depends on:

  - ``ModelResult`` dataclass
  - ``call_model(model, prompt, **kwargs) -> ModelResult``
  - ``MODEL_CONFIGS`` dict (used for per-model concurrency hints)
  - ``list_models()``
  - ``load_dotenv(path)``

so the rest of the pipeline does not need to change when you swap providers.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public defaults (consumed by eval.py CLI)
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TOKENS = 16384
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_MAX_DELAY = 60.0
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Unified return value of every backend.

    Fields
    ------
    response :
        The final assistant text. Empty string if the call failed.
    thinking :
        Optional reasoning trace if the API exposes one (e.g. OpenAI
        ``reasoning_content``, Anthropic ``thinking`` blocks).
    latency_ms :
        Wall-clock latency of the successful call (or last attempt on failure).
    error :
        Human-readable error message if the call ultimately failed; ``None``
        on success.
    request_id :
        Provider-side request id when available (useful for debugging).
    raw :
        Raw decoded JSON response of the final attempt (for inspection).
    """

    response: str = ""
    thinking: Optional[str] = None
    latency_ms: int = 0
    error: Optional[str] = None
    request_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# .env loader (zero deps; ignores comments and blank lines)
# ---------------------------------------------------------------------------


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Model registry (USER-EDITABLE)
# ---------------------------------------------------------------------------
#
# Each entry maps a CLI-facing key (what you pass to `--models`) to a config
# dict. Recognized fields:
#
#     backend              "openai" | "anthropic" | callable
#     model                actual model id sent to the provider
#     base_url_env         env var holding the API base URL (default OPENAI_BASE_URL)
#     api_key_env          env var holding the API key   (default OPENAI_API_KEY)
#     params               dict of extra kwargs merged into the request body
#                          (e.g. {"temperature": 0.0, "reasoning_effort": "high"})
#     max_concurrency      per-model in-flight cap honoured by eval.py
#
# An unregistered key falls through to the OpenAI-compatible default (the key
# itself is sent as the model id), so the simplest path is to leave this dict
# empty and just pass `--models <model-name>`.

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ===== Example: Tencent Hy3 preview via OpenRouter =====
    # OpenRouter normalizes Hy3's "disabled / low / high" reasoning levels
    # through the `reasoning` parameter. Both keys point at the same model id
    # (tencent/hy3-preview); thinking on/off is controlled by `params`.
    "hy3": {
        "backend": "openai",
        "model": "tencent/hy3-preview",
        "params": {"reasoning": {"effort": "high"}},
        "max_concurrency": 10,
    },
    "hy3-nothink": {
        "backend": "openai",
        "model": "tencent/hy3-preview",
        "params": {"reasoning": {"enabled": False}},
        "max_concurrency": 10,
    },
}


# Optional registry of fully custom dispatch callables, populated by
# `register_model(...)`. Wins over MODEL_CONFIGS if both define the same key.
_CUSTOM_DISPATCH: Dict[str, Callable[..., ModelResult]] = {}


def register_model(key: str, fn: Callable[..., ModelResult]) -> None:
    """Register a custom (model_key) -> ModelResult dispatcher.

    The callable receives the same kwargs as `call_model` minus the ``model``
    argument, i.e. ``fn(prompt, *, timeout, max_tokens, max_retries, **extra)``.
    """

    _CUSTOM_DISPATCH[key] = fn


def list_models() -> List[str]:
    return sorted(set(MODEL_CONFIGS) | set(_CUSTOM_DISPATCH))


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _should_retry(status_code: Optional[int], err_text: str) -> bool:
    if status_code in RETRYABLE_STATUS:
        return True
    s = err_text.lower()
    return any(
        kw in s
        for kw in (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "connection",
            "bad gateway",
        )
    )


def _backoff(attempt: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())


# ---------------------------------------------------------------------------
# Built-in dispatch: OpenAI-compatible Chat Completions
# ---------------------------------------------------------------------------


def _call_openai_compatible(
    *,
    prompt: str,
    model_id: str,
    base_url_env: str = "OPENAI_BASE_URL",
    api_key_env: str = "OPENAI_API_KEY",
    extra_params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ModelResult:
    base_url = os.environ.get(base_url_env, "https://api.openai.com/v1").rstrip("/")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return ModelResult(error=f"missing env {api_key_env}")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if extra_params:
        body.update(extra_params)

    last_err = "unknown"
    last_status: Optional[int] = None
    started = time.time()
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            last_status = r.status_code
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if not _should_retry(r.status_code, r.text) or attempt == max_retries:
                    return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))
            else:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                response = msg.get("content") or ""
                # Best-effort thinking trace capture.
                thinking = msg.get("reasoning_content") or msg.get("reasoning") or None
                return ModelResult(
                    response=response.strip(),
                    thinking=thinking if isinstance(thinking, str) else None,
                    latency_ms=int((time.time() - started) * 1000),
                    request_id=data.get("id"),
                    raw=data,
                )
        except requests.RequestException as exc:
            last_err = f"request error: {exc}"
            if not _should_retry(None, str(exc)) or attempt == max_retries:
                return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))

        time.sleep(_backoff(attempt, DEFAULT_RETRY_BASE_DELAY, DEFAULT_RETRY_MAX_DELAY))

    return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))


# ---------------------------------------------------------------------------
# Built-in dispatch: Anthropic Messages API
# ---------------------------------------------------------------------------


def _call_anthropic(
    *,
    prompt: str,
    model_id: str,
    base_url_env: str = "ANTHROPIC_BASE_URL",
    api_key_env: str = "ANTHROPIC_API_KEY",
    extra_params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ModelResult:
    base_url = os.environ.get(base_url_env, "https://api.anthropic.com/v1").rstrip("/")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return ModelResult(error=f"missing env {api_key_env}")

    url = f"{base_url}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if extra_params:
        body.update(extra_params)

    last_err = "unknown"
    started = time.time()
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if not _should_retry(r.status_code, r.text) or attempt == max_retries:
                    return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))
            else:
                data = r.json()
                response_parts: List[str] = []
                thinking_parts: List[str] = []
                for block in data.get("content", []) or []:
                    btype = block.get("type")
                    if btype == "text":
                        response_parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        thinking_parts.append(block.get("thinking", ""))
                return ModelResult(
                    response="".join(response_parts).strip(),
                    thinking="\n".join(p for p in thinking_parts if p) or None,
                    latency_ms=int((time.time() - started) * 1000),
                    request_id=data.get("id"),
                    raw=data,
                )
        except requests.RequestException as exc:
            last_err = f"request error: {exc}"
            if not _should_retry(None, str(exc)) or attempt == max_retries:
                return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))

        time.sleep(_backoff(attempt, DEFAULT_RETRY_BASE_DELAY, DEFAULT_RETRY_MAX_DELAY))

    return ModelResult(error=last_err, latency_ms=int((time.time() - started) * 1000))


# ---------------------------------------------------------------------------
# Public entry point used by eval.py
# ---------------------------------------------------------------------------


def call_model(
    model: str,
    prompt: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **_: Any,
) -> ModelResult:
    """Dispatch a single chat completion to whatever backend is configured for `model`.

    Lookup order:

    1.  custom dispatcher registered via :func:`register_model`
    2.  entry in :data:`MODEL_CONFIGS` (uses its ``backend`` field)
    3.  fallback: OpenAI-compatible default with ``model`` sent as-is
    """

    if model in _CUSTOM_DISPATCH:
        return _CUSTOM_DISPATCH[model](
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    cfg = MODEL_CONFIGS.get(model, {})
    backend = cfg.get("backend", "openai")
    model_id = cfg.get("model", model)
    extra = dict(cfg.get("params") or {})

    if callable(backend):
        return backend(
            prompt=prompt,
            model_id=model_id,
            extra_params=extra,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if backend == "anthropic":
        return _call_anthropic(
            prompt=prompt,
            model_id=model_id,
            base_url_env=cfg.get("base_url_env", "ANTHROPIC_BASE_URL"),
            api_key_env=cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            extra_params=extra,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    return _call_openai_compatible(
        prompt=prompt,
        model_id=model_id,
        base_url_env=cfg.get("base_url_env", "OPENAI_BASE_URL"),
        api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
        extra_params=extra,
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


