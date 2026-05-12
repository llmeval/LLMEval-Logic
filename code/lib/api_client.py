"""HTTP client + dotenv + chat-completion helpers shared by all pipeline stages.

Consolidates the formerly duplicated implementations from:
- code/generate/formalize.py
- code/generate/solve_from_formal.py (no HTTP, but extract_chat_content was duplicated)
- code/judge/judge.py
- code/rubric/run_rubric_eval.py

Public API:
- ApiResult dataclass
- load_dotenv(path)
- build_endpoint(base_url)
- post_json(url, headers, payload, timeout) -> ApiResult
- extract_chat_content(parsed) -> str
- call_chat_async(endpoint, api_key, model, messages, *, temperature, timeout, use_json_mode) -> ApiResult
- gather_with_progress(tasks, label, enabled) -> list
- strip_code_fence(text) -> str
- extract_json(text, *, allow_array=False) -> Optional[Any]
- read_api_credentials() -> (endpoint, api_key)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Tuple


@dataclass
class ApiResult:
    content_text: str
    raw_text: str
    error: Optional[str]


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions") or base_url.endswith("/responses"):
        return base_url
    return base_url + "/chat/completions"


def read_api_credentials(dotenv_path: str = ".env") -> Tuple[str, str]:
    """Load .env then return (endpoint, api_key). Raises SystemExit if missing.

    Honours, in order: ``OPENAI_BASE_URL`` (preferred for the public release),
    then the historical ``BASE_URL`` / ``API_URL`` aliases. Same fallback
    chain for the API key (``OPENAI_API_KEY`` → ``API_KEY``).
    """
    load_dotenv(dotenv_path)
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("API_URL")
        or ""
    )
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
    if not base_url or not api_key:
        raise SystemExit(
            "Missing OPENAI_BASE_URL or OPENAI_API_KEY in environment or .env "
            "(see .env.example)."
        )
    return build_endpoint(base_url), api_key


def extract_chat_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        choice0 = choices[0]
        message = choice0.get("message") or {}
        if "content" in message and message["content"] is not None:
            return message["content"]
        if "text" in choice0 and choice0["text"] is not None:
            return choice0["text"]
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    return ""


def post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> ApiResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except Exception:
            return ApiResult(content_text="", raw_text=raw, error="invalid json response")
        return ApiResult(content_text=extract_chat_content(parsed), raw_text=raw, error=None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return ApiResult(content_text="", raw_text=raw, error=f"HTTP {exc.code}")
    except Exception as exc:
        return ApiResult(content_text="", raw_text="", error=str(exc))


async def call_chat_async(
    endpoint: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    timeout: float = 60.0,
    use_json_mode: bool = False,
    retries: int = 3,
    retry_backoff: float = 2.0,
) -> ApiResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    attempt = 0
    last: Optional[ApiResult] = None
    while True:
        result = await asyncio.to_thread(post_json, endpoint, headers, payload, timeout)
        if result.error is None:
            return result
        last = result
        if not _is_transient_error(result.error) or attempt >= retries:
            return result
        delay = retry_backoff * (2 ** attempt)
        await asyncio.sleep(delay)
        attempt += 1


_TRANSIENT_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "remote end closed",
)


def _is_transient_error(error: str) -> bool:
    if not isinstance(error, str):
        return False
    lower = error.lower()
    if any(marker in lower for marker in _TRANSIENT_MARKERS):
        return True
    # HTTP 5xx are transient; 4xx are not.
    if lower.startswith("http "):
        try:
            code = int(lower.split()[1])
        except (IndexError, ValueError):
            return False
        return 500 <= code < 600
    # urllib's urlopen error text usually starts with "<urlopen error ...>"
    if "urlopen error" in lower:
        return True
    return False


def format_progress(label: str, done: int, total: int, start_time: float) -> str:
    if total <= 0:
        return f"{label} 0/0"
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = int((done / total) * 100)
    bar_len = 24
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    return f"{label} [{bar}] {done}/{total} {pct:3d}% eta {eta:0.1f}s"


async def gather_with_progress(tasks: List[Awaitable[Any]], label: str, enabled: bool) -> List[Any]:
    if not tasks:
        return []
    if not enabled:
        return await asyncio.gather(*tasks)
    start = time.time()
    total = len(tasks)
    results: List[Any] = [None] * total

    async def _wrap(idx: int, awaitable: Awaitable[Any]) -> Tuple[int, Any]:
        return idx, await awaitable

    wrapped = [asyncio.create_task(_wrap(i, t)) for i, t in enumerate(tasks)]
    done = 0
    for fut in asyncio.as_completed(wrapped):
        idx, result = await fut
        results[idx] = result
        done += 1
        print(format_progress(label, done, total, start), end="\r", flush=True)
    print(format_progress(label, done, total, start), flush=True)
    return results


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def extract_json(text: str, *, allow_array: bool = False) -> Optional[Any]:
    """Lenient JSON extractor: full parse, then first {...} block, then first [...] block if allowed."""
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    if allow_array:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None
