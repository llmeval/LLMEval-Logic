"""Formalize stage: ask each LLM to produce solver-compatible FL JSON.

This is the first stage of the FL evaluation track and shares the dispatch
layer with the NL track (``code.nl_eval.eval``):

- Uses ``code.client.call_model`` (the same pluggable backend the NL eval uses)
  via ``asyncio.to_thread``, so all 14 model keys defined in ``MODEL_CONFIGS``
  are supported out of the box.
- Per ``(model, item)`` pair gets its own checkpoint file at
  ``<output_dir>/detail_results/<sanitized_model>/<index>.json``. The checkpoint
  is the single source of truth, so re-running with the same args resumes
  cleanly (skipping items whose checkpoint already carries a successful API
  response, regardless of whether the FL passed parsing).
- After all generations finish for a given model, a rolled-up JSONL
  ``formalize.<sanitized_model>.jsonl`` is written. A combined
  ``formalize_summary.json`` is written across all models at the end.

Both Free-FL and Fixed-FL prompts are loaded from
``code/fl_eval/formalize/formalize_system.md``. Mode is selected via the
``mode`` argument: ``"free"`` (default) or ``"fixed"`` (the latter injects
``parameters``/``translation`` from each item's ``formalization`` block as
read-only context, and the model only emits ``premise`` / ``question`` /
``reason``).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    MODEL_CONFIGS,
    ModelResult,
    call_model,
)


# ---------------------------------------------------------------------------
# Constants & dataclasses
# ---------------------------------------------------------------------------


MODE_FREE = "free"
MODE_FIXED = "fixed"
ALLOWED_MODES = (MODE_FREE, MODE_FIXED)

# Loaded once per process from formalize_system.md.
PROMPT_FILENAME = "formalize_system.md"

FL_LABEL_PATTERN = re.compile(r"\bfl\b\s*[:\uFF1A]\s*", re.IGNORECASE)
REASON_CAPTURE_PATTERN = re.compile(r"\breason\b\s*[:\uFF1A]\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class FormalizeRecord:
    """In-memory representation; mirrors the on-disk schema."""

    eval_id: str
    sample_id: Any
    title: str
    background: str
    question: str
    reference_answer: str
    mode: str
    model: str
    FL: Optional[Dict[str, Any]]
    reason: Optional[str]
    request_id: str
    thinking: str
    gen_raw: str
    latency_ms: int
    request_error: Optional[str]
    parse_error: bool
    parse_error_detail: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "sample_id": self.sample_id,
            "title": self.title,
            "background": self.background,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "mode": self.mode,
            "model": self.model,
            "FL": self.FL,
            "reason": self.reason,
            "request_id": self.request_id,
            "thinking": self.thinking,
            "gen_raw": self.gen_raw,
            "latency_ms": self.latency_ms,
            "_gen_request_error": self.request_error,
            "_gen_parse_error": self.parse_error,
            "_gen_parse_error_detail": self.parse_error_detail,
        }


# ---------------------------------------------------------------------------
# Prompt loading & assembly
# ---------------------------------------------------------------------------


def load_prompt_md() -> str:
    path = Path(__file__).resolve().parent / PROMPT_FILENAME
    return path.read_text(encoding="utf-8")


def _extract_section(prompt_md: str, heading: str) -> str:
    """Extract the ```text ... ``` body that follows ``## <heading>``."""
    pattern = re.compile(
        rf"## {re.escape(heading)}\s*\n+```(?:text)?\n(.*?)\n```",
        re.DOTALL,
    )
    match = pattern.search(prompt_md)
    if not match:
        raise RuntimeError(f"Section not found in {PROMPT_FILENAME}: {heading}")
    return match.group(1).strip()


def _extract_schema_hint(prompt_md: str, mode: str) -> str:
    flow = "Fixed-FL" if mode == MODE_FIXED else "Free-FL"
    pattern = re.compile(
        rf"## {re.escape(flow)} Schema Hint\s*\n+```json\n(.*?)\n```",
        re.DOTALL,
    )
    match = pattern.search(prompt_md)
    return match.group(1).strip() if match else ""


def build_system_prompt(prompt_md: str, mode: str) -> str:
    flow = "Fixed-FL" if mode == MODE_FIXED else "Free-FL"
    base = _extract_section(prompt_md, f"{flow} System Prompt")
    schema = _extract_schema_hint(prompt_md, mode)
    return base.replace("<SCHEMA_JSON>", schema)


def build_question_block(item: Dict[str, Any]) -> Tuple[str, str, str]:
    original = item.get("original") or {}
    background_raw = original.get("background")
    if isinstance(background_raw, list):
        background = "\n\n".join(str(p).strip() for p in background_raw if str(p).strip())
    else:
        background = (background_raw or "").strip() if isinstance(background_raw, str) else ""
    question_raw = original.get("question")
    question = (question_raw or "").strip() if isinstance(question_raw, str) else ""
    if background:
        block = f"Background: {background}\nQuestion: {question}"
    else:
        block = f"Question: {question}"
    return background, question, block


def extract_base_fl(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull ``parameters`` / ``translation`` from the bench item for Fixed-FL."""
    formalization = item.get("formalization")
    if not isinstance(formalization, dict):
        return None
    params = formalization.get("parameters")
    trans = formalization.get("translation")
    if not isinstance(params, dict) or not isinstance(trans, dict):
        return None
    return {"parameters": params, "translation": trans}


def build_user_prompt(question_block: str, base_fl: Optional[Dict[str, Any]]) -> str:
    if base_fl is None:
        return f"Problem:\n{question_block}"
    params_json = json.dumps(base_fl["parameters"], ensure_ascii=False, indent=2)
    trans_json = json.dumps(base_fl["translation"], ensure_ascii=False, indent=2)
    return (
        f"Problem:\n{question_block}\n\n"
        "Declared symbols (read-only; reference these in premise/question, "
        "do NOT restate or modify):\n"
        f"parameters:\n{params_json}\n\n"
        f"translation (symbol -> natural-language meaning):\n{trans_json}"
    )


def build_full_prompt(
    item: Dict[str, Any], mode: str, prompt_md: str
) -> Tuple[str, str, str, str, Optional[Dict[str, Any]]]:
    """Return ``(prompt, background, question, question_block, base_fl)``.

    ``prompt`` is the single concatenated string we hand to ``call_model``
    (system + user, separated by a blank line). ``client.call_model``
    accepts a single text prompt rather than a chat-style messages list.
    """
    background, question, question_block = build_question_block(item)
    base_fl = extract_base_fl(item) if mode == MODE_FIXED else None
    system = build_system_prompt(prompt_md, mode)
    user = build_user_prompt(question_block, base_fl)
    return f"{system}\n\n{user}", background, question, question_block, base_fl


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_FENCE_BLOCK_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?```",
    re.DOTALL,
)

# A fence block "looks like FL JSON" if it mentions any of the schema keys.
_FL_BLOCK_HINTS = ('"FL"', '"premise"', '"parameters"', '"question"')


def _strip_code_fence(text: str) -> str:
    """Return the inner content of the most-FL-shaped ```` ```...``` ```` block.

    Falls back to the original (stripped) text when no fence is found. This
    handles four layouts uniformly:

      1. Whole text is a fenced block (the simple case).
      2. Fenced block surrounded by Markdown preamble / postscript
         (Claude's typical output).
      3. *Multiple* fenced blocks (e.g. an ASCII-art diagram in the first
         block and the actual JSON in a later block) — we prefer the first
         block whose content mentions an FL schema key.
      4. No fence at all (just raw JSON).
    """
    text = (text or "").strip()
    if not text:
        return text
    blocks = [m.group(1).strip() for m in _FENCE_BLOCK_PATTERN.finditer(text)]
    if blocks:
        for blk in blocks:
            if any(h in blk for h in _FL_BLOCK_HINTS):
                return blk
        return blocks[0]
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _extract_first_json_object(text: str, start_index: int = 0) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start: Optional[int] = None
    for idx, ch in enumerate(text[start_index:], start_index):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]
    return None


_JSON_VALUE_TERMINATORS = frozenset(",:}]")
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _smart_escape_inner_chars(text: str) -> str:
    """Escape problematic chars that appear *inside* JSON string values.

    LLMs (Claude / Gemini in particular) sometimes write Chinese text that
    embeds raw ASCII characters that would otherwise terminate / corrupt the
    enclosing JSON string:

    * **Unescaped inner double quotes**, e.g.::

          "reason": "他说"如果下雨"，于是..."

      Strict ``json.loads`` aborts at the inner ``"``. We rewrite such inner
      quotes to ``\\"`` by lookahead: a ``"`` is treated as a closer iff the
      next non-whitespace char is a JSON value terminator
      (``,`` / ``:`` / ``}`` / ``]``) or end-of-input; otherwise it is
      escaped in place.

    * **Bare backslash followed by a non-escape char**, e.g.::

          "reason": "...表示为 \\neg A \\rightarrow \\neg C..."

      Inside a JSON string ``\\n`` / ``\\r`` / ``\\t`` / ``\\u…`` are valid
      escape sequences; ``\\w``, ``\\v``, ``\\d`` etc. raise
      ``Invalid \\escape``. We promote any orphan ``\\`` (a backslash whose
      next char is not in ``"\\/bfnrtu``) to a literal ``\\\\``. A
      ``\\u`` immediately followed by 4 hex digits is preserved.

    The function is a no-op on already-valid JSON.
    """
    out: List[str] = []
    n = len(text)
    i = 0
    in_string = False
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == "\"":
                in_string = True
            i += 1
            continue
        # Inside a string ----------------------------------------------------
        if ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _VALID_JSON_ESCAPES:
                if nxt == "u":
                    # Validate \uXXXX; if hex digits missing, escape the backslash.
                    hex4 = text[i + 2 : i + 6]
                    if len(hex4) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex4):
                        out.append("\\u")
                        out.extend(hex4)
                        i += 6
                        continue
                    out.append("\\\\u")
                    i += 2
                    continue
                out.append("\\")
                out.append(nxt)
                i += 2
                continue
            # Orphan backslash → literal backslash
            out.append("\\\\")
            i += 1
            continue
        if ch == "\"":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in _JSON_VALUE_TERMINATORS:
                out.append("\"")
                in_string = False
            else:
                out.append("\\\"")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# Back-compat alias so older code (and tests) keep working.
_smart_escape_inner_quotes = _smart_escape_inner_chars


def _try_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_load_lenient(text: str) -> Optional[Any]:
    """``json.loads`` with smart inner-char repair for LLM-produced JSON."""
    direct = _try_load(text)
    if direct is not None:
        return direct
    return _try_load(_smart_escape_inner_chars(text))


def _looks_like_fl_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    inner = obj.get("FL", obj)
    if not isinstance(inner, dict):
        return False
    return any(k in inner for k in ("premise", "question", "parameters"))


def _iter_balanced_objects(text: str) -> Iterable[str]:
    """Yield every top-level balanced ``{...}`` substring in *text*.

    Quote-aware (won't open/close on braces inside string literals).
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        in_string = False
        escape = False
        depth = 0
        start = i
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                j += 1
                continue
            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    i = j + 1
                    break
            j += 1
        else:
            return


def parse_formal_output(
    raw_text: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Return ``(fl_dict_or_None, reason_or_None, parse_error_detail_or_None)``.

    The parser is intentionally tolerant: it
      1. extracts the first fenced ``` ```json ... ``` ``` block (if any),
      2. tries direct ``json.loads`` first,
      3. falls back to a lenient parser that auto-escapes inner ``"`` chars
         inside string values (a common LLM mistake), and
      4. as a last resort, walks the string to find the first balanced
         ``{...}`` object and runs the same lenient parsing on it.

    If a top-level ``FL`` key exists, that's the FL; otherwise the parsed
    object itself is treated as the FL.
    """
    text = _strip_code_fence(raw_text or "").replace("\u00a0", " ")
    if not text:
        return None, None, "empty model response"

    parsed = _try_load_lenient(text)
    if parsed is None:
        snippet = _extract_first_json_object(text)
        if snippet is not None:
            parsed = _try_load_lenient(snippet)

    reason: Optional[str] = None
    fl: Optional[Dict[str, Any]] = None
    if isinstance(parsed, dict):
        reason_value = parsed.get("reason")
        if reason_value is not None:
            reason = str(reason_value).strip() or None
        fl_data = parsed.get("FL", parsed)
        if isinstance(fl_data, dict):
            fl = fl_data

    if fl is None:
        # Fallback: try regex-style "FL: {...}" + "reason: ..." stripping.
        reason_match = REASON_CAPTURE_PATTERN.search(text)
        if reason_match and reason is None:
            reason = reason_match.group(1).strip() or None
            text = text[: reason_match.start()].strip()
        fl_match = FL_LABEL_PATTERN.search(text)
        start_index = fl_match.end() if fl_match else 0
        snippet = _extract_first_json_object(text, start_index=start_index)
        if snippet is not None:
            candidate = _try_load_lenient(snippet)
            if isinstance(candidate, dict):
                inner = candidate.get("FL", candidate)
                if isinstance(inner, dict):
                    fl = inner

    if fl is None:
        # Last resort: walk the whole *original* (post-fence-strip) text and
        # try every balanced ``{...}`` substring under the lenient parser.
        # Prefer the one that already looks like FL JSON; fall back to the
        # largest dict.
        best: Optional[Dict[str, Any]] = None
        best_score = -1
        for cand_text in _iter_balanced_objects(text):
            obj = _try_load_lenient(cand_text)
            if not isinstance(obj, dict):
                continue
            inner = obj.get("FL", obj)
            if not isinstance(inner, dict):
                continue
            score = len(cand_text) + (10_000 if _looks_like_fl_dict(obj) else 0)
            if score > best_score:
                best_score = score
                best = obj
                if reason is None:
                    rv = obj.get("reason")
                    if rv is not None:
                        rcand = str(rv).strip()
                        if rcand:
                            reason = rcand
                fl = inner

        if fl is None and best is not None:
            inner = best.get("FL", best)
            if isinstance(inner, dict):
                fl = inner

    if fl is None:
        return None, reason, "could not parse FL JSON object from model response"

    return fl, reason, None


# ---------------------------------------------------------------------------
# Lightweight FL validation (no external solver dependency)
# ---------------------------------------------------------------------------


def _validate_free_fl(fl: Dict[str, Any]) -> Optional[str]:
    required = ("parameters", "translation", "premise", "question")
    missing = [k for k in required if k not in fl]
    if missing:
        return f"missing FL keys: {missing}"
    if not isinstance(fl.get("parameters"), dict):
        return "parameters must be an object"
    if not isinstance(fl.get("translation"), dict):
        return "translation must be an object"
    if not isinstance(fl.get("premise"), list):
        return "premise must be an array"
    if not isinstance(fl.get("question"), list):
        return "question must be an array"
    if not fl["question"]:
        return "question array is empty"
    for arr_name in ("premise", "question"):
        for elem in fl[arr_name]:
            if not isinstance(elem, str):
                return f"{arr_name} must contain only strings"
    return None


def _validate_fixed_fl(fl: Dict[str, Any]) -> Optional[str]:
    required = ("premise", "question")
    missing = [k for k in required if k not in fl]
    if missing:
        return f"missing FL keys: {missing}"
    if not isinstance(fl.get("premise"), list):
        return "premise must be an array"
    if not isinstance(fl.get("question"), list):
        return "question must be an array"
    if not fl["question"]:
        return "question array is empty"
    for arr_name in ("premise", "question"):
        for elem in fl[arr_name]:
            if not isinstance(elem, str):
                return f"{arr_name} must contain only strings"
    return None


def validate_fl(fl: Optional[Dict[str, Any]], mode: str, reason: Optional[str]) -> Optional[str]:
    if fl is None:
        return "FL is missing"
    if mode == MODE_FIXED:
        err = _validate_fixed_fl(fl)
    else:
        err = _validate_free_fl(fl)
    if err is None and not reason:
        err = "missing reason"
    return err


# ---------------------------------------------------------------------------
# Detail file paths & resume helpers (mirrors eval.py layout)
# ---------------------------------------------------------------------------


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def detail_path(output_dir: Path, model: str, item_id: int) -> Path:
    return output_dir / "detail_results" / sanitize_name(model) / f"{item_id}.json"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_detail(output_dir: Path, model: str, item_id: int) -> Optional[Dict[str, Any]]:
    path = detail_path(output_dir, model, item_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_generation_done(detail: Optional[Dict[str, Any]]) -> bool:
    """A detail is "done" when the API call succeeded.

    A successful call without a parseable FL is still considered done — we
    don't auto-retry parse failures (consistent with eval.py behavior). Use
    ``--force`` to force regeneration.
    """
    if not detail:
        return False
    if detail.get("_gen_request_error"):
        return False
    raw = detail.get("gen_raw") or ""
    return bool(raw.strip())


# ---------------------------------------------------------------------------
# Single-pair processing
# ---------------------------------------------------------------------------


def _build_record(
    item: Dict[str, Any],
    model: str,
    mode: str,
    background: str,
    question: str,
    base_fl: Optional[Dict[str, Any]],
    api_result: ModelResult,
    fl: Optional[Dict[str, Any]],
    reason: Optional[str],
    parse_error_detail: Optional[str],
) -> FormalizeRecord:
    if mode == MODE_FIXED and base_fl is not None and fl is not None:
        # In fixed mode, splice the read-only parameters/translation back in
        # so the on-disk record is a complete, solver-ready FL.
        fl_out: Optional[Dict[str, Any]] = {
            "parameters": base_fl["parameters"],
            "translation": base_fl["translation"],
            "premise": fl.get("premise"),
            "question": fl.get("question"),
        }
    else:
        fl_out = fl

    validation_err = validate_fl(fl_out, mode, reason)
    if api_result.error:
        # Transport-level error: mark parse_error too so downstream filters
        # treat it consistently.
        validation_err = api_result.error
    parse_err = validation_err is not None

    sample_id = item.get("id")
    eval_id = f"{sample_id}:formalize:{mode}"
    title = item.get("title") or f"item_{sample_id}"
    reference_answer = (item.get("original") or {}).get("answer") or ""
    if isinstance(reference_answer, list):
        reference_answer = "；".join(str(a).strip() for a in reference_answer if str(a).strip())

    return FormalizeRecord(
        eval_id=eval_id,
        sample_id=sample_id,
        title=str(title).strip() or f"item_{sample_id}",
        background=background,
        question=question,
        reference_answer=str(reference_answer),
        mode=mode,
        model=model,
        FL=fl_out,
        reason=reason,
        request_id=api_result.request_id or "",
        thinking=api_result.thinking or "",
        gen_raw=api_result.response or "",
        latency_ms=api_result.latency_ms,
        request_error=api_result.error,
        parse_error=parse_err,
        parse_error_detail=validation_err,
    )


async def _process_one(
    model: str,
    item: Dict[str, Any],
    mode: str,
    prompt_md: str,
    output_dir: Path,
    global_sem: asyncio.Semaphore,
    model_sem: asyncio.Semaphore,
    timeout: int,
    max_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
    prompt, background, question, _q_block, base_fl = build_full_prompt(item, mode, prompt_md)

    if mode == MODE_FIXED and base_fl is None:
        # Item has no parameters/translation declared; record a clean failure
        # without burning an API call.
        api_result = ModelResult(
            response="",
            thinking="",
            request_id="",
            error="missing formalization translation/parameters",
            latency_ms=0,
        )
        record = _build_record(
            item, model, mode, background, question, base_fl,
            api_result, None, None, "missing formalization translation/parameters",
        )
        payload = record.to_dict()
        _save_json(detail_path(output_dir, model, item.get("id", 0)), payload)
        return payload

    async with model_sem:
        async with global_sem:
            api_result: ModelResult = await asyncio.to_thread(
                call_model,
                model,
                prompt,
                timeout=timeout,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

    if api_result.error:
        record = _build_record(
            item, model, mode, background, question, base_fl,
            api_result, None, None, api_result.error,
        )
    else:
        fl, reason, parse_err = parse_formal_output(api_result.response)
        record = _build_record(
            item, model, mode, background, question, base_fl,
            api_result, fl, reason, parse_err,
        )

    payload = record.to_dict()
    _save_json(detail_path(output_dir, model, item.get("id", 0)), payload)
    return payload


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _normalize_items(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for fallback_idx, item in enumerate(samples):
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("id"), int):
            item = {**item, "id": fallback_idx}
        items.append(item)
    return items


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Input JSON must be a list of items: {input_path}")
    return _normalize_items(data)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_for_model(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    request_err = sum(1 for r in records if r.get("_gen_request_error"))
    parse_err = sum(1 for r in records if r.get("_gen_parse_error"))
    fl_ok = sum(1 for r in records if not r.get("_gen_parse_error"))
    avg_latency = (
        sum(int(r.get("latency_ms") or 0) for r in records) // total if total else 0
    )
    return {
        "total": total,
        "fl_ok": fl_ok,
        "fl_ok_rate": (fl_ok / total) if total else 0.0,
        "request_error_count": request_err,
        "parse_error_count": parse_err,
        "avg_latency_ms": avg_latency,
    }


async def run_async(
    input_path: Path,
    output_dir: Path,
    *,
    models: List[str],
    mode: str = MODE_FREE,
    concurrency: int = 30,
    per_model_concurrency: int = 8,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    limit: Optional[int] = None,
    force: bool = False,
    log_fn=print,
) -> Dict[str, Any]:
    if mode not in ALLOWED_MODES:
        raise SystemExit(f"--mode must be one of {ALLOWED_MODES}, got {mode}")

    # Match eval.py's executor sizing so per-model semaphores don't compete
    # for a tiny default thread pool.
    pool_size = max(concurrency * 2, 64)
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="formalize_call")
    )

    samples = load_samples(input_path)
    if limit is not None and limit > 0:
        samples = samples[:limit]
    if not samples:
        log_fn("No samples loaded; exiting.")
        return {"item_total": 0, "models": models, "mode": mode}

    prompt_md = load_prompt_md()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_sem = asyncio.Semaphore(max(1, concurrency))
    model_sems: Dict[str, asyncio.Semaphore] = {}
    cap_summary: List[str] = []
    for m in models:
        cap = MODEL_CONFIGS.get(m, {}).get("max_concurrency")
        effective = min(per_model_concurrency, cap) if cap else per_model_concurrency
        model_sems[m] = asyncio.Semaphore(max(1, effective))
        cap_summary.append(f"{m}={effective}")

    tasks: List[asyncio.Task[Dict[str, Any]]] = []
    skipped = 0
    cached_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    for model in models:
        if model not in MODEL_CONFIGS:
            log_fn(f"WARNING: unknown model key '{model}' (skipping)")
            continue
        for item in samples:
            existing = load_existing_detail(output_dir, model, item["id"])
            if not force and is_generation_done(existing):
                cached_by_model[model].append(existing)  # type: ignore[arg-type]
                skipped += 1
                continue
            tasks.append(
                asyncio.create_task(
                    _process_one(
                        model=model,
                        item=item,
                        mode=mode,
                        prompt_md=prompt_md,
                        output_dir=output_dir,
                        global_sem=global_sem,
                        model_sem=model_sems[model],
                        timeout=timeout,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )
                )
            )

    total_pairs = len(models) * len(samples)
    log_fn(
        f"Formalize[{mode}]: {total_pairs} pairs = {len(models)} models x {len(samples)} items; "
        f"skip(cached)={skipped}, to_run={len(tasks)}, "
        f"global_concurrency={concurrency}, per_model_concurrency={per_model_concurrency}; "
        f"effective per-model: {', '.join(cap_summary)}"
    )

    done = 0
    start_time = time.time()
    for task in asyncio.as_completed(tasks):
        row = await task
        done += 1
        elapsed = max(time.time() - start_time, 1e-6)
        rate = done / elapsed
        eta = (len(tasks) - done) / rate if rate > 0 else 0.0
        if row.get("_gen_request_error"):
            status = f"err={row['_gen_request_error']}"
        elif row.get("_gen_parse_error"):
            status = f"parse_fail={row.get('_gen_parse_error_detail')}"
        else:
            status = "ok"
        log_fn(
            f"gen [{done}/{len(tasks)}] mode={mode} model={row.get('model')} "
            f"item={row.get('sample_id')} status={status} "
            f"latency={row.get('latency_ms')}ms eta={eta:.1f}s"
        )

    # Roll up per-model JSONL + per-model summary
    per_model_summary: List[Dict[str, Any]] = []
    for model in models:
        if model not in MODEL_CONFIGS:
            continue
        model_records: List[Dict[str, Any]] = []
        for item in samples:
            detail = load_existing_detail(output_dir, model, item["id"])
            if detail is not None:
                model_records.append(detail)
        # Sort deterministically by sample_id for diffability.
        model_records.sort(key=lambda r: (r.get("sample_id") if isinstance(r.get("sample_id"), int) else -1))
        write_jsonl(output_dir / f"formalize.{sanitize_name(model)}.jsonl", model_records)
        stats = summarize_for_model(model_records)
        stats["model"] = model
        per_model_summary.append(stats)

    summary = {
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mode": mode,
        "models": [m for m in models if m in MODEL_CONFIGS],
        "item_total": len(samples),
        "model_metrics": per_model_summary,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(output_dir / "formalize_summary.json", summary)
    log_fn(f"Wrote {output_dir / 'formalize_summary.json'}")
    return summary


def run(input_path: Path, output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(run_async(input_path, output_dir, **kwargs))
