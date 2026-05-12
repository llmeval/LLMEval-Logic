#!/usr/bin/env python3
"""LLMEval-Logic evaluator (resumable, per-item checkpoints, LLM-as-Judge).

Single entry point for both Base and Hard subsets. The workflow is:

1.  Generation: each (model, item) pair calls `client.call_model` and writes
    one detail file at ``<output_dir>/detail_results/<sanitized_model>/<idx>.json``.
2.  Judge: each detail file is sent to the judge model for semantic-equivalence
    scoring, and the same detail file is updated in place.
3.  Summary: a per-run ``eval_summary.json`` aggregates per-model Item Acc.\
    and Sub-Q Acc.\ for the run.

The evaluator is endpoint-agnostic: which providers/models you call against,
and which judge you use, are entirely defined in ``code/client.py``. See its
module docstring for the three ways to register a model.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    MODEL_CONFIGS,
    ModelResult,
    call_model,
    list_models,
    load_dotenv,
)

TASK_MULTI = "multi"
TASK_SINGLE = "single"
TASK_AUTO = "auto"


# ---------------------------------------------------------------------------
# Logging helper (stdout + file)
# ---------------------------------------------------------------------------


class TeeLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# IO / loading
# ---------------------------------------------------------------------------


def _detect_task_format(data: Any) -> Optional[str]:
    if not isinstance(data, list):
        return None
    sample = next((item for item in data if isinstance(item, dict)), None)
    if sample is None:
        return None
    if isinstance(sample.get("question"), list) and isinstance(sample.get("answer"), list):
        return TASK_MULTI
    original = sample.get("original")
    if isinstance(original, dict):
        q = original.get("question")
        a = original.get("answer")
        if isinstance(q, list) and isinstance(a, list):
            return TASK_MULTI
        if isinstance(q, str) and a is not None and not isinstance(a, list):
            return TASK_SINGLE
    return None


def _normalize_background(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return "\n\n".join(parts)
    if value is None:
        return ""
    return str(value).strip()


def _normalize_item(item: Dict[str, Any], fallback_index: int, task_format: str) -> Optional[Dict[str, Any]]:
    source = item.get("original") if isinstance(item.get("original"), dict) else item
    questions = source.get("question")
    answers = source.get("answer")
    if task_format == TASK_MULTI:
        if not isinstance(questions, list) or not isinstance(answers, list):
            return None
        if len(questions) != len(answers):
            return None
    else:
        if not isinstance(questions, str) or answers is None or isinstance(answers, list):
            return None
        questions = [questions]
        answers = [answers]

    item_id = item.get("id", fallback_index)
    if not isinstance(item_id, int):
        item_id = fallback_index
    title = item.get("title") if isinstance(item.get("title"), str) else f"item_{item_id}"
    return {
        "id": item_id,
        "title": title.strip() or f"item_{item_id}",
        "task_format": task_format,
        "background": _normalize_background(source.get("background")),
        "questions": list(questions),
        "answers": list(answers),
    }


def load_items(path: Path, task_format: str) -> Tuple[List[Dict[str, Any]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list.")
    detected = _detect_task_format(data)
    resolved = detected if task_format == TASK_AUTO else task_format
    if resolved is None:
        raise SystemExit("Unable to auto-detect task format; use --task-format multi|single.")
    if detected and detected != resolved:
        raise SystemExit(f"Input looks like {detected}, but --task-format={resolved}.")

    items: List[Dict[str, Any]] = []
    for fallback_index, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_item(entry, fallback_index, resolved)
        if normalized is not None:
            items.append(normalized)
    return items, resolved


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "你是一个逻辑学专家，你要回答逻辑推理问题。"
    " 最终只输出一个 JSON 对象，不要输出额外说明、Markdown 或代码块。"
    " answer 应是直接回答该问题的结构化短文本，可以是权限、状态、对象集合、动作、顺序、证据需求或简洁结论。"
    " reasoning 应是对应的简短理由。"
    ' 如果某条问题以"反事实："开头，只在该反事实条件下作答。'
)


def build_prompt(item: Dict[str, Any]) -> str:
    questions = item["questions"]
    question_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    if item["task_format"] == TASK_SINGLE:
        format_instruction = ' 输出格式必须是 {"answer": "问题答案", "reasoning": "简短理由"}。'
    else:
        format_instruction = (
            " 输出格式必须是按小问编号分组的对象，例如："
            ' {"1": {"answer": "第1小问答案", "reasoning": "第1小问理由"}, '
            '"2": {"answer": "第2小问答案", "reasoning": "第2小问理由"}, ...}。'
        )
    user = (
        f"题目标题：{item['title']}\n\n"
        f"背景：\n{item['background']}\n\n"
        f"问题（共 {len(questions)} 条）：\n{question_block}\n\n"
        "请逐条作答，并给出对应的简短理由。"
    )
    return SYSTEM_PROMPT + format_instruction + "\n\n" + user


# ---------------------------------------------------------------------------
# Response parsing (simplified)
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _normalize_quotes(text: str) -> str:
    """Replace Chinese/fullwidth quotes with ASCII equivalents inside JSON strings."""
    return text.replace("\u201c", "'").replace("\u201d", "'").replace("\u300c", "'").replace("\u300d", "'").replace("\uff02", "'")


def _extract_json_obj(text: str) -> Optional[Any]:
    text = _strip_code_fence(text)
    if not text:
        return None
    # Try raw first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try with Chinese quotes normalized
    normalized = _normalize_quotes(text)
    if normalized != text:
        try:
            return json.loads(normalized)
        except Exception:
            pass
    decoder = json.JSONDecoder()
    for candidate in (text, normalized):
        for index, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
                return parsed
            except Exception:
                continue
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "；".join(_stringify(v) for v in value if _stringify(v))
    if isinstance(value, dict):
        for key in ("answer", "prediction", "output", "result"):
            if key in value:
                return _stringify(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _parse_numbered_dict(data: Dict[str, Any], expected: int) -> Optional[Tuple[List[str], List[str]]]:
    numbered: List[Tuple[int, Any]] = []
    for key, value in data.items():
        match = re.fullmatch(r"(?:q|question|a|answer)?\s*([0-9]+)", str(key).strip(), re.IGNORECASE)
        if match:
            numbered.append((int(match.group(1)), value))
    if len(numbered) != expected:
        return None
    numbered.sort()
    answers: List[str] = []
    reasonings: List[str] = []
    for _, value in numbered:
        if isinstance(value, dict):
            ans = value.get("answer") or value.get("ans") or value.get("prediction") or ""
            reason = value.get("reasoning") or value.get("reason") or value.get("rationale") or ""
            answers.append(_stringify(ans))
            reasonings.append(_stringify(reason))
        else:
            answers.append(_stringify(value))
            reasonings.append("")
    return answers, reasonings


def _regex_extract_numbered(text: str, expected: int) -> Optional[Tuple[List[str], List[str]]]:
    """Lenient regex fallback for malformed JSON (e.g., unescaped inner quotes).

    Splits the text into ``"N": {...}`` blocks per top-level numbered key by
    walking braces; for each block extracts ``answer`` / ``reasoning`` either
    via JSON.loads on the block or via regex of "answer": "..." . Tolerates
    unescaped Chinese-style quotes by greedy matching to the next ``,`` or
    block end.
    """
    text = _strip_code_fence(text)
    # Find the outer object body
    obj_match = re.search(r"\{(.*)\}", text, re.DOTALL)
    if not obj_match:
        return None
    body = obj_match.group(1)

    # Split on top-level "N": occurrences while tracking brace depth.
    # Always test for the numbered-key pattern first when at depth==0 and not inside a string,
    # since the key itself starts with `"`.
    key_pattern = re.compile(r'"\s*(\d+)\s*"\s*:\s*')
    positions: List[Tuple[int, int]] = []  # (key_int, value_start_idx)
    depth = 0
    in_string = False
    escape = False
    i = 0
    while i < len(body):
        if not in_string and depth == 0:
            m = key_pattern.match(body, i)
            if m:
                positions.append((int(m.group(1)), m.end()))
                i = m.end()
                continue
        ch = body[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if not positions:
        return None

    answer_re = re.compile(r'"\s*answer\s*"\s*:\s*"(.*?)"\s*(?:,|\}|$)', re.DOTALL)
    reason_re = re.compile(r'"\s*reasoning\s*"\s*:\s*"(.*?)"\s*(?:,|\}|$)', re.DOTALL)
    by_no: Dict[int, Tuple[str, str]] = {}
    for idx, (key_int, start) in enumerate(positions):
        end = positions[idx + 1][1] if idx + 1 < len(positions) else len(body)
        block = body[start:end]
        ans_m = answer_re.search(block)
        rea_m = reason_re.search(block)
        ans = ans_m.group(1).strip() if ans_m else ""
        rea = rea_m.group(1).strip() if rea_m else ""
        by_no[key_int] = (ans, rea)

    answers: List[str] = []
    reasonings: List[str] = []
    for n in range(1, expected + 1):
        ans, rea = by_no.get(n, ("", ""))
        answers.append(ans)
        reasonings.append(rea)
    if not any(answers):
        return None
    return answers, reasonings


def parse_model_output(text: str, expected: int) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    parsed = _extract_json_obj(text)
    if isinstance(parsed, dict):
        if expected == 1 and ("answer" in parsed or "ans" in parsed):
            answer = _stringify(parsed.get("answer") or parsed.get("ans"))
            reason = _stringify(parsed.get("reasoning") or parsed.get("reason") or "")
            return [answer], ([reason] if reason else None)
        numbered = _parse_numbered_dict(parsed, expected)
        if numbered:
            answers, reasons = numbered
            return answers, (reasons if any(reasons) else None)
        values = parsed.get("answers") or parsed.get("predictions")
        if isinstance(values, list) and len(values) == expected:
            return [_stringify(v) for v in values], None
    if isinstance(parsed, list) and len(parsed) == expected:
        return [_stringify(v) for v in parsed], None
    # Fallback: regex-based extraction tolerant of unescaped inner quotes
    raw = text or ""
    fallback = _regex_extract_numbered(raw, expected)
    if fallback:
        answers, reasons = fallback
        return answers, (reasons if any(reasons) else None)
    # Last resort for single-question: greedy regex on "answer": "..."
    if expected == 1:
        ans_m = re.search(r'"answer"\s*:\s*"(.*?)"\s*(?:,\s*"reasoning"|$)', _normalize_quotes(raw), re.DOTALL)
        if ans_m:
            answer = ans_m.group(1).strip()
            rea_m = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*\}', _normalize_quotes(raw), re.DOTALL)
            reason = rea_m.group(1).strip() if rea_m else ""
            return [answer], ([reason] if reason else None)
    return None, None


# ---------------------------------------------------------------------------
# Detail file helpers (single source of truth per (model, item))
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
    if not detail:
        return False
    if detail.get("error"):
        return False
    response = detail.get("response") or ""
    return bool(response.strip())


def is_judge_done(detail: Optional[Dict[str, Any]]) -> bool:
    if not is_generation_done(detail):
        return False
    per_q = (detail or {}).get("per_question") or []
    if not per_q:
        return False
    return all(isinstance(q.get("judge_match"), bool) for q in per_q)


def build_detail_payload(item: Dict[str, Any], model: str, result: ModelResult) -> Dict[str, Any]:
    expected = len(item["questions"])
    answers, reasonings = parse_model_output(result.response, expected)
    per_question: List[Dict[str, Any]] = []
    for offset, (question, reference) in enumerate(zip(item["questions"], item["answers"]), start=1):
        pred = answers[offset - 1] if (answers and offset - 1 < len(answers)) else None
        reason = reasonings[offset - 1] if (reasonings and offset - 1 < len(reasonings)) else None
        per_question.append(
            {
                "subquestion_no": offset,
                "question": question,
                "reference_answer": reference,
                "model_answer": pred,
                "model_reasoning": reason,
                "judge_match": None,
                "judge_reason": None,
            }
        )
    return {
        "model": model,
        "id": item["id"],
        "title": item["title"],
        "task_format": item["task_format"],
        "question_count": expected,
        "correct_count": 0,
        "question_exact_match": False,
        "request_id": result.request_id,
        "error": result.error,
        "parse_ok": answers is not None,
        "latency_ms": result.latency_ms,
        "response": result.response,
        "thinking": result.thinking,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Async generation with resume + immediate checkpointing
# ---------------------------------------------------------------------------


async def _process_one(
    model: str,
    item: Dict[str, Any],
    output_dir: Path,
    global_sem: asyncio.Semaphore,
    model_sem: asyncio.Semaphore,
    timeout: int,
    max_tokens: int,
    max_retries: int,
    logger: TeeLogger,
) -> Dict[str, Any]:
    prompt = build_prompt(item)
    async with model_sem:
        async with global_sem:
            result: ModelResult = await asyncio.to_thread(
                call_model,
                model,
                prompt,
                timeout=timeout,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
    payload = build_detail_payload(item, model, result)
    _save_json(detail_path(output_dir, model, item["id"]), payload)
    return payload


async def run_generation(
    items: List[Dict[str, Any]],
    models: List[str],
    output_dir: Path,
    concurrency: int,
    per_model_concurrency: int,
    timeout: int,
    max_tokens: int,
    max_retries: int,
    force: bool,
    logger: TeeLogger,
) -> List[Dict[str, Any]]:
    global_sem = asyncio.Semaphore(max(1, concurrency))
    # Use min(per_model_concurrency, MODEL_CONFIGS[model].max_concurrency)
    model_sems = {}
    cap_summary = []
    for m in models:
        cap = MODEL_CONFIGS.get(m, {}).get("max_concurrency")
        effective = min(per_model_concurrency, cap) if cap else per_model_concurrency
        model_sems[m] = asyncio.Semaphore(max(1, effective))
        cap_summary.append(f"{m}={effective}")
    tasks: List[asyncio.Task[Dict[str, Any]]] = []
    cached: List[Dict[str, Any]] = []
    skipped = 0
    for model in models:
        for item in items:
            existing = load_existing_detail(output_dir, model, item["id"])
            if not force and is_generation_done(existing):
                cached.append(existing)  # type: ignore[arg-type]
                skipped += 1
                continue
            tasks.append(
                asyncio.create_task(
                    _process_one(
                        model, item, output_dir, global_sem, model_sems[model],
                        timeout, max_tokens, max_retries, logger,
                    )
                )
            )
    total_pairs = len(models) * len(items)
    logger.log(
        f"Generation: {total_pairs} pairs = {len(models)} models x {len(items)} items; "
        f"skip(cached)={skipped}, to_run={len(tasks)}, "
        f"global_concurrency={concurrency}, per_model_concurrency={per_model_concurrency}; "
        f"effective per-model: {', '.join(cap_summary)}"
    )

    results: List[Dict[str, Any]] = list(cached)
    done = 0
    start = time.time()
    for task in asyncio.as_completed(tasks):
        row = await task
        results.append(row)
        done += 1
        elapsed = max(time.time() - start, 1e-6)
        rate = done / elapsed
        eta = (len(tasks) - done) / rate if rate > 0 else 0.0
        status = "ok" if not row.get("error") and row.get("parse_ok") else f"err={row.get('error') or 'parse_fail'}"
        logger.log(
            f"gen [{done}/{len(tasks)}] model={row['model']} item={row['id']} "
            f"status={status} latency={row.get('latency_ms')}ms eta={eta:.1f}s"
        )
    return results


# ---------------------------------------------------------------------------
# Judge stage (incremental) -- inline, no subprocess
# ---------------------------------------------------------------------------


JUDGE_SYSTEM = (
    "你是中文语义等价裁判，不是字面匹配器。"
    "输入中的每个 item 都是一整道包含多个小问的题。"
    "你必须先完整阅读整道题，再逐个判断每个小问的 model_answer 是否与 reference_answer 语义等价。"
    "如果集合、枚举、分组、投影、计数的含义一致，只是书写顺序不同、列表顺序不同、集合顺序不同，也必须判为正确。"
    "数量表达如“4”“4种”“四种”“共4种”视为等价；同极性的是/否、能/不能、存在/不存在视为等价。"
    "只有在 model_answer 与 reference_answer 实质矛盾、遗漏必要信息或加入实质不相容信息时才判为 false。"
    "你必须只返回一个 JSON 对象，不能返回 markdown、代码块或任何额外说明。"
    "返回格式必须严格满足下列结构："
    '{"results":[{"id":"bundle_id","subresults":[{"subquestion_no":1,"match":true,"reason":"中文简短理由"}]}]}。'
    "match 必须是 JSON 布尔 true 或 false，不能写成字符串。"
    "你必须为每个输入 item 的每个 subquestion 都返回一条 subresult。"
)


def _extract_judge_results(text: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(text.strip())
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            return []
    if not isinstance(parsed, dict):
        return []
    results = parsed.get("results")
    if not isinstance(results, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        sub = item.get("subresults")
        if isinstance(sub, list):
            for s in sub:
                if not isinstance(s, dict):
                    continue
                try:
                    subq = int(s.get("subquestion_no"))
                except (TypeError, ValueError):
                    continue
                match = s.get("match")
                if isinstance(match, bool):
                    out.append(
                        {
                            "subquestion_no": subq,
                            "match": match,
                            "reason": s.get("reason") if isinstance(s.get("reason"), str) else None,
                        }
                    )
    return out


async def _judge_one_detail(
    detail: Dict[str, Any],
    judge_model: str,
    sem: asyncio.Semaphore,
    timeout: int,
    max_tokens: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Optional[str]]:
    payload = {
        "id": f"{detail['id']}:{sanitize_name(detail['model'])}",
        "subquestions": [
            {
                "subquestion_no": q["subquestion_no"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "model_answer": q["model_answer"],
            }
            for q in detail.get("per_question", [])
        ],
    }
    prompt = (
        JUDGE_SYSTEM
        + "\n\n待评判条目如下。请严格按指定 JSON 返回，不要输出任何额外文本。\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    async with sem:
        result: ModelResult = await asyncio.to_thread(
            call_model,
            judge_model,
            prompt,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    if result.error:
        return detail, result.error
    parsed = _extract_judge_results(result.response)
    if not parsed:
        return detail, "judge response unparseable"
    by_no = {item["subquestion_no"]: item for item in parsed}
    for q in detail.get("per_question", []):
        match_item = by_no.get(q["subquestion_no"])
        if match_item is None:
            continue
        q["judge_match"] = bool(match_item.get("match"))
        q["judge_reason"] = match_item.get("reason")
    correct = sum(1 for q in detail.get("per_question", []) if q.get("judge_match") is True)
    detail["correct_count"] = correct
    detail["question_exact_match"] = correct == detail.get("question_count", 0)
    return detail, None


async def run_judge(
    output_dir: Path,
    models: List[str],
    items: List[Dict[str, Any]],
    judge_model: str,
    concurrency: int,
    timeout: int,
    max_tokens: int,
    max_retries: int,
    force: bool,
    logger: TeeLogger,
) -> None:
    tasks: List[asyncio.Task[Tuple[Dict[str, Any], Optional[str]]]] = []
    sem = asyncio.Semaphore(max(1, concurrency))
    to_judge: List[Dict[str, Any]] = []
    for model in models:
        for item in items:
            detail = load_existing_detail(output_dir, model, item["id"])
            if not is_generation_done(detail):
                continue
            if not force and is_judge_done(detail):
                continue
            to_judge.append(detail)  # type: ignore[arg-type]

    logger.log(f"Judge: {len(to_judge)} bundle(s) need judging (judge_model={judge_model})")
    for detail in to_judge:
        tasks.append(
            asyncio.create_task(
                _judge_one_detail(detail, judge_model, sem, timeout, max_tokens, max_retries)
            )
        )

    done = 0
    start = time.time()
    for task in asyncio.as_completed(tasks):
        detail, err = await task
        done += 1
        elapsed = max(time.time() - start, 1e-6)
        eta = (len(tasks) - done) / (done / elapsed) if done > 0 else 0.0
        if err is None:
            _save_json(detail_path(output_dir, detail["model"], detail["id"]), detail)
            logger.log(
                f"judge [{done}/{len(tasks)}] model={detail['model']} item={detail['id']} "
                f"correct={detail['correct_count']}/{detail['question_count']} eta={eta:.1f}s"
            )
        else:
            logger.log(
                f"judge [{done}/{len(tasks)}] model={detail['model']} item={detail['id']} "
                f"ERROR: {err}"
            )


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def build_summary(
    output_dir: Path,
    items: List[Dict[str, Any]],
    models: List[str],
) -> Dict[str, Any]:
    subq_total = sum(len(item["questions"]) for item in items)
    model_metrics: List[Dict[str, Any]] = []
    for model in models:
        item_exact = 0
        subq_correct = 0
        parse_fail = 0
        request_error = 0
        judge_missing = 0
        total_latency = 0
        loaded = 0
        for item in items:
            detail = load_existing_detail(output_dir, model, item["id"])
            if detail is None:
                continue
            loaded += 1
            if detail.get("error"):
                request_error += 1
            if not detail.get("parse_ok"):
                parse_fail += 1
            total_latency += int(detail.get("latency_ms") or 0)
            per_q = detail.get("per_question") or []
            if per_q and all(isinstance(q.get("judge_match"), bool) for q in per_q):
                if all(q.get("judge_match") for q in per_q):
                    item_exact += 1
                subq_correct += sum(1 for q in per_q if q.get("judge_match") is True)
            else:
                judge_missing += 1
        model_metrics.append(
            {
                "model": model,
                "question_total": len(items),
                "loaded": loaded,
                "question_exact_match_count": item_exact,
                "question_exact_match_accuracy": item_exact / len(items) if items else 0.0,
                "subquestion_total": subq_total,
                "subquestion_correct_count": subq_correct,
                "subquestion_accuracy": subq_correct / subq_total if subq_total else 0.0,
                "parse_fail_count": parse_fail,
                "request_error_count": request_error,
                "judge_missing_count": judge_missing,
                "avg_latency_ms": round(total_latency / max(loaded, 1), 2),
            }
        )
    return {
        "models": models,
        "item_total": len(items),
        "subquestion_total": subq_total,
        "model_metrics": model_metrics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    load_dotenv(args.dotenv)

    # Replace asyncio's default executor (which caps at min(32, cpu+4) threads).
    # Without this, all per-model semaphores share the same tiny thread pool, so a
    # slow model can block other models even though their semaphores are free.
    pool_size = max(args.concurrency * 2, 64)
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="model_call")
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger(output_dir / "run.log")
    try:
        logger.log(f"cmd: {' '.join(sys.argv)}")
        logger.log(f"input={input_path}  output_dir={output_dir}")
        logger.log(f"models={args.models}  concurrency={args.concurrency}  max_retries={args.max_retries}")

        items, resolved_format = load_items(input_path, args.task_format)
        if args.limit is not None and args.limit > 0:
            items = items[: args.limit]
        models = list(args.models)
        if not items:
            logger.log("No items loaded; writing empty summary and exiting.")
            _save_json(output_dir / "eval_summary.json", {"item_total": 0, "models": models})
            return {"item_total": 0}

        logger.log(f"Loaded {len(items)} items ({resolved_format}); running {len(models)} model(s)")

        # Phase 1: generation for ALL models (parallel, each model has own sem)
        await run_generation(
            items=items,
            models=models,
            output_dir=output_dir,
            concurrency=args.concurrency,
            per_model_concurrency=args.per_model_concurrency,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            force=args.force,
            logger=logger,
        )

        # Phase 2: judge per-model independently. Each model's items are judged
        # as soon as that model's gen is done (which it already is from Phase 1).
        # This decouples models: a slow model (kimi) doesn't block judge for others.
        if not args.skip_judge:
            for model in models:
                n_need = sum(
                    1
                    for item in items
                    if is_generation_done(load_existing_detail(output_dir, model, item["id"]))
                    and not is_judge_done(load_existing_detail(output_dir, model, item["id"]))
                )
                if n_need == 0:
                    continue
                logger.log(f"Judging {n_need} items for model={model}")
                await run_judge(
                    output_dir=output_dir,
                    models=[model],
                    items=items,
                    judge_model=args.judge_model,
                    concurrency=args.judge_concurrency,
                    timeout=args.timeout,
                    max_tokens=args.judge_max_tokens,
                    max_retries=args.max_retries,
                    force=args.force_judge,
                    logger=logger,
                )

        summary = build_summary(output_dir, items, models)
        summary.update(
            {
                "input": str(input_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "task_format": resolved_format,
                "judge_model": None if args.skip_judge else args.judge_model,
            }
        )
        _save_json(output_dir / "eval_summary.json", summary)
        logger.log("Summary written to eval_summary.json")
        return summary
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLMEval-Logic evaluator: runs answer generation and LLM-as-Judge "
            "scoring for one or more model keys against a bench JSON file. "
            "Resumable per (model, item) detail file."
        )
    )
    parser.add_argument("--input", required=True, help="Input bench JSON (Base or Hard).")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "One or more model keys understood by code/client.py "
            "(either a key in MODEL_CONFIGS or a model id forwarded as-is "
            "to the OpenAI-compatible default backend)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help=(
            "Model key used as the LLM-as-Judge for semantic-equivalence "
            "scoring (resolved through the same code/client.py registry)."
        ),
    )
    parser.add_argument(
        "--task-format",
        choices=[TASK_AUTO, TASK_MULTI, TASK_SINGLE],
        default=TASK_AUTO,
        help="Auto-detected from the input by default; override only if needed.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items (smoke test).")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Global in-flight concurrency cap across all models.")
    parser.add_argument("--per-model-concurrency", type=int, default=8,
                        help="Per-model concurrency cap (capped by client.MODEL_CONFIGS[model]['max_concurrency'] if set).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Retries per request.")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--skip-judge", action="store_true", help="Skip the LLM-judge stage.")
    parser.add_argument("--force", action="store_true", help="Re-run generation even if detail files exist.")
    parser.add_argument("--force-judge", action="store_true", help="Re-run judge even if already judged.")
    parser.add_argument("--dotenv", default=".env", help="Path to a dotenv file with API credentials (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_eval(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
