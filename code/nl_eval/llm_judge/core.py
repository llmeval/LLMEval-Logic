"""LLM judge core. Output schema (judged.jsonl) matches the previous
`code/judge/judge.py`: adds `judge_match`, `judge_raw` on top of the
solve.jsonl fields; `_judge_*` fields are kept out of the written file.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Tuple

from ...lib.api_client import (
    call_chat_async,
    extract_json,
    gather_with_progress,
    read_api_credentials,
)
from ...lib.io_utils import load_json_list, load_jsonl, write_json, write_jsonl
from ...lib.prompts import extract_prompt_section, load_prompt


DEFAULT_JUDGE_MODEL = "gpt-5.2"
PASS_VERDICTS = {"pass"}
FAIL_VERDICTS = {"fail"}


def chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_judge_messages(items: List[Dict[str, Any]], prompt_md: str) -> List[Dict[str, str]]:
    system = extract_prompt_section(prompt_md, "System Prompt")
    user = "Items:\n" + json.dumps(items, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_bench_map(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    rows = load_json_list(Path(path))
    mapping: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, int):
            sample_id = idx
        mapping[sample_id] = row
    return mapping


def parse_sample_id(record: Dict[str, Any]) -> Optional[int]:
    for key in ("sample_id", "id"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
    eval_id = record.get("eval_id")
    if isinstance(eval_id, str):
        prefix = eval_id.split(":", 1)[0]
        try:
            return int(prefix)
        except ValueError:
            return None
    return None


def _normalize_verdict(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("-", "_")
    aliases = {
        "match": "pass",
        "matched": "pass",
        "true": "pass",
        "strict_pass": "pass",
        "strict_match": "pass",
        "strict": "pass",
        "soft_pass": "pass",
        "soft": "pass",
        "softpass": "pass",
        "soft_passed": "pass",
        "mismatch": "fail",
        "false": "fail",
        "wrong": "fail",
        "data": "fail",
        "data_issue": "fail",
        "gold_issue": "fail",
        "reference_issue": "fail",
    }
    text = aliases.get(text, text)
    if text in PASS_VERDICTS or text in FAIL_VERDICTS:
        return text
    return None


def normalize_judge_item(judge_item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize current {match} responses and legacy structured verdicts."""
    verdict = _normalize_verdict(
        judge_item.get("verdict")
        or judge_item.get("status")
    )
    raw_match = judge_item.get("match")
    if verdict is None:
        if raw_match is True:
            verdict = "pass"
        elif raw_match is False:
            verdict = "fail"
        else:
            verdict = "fail"

    match = verdict in PASS_VERDICTS

    reason = judge_item.get("reason") or judge_item.get("judge_reason") or ""
    return {
        "judge_match": match,
        "judge_reason": str(reason),
    }


async def run_judge_batch(
    sem: asyncio.Semaphore,
    endpoint: str,
    api_key: str,
    model: str,
    batch_items: List[Dict[str, Any]],
    prompt_md: str,
    *,
    timeout: float,
    use_json_mode: bool,
    store_raw: bool,
) -> Tuple[List[Dict[str, Any]], Optional[str], int, Optional[str]]:
    async with sem:
        start = time.time()
        messages = build_judge_messages(batch_items, prompt_md)
        api_result = await call_chat_async(
            endpoint, api_key, model, messages,
            temperature=0.0, timeout=timeout, use_json_mode=use_json_mode,
        )
        latency_ms = int((time.time() - start) * 1000)

    parsed = extract_json(api_result.content_text, allow_array=True)
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        results = parsed["results"]
    elif isinstance(parsed, list):
        results = parsed
    else:
        results = []

    raw_text = api_result.content_text if store_raw else None
    raw_http = api_result.raw_text if (store_raw and api_result.error) else None
    return results, api_result.error, latency_ms, raw_text or raw_http


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total": len(results), "by_prompt": {}}
    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_prompt.setdefault(r.get("prompt_type") or "unknown", []).append(r)
    for ptype, items in by_prompt.items():
        total = len(items)
        judge_parse_error = sum(1 for r in items if r.get("_judge_parse_error"))
        judge_match = sum(1 for r in items if r.get("judge_match") is True)
        judged_total = sum(1 for r in items if r.get("judge_match") in (True, False))
        summary["by_prompt"][ptype] = {
            "total": total,
            "judge_parse_error": judge_parse_error,
            "judge_match": judge_match,
            "judge_match_rate": (judge_match / judged_total) if judged_total else 0.0,
        }
    return summary


async def run_async(
    input_path: Path,
    output_path: Path,
    *,
    summary_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 4,
    batch_size: int = 5,
    timeout: float = 60.0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_json_mode: bool = True,
    store_raw: bool = False,
    show_progress: bool = True,
    overwrite: bool = False,
    bench_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    endpoint, api_key = read_api_credentials()
    results = load_jsonl(Path(input_path))
    if not results:
        raise SystemExit("No results found in input JSONL.")

    for idx, item in enumerate(results):
        if not isinstance(item.get("id"), int):
            item["id"] = idx
    if start is not None or end is not None:
        lo = start if start is not None else 0
        hi = end if end is not None else len(results) - 1
        results = [item for item in results if lo <= item.get("id", 0) <= hi]

    bench_map = load_bench_map(Path(bench_path) if bench_path else None)
    record_map: Dict[str, Dict[str, Any]] = {}
    judge_items: List[Dict[str, Any]] = []
    skipped = 0
    for record in results:
        eval_id = record.get("eval_id")
        if not eval_id:
            eval_id = f"{record.get('sample_id')}:{record.get('prompt_type') or 'unknown'}"
            record["eval_id"] = eval_id
        if (
            not overwrite
            and (
                isinstance(record.get("judge_match"), bool)
            )
        ):
            continue
        model_answer = record.get("model_answer")
        reference_answer = record.get("reference_answer")
        if not model_answer or reference_answer is None:
            record["judge_match"] = None
            record["_judge_parse_error"] = True
            skipped += 1
            continue
        sample_id = parse_sample_id(record)
        bench_record = bench_map.get(sample_id, {}) if sample_id is not None else {}
        original_block = bench_record.get("original") if isinstance(bench_record.get("original"), dict) else {}
        candidate_fl = record.get("FL") if isinstance(record.get("FL"), dict) else {}
        item: Dict[str, Any] = {
            "id": eval_id,
            "title": record.get("title") or bench_record.get("title") or "",
            "background": record.get("background") or original_block.get("background") or "",
            "question": record.get("question") or original_block.get("question") or "",
            "reference_answer": reference_answer,
            "model_answer": model_answer,
        }
        if isinstance(record.get("answer_tokens"), list):
            item["answer_tokens"] = record.get("answer_tokens")
        if isinstance(record.get("answer_payload"), list):
            item["answer_payload"] = record.get("answer_payload")
        if candidate_fl:
            item["fl_parameters"] = candidate_fl.get("parameters")
            item["fl_translation"] = candidate_fl.get("translation")
            item["fl_premise"] = candidate_fl.get("premise")
            item["fl_question"] = candidate_fl.get("question")
        judge_items.append(item)
        record_map[eval_id] = record

    prompt_md = load_prompt("judge_system.md", caller_file=__file__)
    sem = asyncio.Semaphore(concurrency)
    tasks: List[Awaitable[Any]] = []
    for batch in chunked(judge_items, batch_size):
        tasks.append(
            run_judge_batch(
                sem, endpoint, api_key, model, batch, prompt_md,
                timeout=timeout, use_json_mode=use_json_mode, store_raw=store_raw,
            )
        )
    judge_batches = await gather_with_progress(tasks, "Judge", enabled=show_progress)

    total_parse_errors = 0
    for batch_items, (judge_results, judge_error, latency_ms, raw_text) in zip(
        chunked(judge_items, batch_size), judge_batches
    ):
        by_id = {item.get("id"): item for item in judge_results if isinstance(item, dict)}
        for item in batch_items:
            eval_id = item["id"]
            record = record_map.get(eval_id)
            if record is None:
                continue
            judge_item = by_id.get(eval_id)
            if judge_item is None:
                record["_judge_parse_error"] = True
                record["judge_match"] = None
                record["judge_reason"] = "judge result missing"
                total_parse_errors += 1
            else:
                record["_judge_parse_error"] = False
                normalized = normalize_judge_item(judge_item)
                record.update(normalized)
            record["_judge_request_error"] = judge_error
            record["_judge_latency_ms"] = latency_ms
            if raw_text:
                record["judge_raw"] = raw_text

    write_jsonl(Path(output_path), results, drop_underscore=True)
    summary = summarize(results)
    summary["skipped"] = skipped
    summary["judge_parse_error"] = total_parse_errors
    write_json(Path(summary_path), summary)
    return results


def run(input_path: Path, output_path: Path, summary_path: Path, **kwargs: Any) -> List[Dict[str, Any]]:
    return asyncio.run(run_async(input_path, output_path, summary_path=summary_path, **kwargs))
