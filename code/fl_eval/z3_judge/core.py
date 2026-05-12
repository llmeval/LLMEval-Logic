"""Z3-backed solver for candidate FL records.

Input schema (from formalize.jsonl): records with `sample_id`, `FL` (or
`formal_language`/`formalization`), optional `title`/`question`.
Output schema (solve.jsonl): preserved verbatim from the previous
code/generate/solve_from_formal.py — see repo README for the field list.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...lib.fl_schema import extract_formal_candidate
from ...lib.io_utils import load_json_list, load_jsonl, write_json, write_jsonl
from ...lib.z3_engine import (
    FormulaParseError,
    LogicEnvironment,
    LogicParser,
    QueryResult,
    check_necessary,
    check_possible,
    clean_symbol_name,
    enumerate_bool_models,
    enumerate_fol_objects,
    parse_query,
)


def format_models(models: Sequence[Sequence[str]]) -> str:
    tuples = ["(" + ", ".join(model) + ")" for model in models]
    return "{" + ", ".join(tuples) + "}"


def format_counts(counts: Sequence[int]) -> str:
    if len(counts) == 1:
        return str(counts[0])
    return "{" + ", ".join(str(count) for count in counts) + "}"


def translation_label(formal_language: Dict[str, Any], symbol: str) -> str:
    translation = formal_language.get("translation") or {}
    if symbol in translation:
        return str(translation[symbol])
    for key, value in translation.items():
        key_text = str(key)
        if key_text.startswith(symbol + "("):
            return str(value)
    return symbol


def models_to_text(
    formal_language: Dict[str, Any],
    models: Sequence[Sequence[str]],
    target_names: Optional[Sequence[str]] = None,
) -> str:
    if not models:
        return "没有可能的情况。"
    rendered = []
    for model in models:
        if target_names:
            active = set(model)
            parts = []
            for name in target_names:
                label = translation_label(formal_language, name)
                parts.append(f"{label}=是" if name in active else f"{label}=否")
            rendered.append("，".join(parts))
        elif not model:
            rendered.append("无列出命题成立")
        else:
            rendered.append("和".join(translation_label(formal_language, name) for name in model))
    return "可能的情况有：" + "；".join(rendered) + "。"


def tokens_to_model_answer(
    formal_language: Dict[str, Any],
    payload: Sequence[Dict[str, Any]],
    tokens: Sequence[str],
) -> str:
    if len(payload) == 1 and tokens and tokens[0] == "unknown":
        return "无法确定。"

    if len(payload) == 1:
        item = payload[0]
        if item["query_type"] == "enumerate_models":
            return models_to_text(formal_language, item.get("models") or [], item.get("targets"))
        if item["query_type"] == "count_models":
            count = item.get("count", 0)
            if isinstance(count, list):
                return "可能数量有：" + "、".join(str(value) for value in count) + "。"
            return f"可能数量为 {count}。"

    if len(tokens) == 2 and payload:
        query_types = [item.get("query_type") for item in payload]
        if query_types == ["possible", "necessary"]:
            pair_text = {
                ("possible", "necessary"): "有这种可能。一定。",
                ("possible", "unnecessary"): "有这种可能。不一定。",
                ("impossible", "unnecessary"): "没有这种可能。不会。",
                ("unknown", "unknown"): "无法确定。无法确定。",
            }
            text = pair_text.get((tokens[0], tokens[1]))
            if text:
                return text

    phrase_map = {
        "possible": "有这种可能。",
        "impossible": "没有这种可能。",
        "necessary": "一定。",
        "unnecessary": "不一定。",
        "unknown": "无法确定。",
    }
    return "".join(phrase_map.get(token, token) for token in tokens)


def solve_query(
    env: LogicEnvironment,
    parser: LogicParser,
    premises: Sequence[Any],
    query: str,
    timeout_ms: int,
    *,
    max_enumerate_assignments: Optional[int] = None,
    deadline: Optional[float] = None,
) -> QueryResult:
    query_type, target, args = parse_query(query)
    if deadline is not None and time.monotonic() >= deadline:
        return QueryResult(query, query_type, target, "unknown:record_timeout", "unknown")

    if query_type == "possible":
        if len(args) != 1:
            raise FormulaParseError(f"possible expects one argument: {query}")
        status, token = check_possible(premises, parser.parse(args[0]), timeout_ms)
        return QueryResult(query, query_type, target, status, token)

    if query_type == "necessary":
        if len(args) != 1:
            raise FormulaParseError(f"necessary expects one argument: {query}")
        status, token = check_necessary(premises, parser.parse(args[0]), timeout_ms)
        return QueryResult(query, query_type, target, status, token)

    if query_type in ("enumerate_models", "count_models"):
        is_fol = len(args) == 2 and re.search(r"\b" + re.escape(args[1]) + r"\b", args[0])
        target_names: Optional[List[str]] = None
        if is_fol:
            status, models = enumerate_fol_objects(
                env, parser, premises, args[0], args[1], timeout_ms,
                max_objects=max_enumerate_assignments,
                deadline=deadline,
            )
        else:
            target_names = [clean_symbol_name(arg) for arg in args]
            status, models = enumerate_bool_models(
                env, premises, target_names, timeout_ms,
                max_assignments=max_enumerate_assignments,
                deadline=deadline,
            )

        if status != "success":
            return QueryResult(
                query, query_type, target, status, "unknown",
                targets=target_names, models=models,
            )

        if query_type == "count_models":
            counts = sorted({len(model) for model in models}) if models else [0]
            count_value: Any = counts[0] if len(counts) == 1 else counts
            return QueryResult(
                query, query_type, target, status, format_counts(counts),
                targets=target_names, models=models, count=count_value,
            )
        return QueryResult(
            query, query_type, target, status, format_models(models),
            targets=target_names, models=models,
        )

    raise FormulaParseError(f"unsupported query type: {query_type}")


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


def load_data_map(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    items = load_json_list(Path(path))
    mapping: Dict[int, Dict[str, Any]] = {}
    for idx, item in enumerate(items):
        sample_id = item.get("id")
        if not isinstance(sample_id, int):
            sample_id = idx
        mapping[sample_id] = item
    return mapping


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    parse_error = sum(1 for r in results if r.get("_gen_parse_error"))
    unknown = sum(1 for r in results if r.get("z3_status") == "unknown")
    solved = sum(1 for r in results if r.get("z3_status") == "success")
    avg_latency = sum(r.get("_gen_latency_ms") or 0 for r in results) // total if total else 0
    return {
        "total": total,
        "solved": solved,
        "gen_parse_error": parse_error,
        "z3_unknown": unknown,
        "avg_gen_latency_ms": avg_latency,
    }


def solve_record(
    record: Dict[str, Any],
    data_map: Dict[int, Dict[str, Any]],
    timeout_ms: int,
    *,
    max_enumerate_assignments: Optional[int] = 4096,
    record_timeout_s: Optional[float] = 60.0,
) -> Dict[str, Any]:
    start = time.time()
    deadline = (
        time.monotonic() + record_timeout_s
        if record_timeout_s is not None and record_timeout_s > 0
        else None
    )
    sample_id = parse_sample_id(record)
    original = data_map.get(sample_id, {}) if sample_id is not None else {}
    original_block = original.get("original") if isinstance(original.get("original"), dict) else {}
    title = record.get("title") or original.get("title")
    question_text = record.get("question") or original_block.get("question") or ""
    reference_answer = original_block.get("answer") or record.get("reference_answer") or ""

    formal_language = extract_formal_candidate(record)
    if sample_id is None or formal_language is None:
        error = "missing sample_id" if sample_id is None else "missing candidate formalization with question"
        latency_ms = int((time.time() - start) * 1000)
        return {
            "eval_id": record.get("eval_id") or f"{sample_id}:solve",
            "sample_id": sample_id,
            "prompt_type": "solve",
            "title": title,
            "question": question_text,
            "reference_answer": reference_answer,
            "answer_tokens": [],
            "answer_payload": [],
            "fl_answer": "",
            "nl_answer": "",
            "model_answer": "",
            "reason": None,
            "FL": formal_language,
            "z3_status": "parse_error",
            "z3_error": error,
            "_gen_parse_error": True,
            "_gen_request_error": None,
            "_gen_latency_ms": latency_ms,
            "gen_raw": "",
        }

    try:
        env = LogicEnvironment(formal_language["parameters"], formal_language["translation"])
        parser = LogicParser(env)
        premises = [parser.parse(item) for item in formal_language["premise"] if str(item).strip()]
        query_results = [
            solve_query(
                env, parser, premises, query, timeout_ms,
                max_enumerate_assignments=max_enumerate_assignments,
                deadline=deadline,
            )
            for query in formal_language["question"]
        ]
        payload = [
            {
                "query": result.query,
                "query_type": result.query_type,
                "target": result.target,
                "status": result.status,
                "token": result.token,
                **({"targets": result.targets} if result.targets is not None else {}),
                **({"models": result.models} if result.models is not None else {}),
                **({"count": result.count} if result.count is not None else {}),
            }
            for result in query_results
        ]
        answer_tokens = [result.token for result in query_results]
        unknown_reasons = [
            result.status
            for result in query_results
            if result.token == "unknown" or result.status.startswith("unknown")
        ]
        z3_status = "unknown" if unknown_reasons else "success"
        z3_error = "; ".join(sorted(set(unknown_reasons))) if unknown_reasons else None
        fl_answer = "; ".join(answer_tokens)
        model_answer = tokens_to_model_answer(formal_language, payload, answer_tokens)
        parse_error = False
    except Exception as exc:
        payload = []
        answer_tokens = []
        z3_status = "parse_error"
        z3_error = str(exc)
        fl_answer = ""
        model_answer = ""
        parse_error = True

    latency_ms = int((time.time() - start) * 1000)
    return {
        "eval_id": f"{sample_id}:solve",
        "sample_id": sample_id,
        "prompt_type": "solve",
        "title": title,
        "question": question_text,
        "reference_answer": reference_answer,
        "answer_tokens": answer_tokens,
        "answer_payload": payload,
        "fl_answer": fl_answer,
        "nl_answer": model_answer,
        "model_answer": model_answer,
        "reason": (
            "Solved deterministically with Z3 from candidate FL.premise and FL.question."
            if z3_status == "success"
            else "Z3 returned unknown for at least one query."
        ) if not parse_error else None,
        "FL": formal_language,
        "z3_status": z3_status,
        "z3_error": z3_error,
        "_gen_parse_error": parse_error,
        "_gen_request_error": None,
        "_gen_latency_ms": latency_ms,
        "gen_raw": "",
    }


def run(
    input_path: Path,
    output_path: Path,
    *,
    data_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_samples: Optional[int] = None,
    z3_timeout_ms: int = 5000,
    max_enumerate_assignments: Optional[int] = 4096,
    record_timeout_s: Optional[float] = 60.0,
    progress_every: int = 10,
) -> List[Dict[str, Any]]:
    records = load_jsonl(input_path)
    if not records:
        raise SystemExit("no records found in input JSONL")

    for idx, item in enumerate(records):
        if not isinstance(item.get("id"), int):
            item["id"] = idx

    if start is not None or end is not None:
        lo = start if start is not None else 0
        hi = end if end is not None else len(records) - 1
        records = [item for item in records if lo <= item.get("id", 0) <= hi]

    if max_samples:
        records = records[:max_samples]

    data_map = load_data_map(str(data_path) if data_path else None)
    results: List[Dict[str, Any]] = []
    total = len(records)
    for pos, record in enumerate(records, 1):
        result = solve_record(
            record, data_map, z3_timeout_ms,
            max_enumerate_assignments=max_enumerate_assignments,
            record_timeout_s=record_timeout_s,
        )
        results.append(result)
        if progress_every > 0 and (pos == 1 or pos % progress_every == 0 or pos == total):
            print(
                "[z3_judge] "
                f"progress={pos}/{total} "
                f"sample_id={result.get('sample_id')} "
                f"status={result.get('z3_status')} "
                f"latency_ms={result.get('_gen_latency_ms')}",
                flush=True,
            )
    write_jsonl(output_path, results)
    if summary_path:
        write_json(summary_path, summarize(results))
    return results
