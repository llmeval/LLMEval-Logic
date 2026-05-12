"""Rubric judge: score a candidate FL against a per-problem rubric.

User-facing entrypoint: `run_score_async`.

Pipeline:
  1. (fixedFL only) Premise-equivalence gate via z3 bidirectional entailment.
     - Equivalent → all LR/SC autopass.
     - Not equivalent + answer matches gold → LLM soft-review every diff.
     - Not equivalent + answer mismatch → inject PE failure items.
  2. Z3 prefilter: per-item entails (LR/SC) and query_equiv (QA1).
     - In freeFL mode, LR/SC are skipped here so the LLM judges over-strength.
  3. LLM scoring for items the prefilter did not auto-pass.

Output schema (JSONL): {problem_id, title, eval_id, passed, total, rate,
                       candidate_parse_error, items: [...], ...}
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Sequence, Tuple

from ...lib.api_client import (
    call_chat_async,
    extract_json,
    gather_with_progress,
    read_api_credentials,
)
from ...lib.bench_utils import load_bench, public_formalization
from ...lib.io_utils import load_json_list, load_jsonl, write_json, write_jsonl
from ...lib.prompts import extract_prompt_section, load_prompt
from .z3_check import (
    candidate_answers_match_gold,
    check_premise_equivalence,
    check_query_equiv,
    check_target,
    substitute_atoms,
)


# Only used as a default if neither --model on the CLI nor an explicit
# `model=` kwarg is supplied. The judge model is otherwise fully user-
# configurable via the CLI (or via the public ``run_score`` wrapper).
DEFAULT_RUBRIC_MODEL = "openai/gpt-4o"

Z3_CHECK_GROUPS = {"logical_relation", "stated_constraint", "query_alignment"}
Z3_CHECK_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def build_score_messages(
    sample: Dict[str, Any],
    candidate: Dict[str, Any],
    rubric: Dict[str, Any],
    prompt_md: str,
    skip_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    system = extract_prompt_section(prompt_md, "System Prompt")
    scoring_rules_raw = extract_prompt_section(prompt_md, "Scoring Rules")
    scoring_rules = [
        line.lstrip("- ").strip()
        for line in scoring_rules_raw.splitlines()
        if line.strip().startswith("-")
    ]
    original = sample.get("original") or {}
    items = rubric.get("items") or []
    if skip_ids:
        skip = set(skip_ids)
        items = [it for it in items if it.get("id") not in skip]
    payload = {
        "task": "Score the candidate formalization against the rubric.",
        "output_schema": {
            "problem_id": sample.get("id"),
            "title": sample.get("title"),
            "items": [
                {
                    "id": "LR1",
                    "group": "logical_relation",
                    "desc": "...",
                    "score": 0,
                    "judge_reason": "Chinese explanation for this score.",
                }
            ],
        },
        "scoring_rules": scoring_rules,
        "original_problem": {
            "background": original.get("background") or "",
            "question": original.get("question") or "",
        },
        "rubric": {
            "problem_id": rubric.get("id"),
            "title": rubric.get("title"),
            "items": items,
        },
        "candidate_formalization": candidate.get("FL"),
        "candidate_reason": candidate.get("reason") or "",
        "candidate_parse_error": candidate.get("_gen_parse_error"),
        "candidate_parse_error_detail": candidate.get("_gen_parse_error_detail"),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def validate_scored_items(
    rubric: Dict[str, Any],
    scored: Dict[str, Any],
    skip_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Normalize LLM-returned items against the full rubric.

    `skip_ids` lists item ids that were intentionally not sent to the LLM (they
    will be filled in later from an external source, e.g. Z3 auto-pass). Those
    items do not count against `ok` when missing from `scored`, and their
    placeholder score is set to 0 here (the caller will overwrite).
    """
    skip = set(skip_ids or [])
    rubric_items = rubric.get("items") or []
    scored_items = scored.get("items") if isinstance(scored, dict) else None
    if not isinstance(scored_items, list):
        scored_items = []
    by_id = {item.get("id"): item for item in scored_items if isinstance(item, dict)}
    normalized: List[Dict[str, Any]] = []
    ok = True
    for item in rubric_items:
        item_id = item.get("id")
        if item_id in skip:
            normalized.append(
                {
                    "id": item_id,
                    "group": item.get("group"),
                    "desc": item.get("desc"),
                    "score": 0,
                    "rubric_reason": item.get("reason"),
                    "judge_reason": "",
                }
            )
            continue
        scored_item = by_id.get(item_id) or {}
        if not scored_item:
            ok = False
        score = scored_item.get("score")
        if score not in (0, 1):
            score = 0
            ok = False
        normalized.append(
            {
                "id": item_id,
                "group": item.get("group"),
                "desc": item.get("desc"),
                "score": score,
                "rubric_reason": item.get("reason"),
                "judge_reason": scored_item.get("judge_reason") or scored_item.get("reason") or "",
            }
        )
    return normalized, ok


def summarize_scores(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_items = 0
    passed_items = 0
    by_group: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, int] = {}
    auto_pass_count = 0
    auto_pass_by_group: Dict[str, int] = {}
    problem_rows = []
    for record in records:
        items = record.get("items") or []
        problem_total = len(items)
        problem_passed = sum(1 for item in items if item.get("score") == 1)
        total_items += problem_total
        passed_items += problem_passed
        for item in items:
            group = item.get("group") or "unknown"
            stats = by_group.setdefault(group, {"total": 0, "passed": 0, "rate": 0.0})
            stats["total"] += 1
            if item.get("score") == 1:
                stats["passed"] += 1
            source = item.get("source") or "llm"
            by_source[source] = by_source.get(source, 0) + 1
            if item.get("score") == 1 and source != "llm":
                auto_pass_count += 1
                auto_pass_by_group[group] = auto_pass_by_group.get(group, 0) + 1
        problem_rows.append(
            {
                "problem_id": record.get("problem_id"),
                "title": record.get("title"),
                "passed": problem_passed,
                "total": problem_total,
                "rate": (problem_passed / problem_total) if problem_total else 0.0,
                "parse_error": record.get("candidate_parse_error"),
                "parse_error_detail": record.get("candidate_parse_error_detail"),
            }
        )
    for stats in by_group.values():
        stats["rate"] = (stats["passed"] / stats["total"]) if stats["total"] else 0.0
    return {
        "total_records": len(records),
        "total_items": total_items,
        "passed_items": passed_items,
        "overall_rate": (passed_items / total_items) if total_items else 0.0,
        "by_group": by_group,
        "by_source": by_source,
        "auto_pass_count": auto_pass_count,
        "auto_pass_by_group": auto_pass_by_group,
        "by_problem": problem_rows,
    }


def _normalize_gloss(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    out = text.strip()
    # full-width to half-width for common punctuation
    table = str.maketrans("（），。；：、！？", "(),.;:,!?")
    out = out.translate(table)
    # collapse internal whitespace
    out = re.sub(r"\s+", "", out)
    return out


def _normalize_gloss_loose(text: Any) -> str:
    out = _normalize_gloss(text).lower()
    out = re.sub(r"[\.\,\;\:\!\?\-\_\(\)\[\]\{\}'\"`~/\\]+", "", out)
    return out


def _candidate_gloss_index(candidate_fl: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build two NL-gloss → candidate-symbol indices (exact and loose).

    Symbol value is the entry's key in the candidate `translation` dict (this
    may be a bare symbol like `M` or a predicate template like `Drinks(o)`,
    which the Z3 parser already handles).
    """
    translation = candidate_fl.get("translation") or {}
    if not isinstance(translation, dict):
        return {}, {}
    exact: Dict[str, str] = {}
    loose: Dict[str, str] = {}
    for symbol, gloss in translation.items():
        if not isinstance(symbol, str):
            continue
        e = _normalize_gloss(gloss)
        if e and e not in exact:
            exact[e] = symbol
        l = _normalize_gloss_loose(gloss)
        if l and l not in loose:
            loose[l] = symbol
    return exact, loose


def _try_string_alignment(
    z3_check: Dict[str, Any], candidate_fl: Dict[str, Any]
) -> Tuple[Optional[Dict[str, str]], str]:
    """Match each atom NL gloss to a candidate symbol via string normalization.

    Returns (mapping, source_tag) when fully aligned, else (None, '').
    `source_tag` is `z3:exact` for the strict pass and `z3:normalized` for the
    loose pass.
    """
    atoms = z3_check.get("atoms") or []
    if not isinstance(atoms, list) or not atoms:
        return None, ""
    exact_idx, loose_idx = _candidate_gloss_index(candidate_fl)

    mapping: Dict[str, str] = {}
    for atom in atoms:
        if not isinstance(atom, dict):
            return None, ""
        key = atom.get("key")
        nl = atom.get("nl")
        if not isinstance(key, str) or not isinstance(nl, str):
            return None, ""
        symbol = exact_idx.get(_normalize_gloss(nl))
        if symbol is None:
            mapping = {}
            break
        mapping[key] = symbol
    if mapping:
        return mapping, "z3:exact"

    mapping = {}
    for atom in atoms:
        key = atom.get("key")
        nl = atom.get("nl")
        symbol = loose_idx.get(_normalize_gloss_loose(nl))
        if symbol is None:
            return None, ""
        mapping[key] = symbol
    return mapping, "z3:normalized"


async def _try_llm_alignment(
    z3_check: Dict[str, Any],
    candidate_fl: Dict[str, Any],
    align_prompt_md: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Ask the LLM to align atoms and emit a translated formula.

    Returns (translated_formula, raw_payload). On any failure returns (None, payload).
    """
    system = extract_prompt_section(align_prompt_md, "System Prompt")
    user_template = extract_prompt_section(align_prompt_md, "User Prompt Template")
    cand_summary = {
        "parameters": candidate_fl.get("parameters") or {},
        "translation": candidate_fl.get("translation") or {},
        "premise": candidate_fl.get("premise") or [],
    }
    user = (
        user_template
        .replace("<CANDIDATE_FL_JSON>", json.dumps(cand_summary, ensure_ascii=False, indent=2))
        .replace("<Z3_CHECK_JSON>", json.dumps(z3_check, ensure_ascii=False, indent=2))
    )
    api_result = await call_chat_async(
        endpoint, api_key, model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, timeout=timeout, use_json_mode=True,
    )
    if api_result.error:
        return None, {"error": api_result.error}
    parsed = extract_json(api_result.content_text)
    if not isinstance(parsed, dict):
        return None, {"error": "parse_error", "raw": api_result.content_text[:500]}
    if (parsed.get("confidence") or "").lower() == "low":
        return None, parsed
    formula = parsed.get("translated_formula")
    if not isinstance(formula, str) or not formula.strip():
        return None, parsed
    return formula, parsed


async def _z3_prefilter_one(
    rubric_items: List[Dict[str, Any]],
    candidate_fl: Optional[Dict[str, Any]],
    align_prompt_md: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
    enable_llm_align: bool,
    gold_fl: Optional[Dict[str, Any]] = None,
    skip_lr_sc: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """For each LR/SC/QA item carrying a `z3_check`, try to auto-pass via Z3.

    Returns a dict keyed by item id with payload {passed, source, reason, ...}.
    Only items that AUTO-PASS are returned; items that fail Z3 fall through to
    LLM judging silently (Z3 fail does not imply true fail).

    QA1 items use mode=query_equiv and require `gold_fl` to compute the gold
    answer set for comparison.

    When `skip_lr_sc` is True, LR/SC items are skipped entirely (caller has
    already decided they pass via a global premise-equivalence gate, or freeFL
    mode wants the LLM to judge over-strengthening). QA1 items are still
    processed.
    """
    if not isinstance(candidate_fl, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in rubric_items:
        if not isinstance(item, dict):
            continue
        group = item.get("group")
        if group not in Z3_CHECK_GROUPS:
            continue
        if skip_lr_sc and group in ("logical_relation", "stated_constraint"):
            continue
        z3_check = item.get("z3_check")
        if not isinstance(z3_check, dict):
            continue
        atoms = z3_check.get("atoms") or []
        mode = (z3_check.get("mode") or "entails").lower()

        if mode == "query_equiv":
            if not isinstance(gold_fl, dict):
                continue
            payload = await _z3_query_equiv_one(
                z3_check, candidate_fl, gold_fl, align_prompt_md,
                endpoint, api_key, model, timeout, enable_llm_align,
            )
            if payload is not None:
                out[item.get("id")] = payload
            continue

        # Default LR/SC path: entails / equiv / literal_in_premise
        formula = z3_check.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            continue

        translated_formula: Optional[str] = None
        source_tag: Optional[str] = None
        align_payload: Optional[Dict[str, Any]] = None

        mapping, src = _try_string_alignment(z3_check, candidate_fl)
        if mapping is not None:
            translated_formula = substitute_atoms(formula, mapping)
            source_tag = src

        if translated_formula is None and enable_llm_align and atoms:
            llm_formula, align_payload = await _try_llm_alignment(
                z3_check, candidate_fl, align_prompt_md,
                endpoint, api_key, model, timeout,
            )
            if llm_formula is not None:
                translated_formula = llm_formula
                source_tag = "z3:llm_align"

        if translated_formula is None:
            continue

        result = check_target(candidate_fl, translated_formula, mode=mode, timeout_ms=Z3_CHECK_TIMEOUT_MS)
        if result.passed:
            payload: Dict[str, Any] = {
                "passed": True,
                "source": source_tag,
                "mode": mode,
                "translated_formula": translated_formula,
                "z3_reason": result.reason,
            }
            if align_payload is not None:
                payload["align"] = align_payload
            out[item.get("id")] = payload
    return out


async def _z3_query_equiv_one(
    z3_check: Dict[str, Any],
    candidate_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    align_prompt_md: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
    enable_llm_align: bool,
) -> Optional[Dict[str, Any]]:
    """Run query_equiv check for a QA item. Returns payload if auto-pass, else None."""
    query_type = z3_check.get("query_type")
    target = z3_check.get("target")
    if not isinstance(query_type, str) or not isinstance(target, list) or not target:
        return None

    mapping, src = _try_string_alignment(z3_check, candidate_fl)
    align_payload: Optional[Dict[str, Any]] = None
    source_tag: Optional[str] = src

    if mapping is None and enable_llm_align and z3_check.get("atoms"):
        # For QA, LLM alignment produces candidate-side symbol mapping (no formula needed,
        # we'll substitute target ourselves). Reuse _try_llm_alignment which returns a
        # full translated formula — we only want the alignment dict here.
        llm_formula, align_payload = await _try_llm_alignment(
            z3_check, candidate_fl, align_prompt_md,
            endpoint, api_key, model, timeout,
        )
        if isinstance(align_payload, dict) and isinstance(align_payload.get("alignment"), dict):
            mapping = {k: str(v) for k, v in align_payload["alignment"].items() if isinstance(v, str)}
            source_tag = "z3:llm_align"

    if mapping is None:
        return None

    # Translate target to candidate symbol space.
    cand_target = [substitute_atoms(t, mapping) if isinstance(t, str) else t for t in target]

    result = check_query_equiv(
        candidate_fl=candidate_fl,
        gold_fl=gold_fl,
        query_type=query_type,
        target=cand_target,
        gold_target=list(target),
        timeout_ms=Z3_CHECK_TIMEOUT_MS,
    )
    if not result.passed:
        return None
    payload: Dict[str, Any] = {
        "passed": True,
        "source": source_tag,
        "mode": "query_equiv",
        "translated_formula": str(cand_target),
        "z3_reason": result.reason,
    }
    if align_payload is not None:
        payload["align"] = align_payload
    return payload


def _is_fixed_fl(cand_fl: Optional[Dict[str, Any]], gold_fl: Dict[str, Any]) -> bool:
    """Detect fixedFL mode: candidate reuses gold's parameters and translation."""
    if not isinstance(cand_fl, dict):
        return False
    return (cand_fl.get("parameters") == gold_fl.get("parameters")
            and cand_fl.get("translation") == gold_fl.get("translation"))


def _build_pe_failure_items(pe_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a non-equivalent premise_equivalence result into score items.

    Returns a list with one summary `PE1` item plus per-difference items
    `PE_extra_<idx>` and `PE_missing_<idx>`. All items have score=0 and
    source="z3:premise_equiv".
    """
    items: List[Dict[str, Any]] = [{
        "id": "PE1",
        "group": "logical_relation",
        "desc": "候选 premise 集合应与 gold premise 集合在公共原子下语义等价",
        "score": 0,
        "rubric_reason": "fixedFL 模式下，候选 premise 整体应与 gold 等价；引入额外约束或缺失关键约束都会破坏等价。",
        "judge_reason": "z3 双向蕴含检查发现 conj(cand) 与 conj(gold) 不等价",
        "source": "z3:premise_equiv",
    }]
    for idx, raw in pe_result.get("extras", []) or []:
        items.append({
            "id": f"PE_extra_{idx}",
            "group": "logical_relation",
            "desc": "候选不应引入 gold 没有的额外约束",
            "score": 0,
            "rubric_reason": "候选 premise 中存在 gold 不蕴含的约束，等价于在 gold 之外又加了一条独立断言。",
            "judge_reason": f"候选引入 gold 未蕴含的约束: {raw}",
            "source": "z3:premise_equiv",
        })
    for idx, raw in pe_result.get("missing", []) or []:
        items.append({
            "id": f"PE_missing_{idx}",
            "group": "logical_relation",
            "desc": "候选不应缺失 gold 中的关键约束",
            "score": 0,
            "rubric_reason": "gold premise 中存在候选不蕴含的约束，说明候选丢失了一条关键事实或推理规则。",
            "judge_reason": f"候选未蕴含 gold 约束: {raw}",
            "source": "z3:premise_equiv",
        })
    return items


def _build_pe_review_rubric_context(rubric: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract rubric desc/reason text for PE soft review.

    The PE soft reviewer decides premise-set differences, so it needs the
    human-facing rubric interpretation, not the item's z3_check formula.
    """
    context: List[Dict[str, Any]] = []
    for item in rubric.get("items") or []:
        if not isinstance(item, dict):
            continue
        group = item.get("group")
        if group not in ("logical_relation", "stated_constraint", "query_alignment"):
            continue
        context.append(
            {
                "id": item.get("id"),
                "group": group,
                "desc": item.get("desc") or "",
                "reason": item.get("reason") or "",
            }
        )
    return context


async def _llm_review_one_diff(
    *,
    is_extra: bool,
    raw_premise: str,
    sample: Dict[str, Any],
    cand_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    rubric_context: Optional[Sequence[Dict[str, Any]]],
    pe_review_prompt_md: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    """Call LLM to soft-review a single PE difference. Returns
    `{'softpass': bool, 'reason': str}`. On any failure returns softpass=False.
    """
    import time as _time
    system = extract_prompt_section(pe_review_prompt_md, "System Prompt")
    user_template = extract_prompt_section(
        pe_review_prompt_md,
        "User Prompt Extras" if is_extra else "User Prompt Missing",
    )
    background = (sample.get("original") or {}).get("background") or ""
    question = (sample.get("original") or {}).get("question") or ""
    translation = json.dumps(gold_fl.get("translation") or {}, ensure_ascii=False, indent=2)
    gold_premises = json.dumps(gold_fl.get("premise") or [], ensure_ascii=False, indent=2)
    cand_premises = json.dumps(cand_fl.get("premise") or [], ensure_ascii=False, indent=2)
    rubric_context_json = json.dumps(rubric_context or [], ensure_ascii=False, indent=2)

    if is_extra:
        user = (
            user_template
            .replace("<ORIGINAL_BACKGROUND>", background)
            .replace("<ORIGINAL_QUESTION>", question)
            .replace("<RUBRIC_CONTEXT_JSON>", rubric_context_json)
            .replace("<TRANSLATION_DICT>", translation)
            .replace("<GOLD_PREMISES_LIST>", gold_premises)
            .replace("<EXTRA_PREMISE_RAW>", raw_premise)
        )
    else:
        user = (
            user_template
            .replace("<ORIGINAL_BACKGROUND>", background)
            .replace("<ORIGINAL_QUESTION>", question)
            .replace("<RUBRIC_CONTEXT_JSON>", rubric_context_json)
            .replace("<TRANSLATION_DICT>", translation)
            .replace("<MISSING_PREMISE_RAW>", raw_premise)
            .replace("<CAND_PREMISES_LIST>", cand_premises)
        )
    start = _time.time()
    api_result = await call_chat_async(
        endpoint, api_key, model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, timeout=timeout, use_json_mode=True,
    )
    latency = int((_time.time() - start) * 1000)
    if api_result.error:
        return {"softpass": False, "reason": "review error", "error": api_result.error, "latency_ms": latency}
    parsed = extract_json(api_result.content_text)
    if not isinstance(parsed, dict):
        return {"softpass": False, "reason": "parse error", "error": "parse_error", "latency_ms": latency}
    reason = str(parsed.get("reason") or "").strip()
    if not reason:
        reason = "(无理由)" if bool(parsed.get("softpass")) else "(未给出理由)"
    return {
        "softpass": bool(parsed.get("softpass")),
        "reason": reason,
        "latency_ms": latency,
    }


async def _llm_review_pe_diffs(
    *,
    pe_result: Dict[str, Any],
    sample: Dict[str, Any],
    cand_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    rubric_context: Optional[Sequence[Dict[str, Any]]],
    pe_review_prompt_md: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    """Concurrently soft-review every PE_extra / PE_missing difference.

    Returns:
        {
          'all_softpass': bool,
          'extras_review':  [(idx, raw, {softpass, reason, ...}), ...],
          'missing_review': [(idx, raw, {softpass, reason, ...}), ...],
        }
    """
    extras = list(pe_result.get("extras") or [])
    missing = list(pe_result.get("missing") or [])
    extra_tasks = [
        _llm_review_one_diff(
            is_extra=True, raw_premise=raw, sample=sample,
            cand_fl=cand_fl, gold_fl=gold_fl,
            rubric_context=rubric_context,
            pe_review_prompt_md=pe_review_prompt_md,
            endpoint=endpoint, api_key=api_key, model=model, timeout=timeout,
        )
        for _, raw in extras
    ]
    missing_tasks = [
        _llm_review_one_diff(
            is_extra=False, raw_premise=raw, sample=sample,
            cand_fl=cand_fl, gold_fl=gold_fl,
            rubric_context=rubric_context,
            pe_review_prompt_md=pe_review_prompt_md,
            endpoint=endpoint, api_key=api_key, model=model, timeout=timeout,
        )
        for _, raw in missing
    ]
    extras_results = await asyncio.gather(*extra_tasks) if extra_tasks else []
    missing_results = await asyncio.gather(*missing_tasks) if missing_tasks else []
    extras_review = [(idx, raw, res) for (idx, raw), res in zip(extras, extras_results)]
    missing_review = [(idx, raw, res) for (idx, raw), res in zip(missing, missing_results)]
    all_pass = all(r["softpass"] for r in extras_results) and all(r["softpass"] for r in missing_results)
    return {
        "all_softpass": all_pass,
        "extras_review": extras_review,
        "missing_review": missing_review,
    }


def _build_pe_failure_items_with_review(
    pe_result: Dict[str, Any], review: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Like `_build_pe_failure_items` but uses LLM-review verdicts to set per-item score.

    The summary `PE1` item's score is 1 only if ALL diffs softpass.
    """
    extras_review = review.get("extras_review") or []
    missing_review = review.get("missing_review") or []
    all_pass = bool(review.get("all_softpass"))
    items: List[Dict[str, Any]] = [{
        "id": "PE1",
        "group": "logical_relation",
        "desc": "候选 premise 集合应与 gold premise 集合在公共原子下语义等价",
        "score": 1 if all_pass else 0,
        "rubric_reason": "fixedFL 模式下，候选 premise 整体应与 gold 等价；引入额外约束或缺失关键约束都会破坏等价。",
        "judge_reason": (
            "PE 不等价但 candidate 答案与 gold 一致 + LLM 软复审全部通过"
            if all_pass else
            "z3 双向蕴含检查发现 conj(cand) 与 conj(gold) 不等价；LLM 软复审未全部通过"
        ),
        "source": "z3:premise_equiv:soft" if all_pass else "z3:premise_equiv",
    }]
    for idx, raw, res in extras_review:
        passed = bool(res.get("softpass"))
        items.append({
            "id": f"PE_extra_{idx}",
            "group": "logical_relation",
            "desc": "候选不应引入 gold 没有的额外约束（除非题面隐含 / 合理填充）",
            "score": 1 if passed else 0,
            "rubric_reason": "候选 premise 中存在 gold 不蕴含的约束；只有在题面 NL 明确支持或合理隐含时才豁免。",
            "judge_reason": (f"LLM 软复审通过：{res.get('reason')}" if passed else f"候选引入 gold 未蕴含的约束: {raw}"),
            "source": "z3:premise_equiv:soft" if passed else "z3:premise_equiv",
            "llm_review": res,
        })
    for idx, raw, res in missing_review:
        passed = bool(res.get("softpass"))
        items.append({
            "id": f"PE_missing_{idx}",
            "group": "logical_relation",
            "desc": "候选不应缺失 gold 中的关键约束（除非以等价形式表达）",
            "score": 1 if passed else 0,
            "rubric_reason": "gold premise 中存在候选不蕴含的约束；只有在候选有等价或可接受的替代形式时才豁免。",
            "judge_reason": (f"LLM 软复审通过：{res.get('reason')}" if passed else f"候选未蕴含 gold 约束: {raw}"),
            "source": "z3:premise_equiv:soft" if passed else "z3:premise_equiv",
            "llm_review": res,
        })
    return items


async def run_score_async(
    *,
    bench_path: Path,
    candidates_path: Path,
    rubric_path: Optional[Path],
    score_output_path: Path,
    score_summary_path: Path,
    reuse_rubric_path: Optional[Path] = None,
    reuse_rubric_dir: Optional[Path] = None,
    rubric_summary_path: Optional[Path] = None,
    model: str = DEFAULT_RUBRIC_MODEL,
    timeout: float = 180.0,
    concurrency: int = 5,
    enable_z3_prefilter: bool = True,
    enable_llm_align: bool = True,
    enable_premise_equiv: bool = True,
    enable_pe_soft_review: bool = True,
    enable_freefl_no_z3_lr_sc: bool = True,
) -> Dict[str, Any]:
    endpoint, api_key = read_api_credentials()
    bench = load_bench(bench_path)
    candidates = load_jsonl(candidates_path)
    sample_ids: List[int] = [record.get("sample_id") for record in candidates]

    if reuse_rubric_path is None and reuse_rubric_dir is None:
        raise SystemExit(
            "rubric source required: pass either reuse_rubric_dir "
            "(directory of {id:03d}.json files, e.g. bench/base/rubrics) or "
            "reuse_rubric_path (a JSONL file)."
        )
    if reuse_rubric_path is not None and reuse_rubric_dir is not None:
        raise SystemExit("Pass only one of reuse_rubric_path / reuse_rubric_dir")
    if reuse_rubric_dir is not None:
        existing_by_id: Dict[int, Dict[str, Any]] = {}
        for sid in sample_ids:
            path = reuse_rubric_dir / f"{int(sid):03d}.json"
            if not path.exists():
                raise SystemExit(f"Per-problem rubric missing: {path}")
            with path.open("r", encoding="utf-8") as fh:
                existing_by_id[sid] = json.load(fh)
        source_label = str(reuse_rubric_dir)
    else:
        existing = load_jsonl(reuse_rubric_path)
        existing_by_id = {
            (record.get("id") if record.get("id") is not None else record.get("problem_id")): record
            for record in existing
        }
        source_label = str(reuse_rubric_path)
    missing = [sid for sid in sample_ids if sid not in existing_by_id]
    if missing:
        raise SystemExit(f"Reused rubric is missing id(s): {missing}")
    rubrics = [existing_by_id[sid] for sid in sample_ids]
    if rubric_path is not None:
        write_jsonl(rubric_path, rubrics)
    if rubric_summary_path is not None:
        write_json(
            rubric_summary_path,
            {
                "model": "reused",
                "source": source_label,
                "total_requested": len(sample_ids),
                "total_ok": len(rubrics),
                "total_errors": 0,
                "errors": [],
                "output": str(rubric_path) if rubric_path else None,
            },
        )
    rubric_errors: List[Dict[str, Any]] = []

    prompt_md = load_prompt("rubric_score.md", caller_file=__file__)
    align_prompt_md = load_prompt("rubric_z3_align.md", caller_file=__file__) if enable_z3_prefilter else ""
    pe_review_prompt_md = load_prompt("rubric_pe_review.md", caller_file=__file__) if enable_pe_soft_review else ""
    rubric_by_id = {
        (record.get("id") if record.get("id") is not None else record.get("problem_id")): record
        for record in rubrics
    }
    candidate_by_id = {record.get("sample_id"): record for record in candidates}

    # Resume: read any existing records in score_output_path and skip those
    # sample_ids on this run. Records are streamed append-only per-sample.
    # Only records with items are considered complete; empty/errored records
    # will be retried on resume.
    score_output_path.parent.mkdir(parents=True, exist_ok=True)
    done_by_id: Dict[int, Dict[str, Any]] = {}
    if score_output_path.exists():
        try:
            all_recs = list(load_jsonl(score_output_path))
        except Exception:
            all_recs = []
            score_output_path.unlink(missing_ok=True)
        kept: List[Dict[str, Any]] = []
        for rec in all_recs:
            pid = rec.get("problem_id")
            if not isinstance(pid, int) or isinstance(pid, bool):
                continue
            if rec.get("total", 0) > 0:
                done_by_id[pid] = rec
                kept.append(rec)
        # Rewrite the file with only complete records so append-stream stays tidy.
        if all_recs and len(kept) != len(all_recs):
            write_jsonl(score_output_path, kept)
    pending_ids = [sid for sid in sample_ids if sid not in done_by_id]
    if done_by_id:
        print(
            f"Resume: found {len(done_by_id)} complete records; scoring {len(pending_ids)} remaining.",
            flush=True,
        )

    sem = asyncio.Semaphore(concurrency)
    score_errors: List[Dict[str, Any]] = []
    completed = 0
    write_lock = asyncio.Lock()

    async def _score_one(sample_id: int) -> None:
        nonlocal completed
        sample = bench.get(sample_id)
        rubric = rubric_by_id.get(sample_id)
        candidate = candidate_by_id.get(sample_id)
        if sample is None or rubric is None or candidate is None:
            return
        async with sem:
            cand_fl = candidate.get("FL") if isinstance(candidate.get("FL"), dict) else None
            gold_fl = public_formalization(sample)

            # Premise equivalence gate (fixedFL only).
            pe_result: Optional[Dict[str, Any]] = None
            pe_equiv: bool = False
            if enable_premise_equiv and _is_fixed_fl(cand_fl, gold_fl):
                try:
                    pe_result = check_premise_equivalence(cand_fl or {}, gold_fl)
                except Exception as exc:
                    pe_result = {"equivalent": None, "extras": [], "missing": [], "error": f"exception: {exc}"}
                pe_equiv = pe_result.get("equivalent") is True

            # Z3 prefilter (skips LR/SC items when PE already declared them equiv,
            # and in freeFL mode to let LLM judge over-strengthening).
            is_fixed = _is_fixed_fl(cand_fl, gold_fl)
            skip_lr_sc = pe_equiv or (enable_freefl_no_z3_lr_sc and not is_fixed)
            autopass: Dict[str, Dict[str, Any]] = {}
            if enable_z3_prefilter:
                autopass = await _z3_prefilter_one(
                    rubric.get("items") or [],
                    cand_fl,
                    align_prompt_md,
                    endpoint, api_key, model, timeout, enable_llm_align,
                    gold_fl=gold_fl,
                    skip_lr_sc=skip_lr_sc,
                )

            # When PE finds the premise sets equivalent, mark every LR/SC item
            # as auto-passed via the global gate (overrides per-item entails).
            if pe_equiv:
                for item in (rubric.get("items") or []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("group") in ("logical_relation", "stated_constraint"):
                        autopass[item.get("id")] = {
                            "passed": True,
                            "source": "z3:premise_equiv",
                            "z3_reason": "整体 premise 集合双向等价（fixedFL）",
                            "mode": "premise_equiv",
                        }

            # LLM scoring for the remaining items only.
            skip_ids = {iid for iid, p in autopass.items() if p.get("passed")}
            remaining = [it for it in (rubric.get("items") or []) if it.get("id") not in skip_ids]

            import time as _time
            if remaining:
                messages = build_score_messages(sample, candidate, rubric, prompt_md, skip_ids=skip_ids)
                start = _time.time()
                api_result = await call_chat_async(
                    endpoint, api_key, model, messages,
                    temperature=0.0, timeout=timeout, use_json_mode=True,
                )
                latency_ms = int((_time.time() - start) * 1000)
                if api_result.error:
                    score_errors.append({"problem_id": sample_id, "error": api_result.error, "latency_ms": latency_ms})
                    parsed = None
                else:
                    parsed = extract_json(api_result.content_text)
                    if not isinstance(parsed, dict):
                        score_errors.append({"problem_id": sample_id, "error": "parse_error", "latency_ms": latency_ms})
                        parsed = None
            else:
                parsed = {"items": []}
                latency_ms = 0

            items, score_ok = validate_scored_items(rubric, parsed or {}, skip_ids=skip_ids)
            for item in items:
                payload = autopass.get(item.get("id"))
                if payload and payload.get("passed"):
                    item["score"] = 1
                    item["source"] = payload.get("source") or "z3"
                    item["z3_reason"] = payload.get("z3_reason")
                    item["z3_translated_formula"] = payload.get("translated_formula")
                    item["z3_mode"] = payload.get("mode")
                    if "align" in payload:
                        item["z3_align"] = payload["align"]
                else:
                    item.setdefault("source", "llm")
            # Append PE failure items when premise sets are not equivalent.
            if pe_result is not None and pe_result.get("equivalent") is False:
                # Optional LLM soft-review gate: only when candidate's z3-solved
                # answer matches gold for every gold question, ask LLM whether
                # each individual diff is a defensible formalization choice.
                ans_match = False
                if enable_pe_soft_review and isinstance(cand_fl, dict):
                    try:
                        ans_match = candidate_answers_match_gold(cand_fl, gold_fl)
                    except Exception:
                        ans_match = False
                if ans_match:
                    review = await _llm_review_pe_diffs(
                        pe_result=pe_result, sample=sample,
                        cand_fl=cand_fl, gold_fl=gold_fl,
                        rubric_context=_build_pe_review_rubric_context(rubric),
                        pe_review_prompt_md=pe_review_prompt_md,
                        endpoint=endpoint, api_key=api_key, model=model, timeout=timeout,
                    )
                    pe_items_to_inject = _build_pe_failure_items_with_review(pe_result, review)
                    # If all diffs softpass, also flip the existing LR/SC items
                    # (which currently carry source='llm' or 'z3:exact' fail) up
                    # to a soft pass. This is the "answer correct + all diffs
                    # defensible" case where the candidate FL is acceptable.
                    if review.get("all_softpass"):
                        for it in items:
                            if it.get("group") in ("logical_relation", "stated_constraint") and it.get("score") == 0:
                                it["score"] = 1
                                it["source"] = "z3:premise_equiv:soft"
                                it["pe_soft_reason"] = "PE 不等价但 candidate 答案与 gold 一致 + LLM 软复审全部通过"
                    items.extend(pe_items_to_inject)
                else:
                    items.extend(_build_pe_failure_items(pe_result))
            total = len(items)
            passed = sum(1 for item in items if item.get("score") == 1)
            record = {
                "problem_id": sample_id,
                "title": sample.get("title"),
                "eval_id": candidate.get("eval_id"),
                "candidate_parse_error": candidate.get("_gen_parse_error"),
                "candidate_parse_error_detail": candidate.get("_gen_parse_error_detail"),
                "score_parse_ok": score_ok,
                "score_latency_ms": latency_ms,
                "score_model": model,
                "passed": passed,
                "total": total,
                "rate": (passed / total) if total else 0.0,
                "items": items,
            }

        async with write_lock:
            with score_output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_by_id[sample_id] = record
            completed += 1
            print(f"Score {completed}/{len(pending_ids)} (pid={sample_id})", flush=True)

    await asyncio.gather(*(_score_one(sid) for sid in pending_ids))

    # Load the full stream (existing + newly written) and assemble summary in
    # the original sample_ids order.
    all_by_id = {rec.get("problem_id"): rec for rec in load_jsonl(score_output_path)}
    scored_records = [all_by_id[sid] for sid in sample_ids if sid in all_by_id]

    summary = summarize_scores(scored_records)
    summary.update(
        {
            "model": model,
            "rubric_input": str(reuse_rubric_dir or reuse_rubric_path or rubric_path),
            "rubric_reused": reuse_rubric_path is not None or reuse_rubric_dir is not None,
            "candidate_input": str(candidates_path),
            "score_output": str(score_output_path),
            "score_errors": score_errors,
            "z3_prefilter_enabled": enable_z3_prefilter,
            "llm_align_enabled": enable_llm_align,
            "resumed_from_existing": len(done_by_id) - len(pending_ids) if len(done_by_id) >= len(pending_ids) else 0,
        }
    )
    write_json(score_summary_path, summary)
    return summary


def run_score(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(run_score_async(**kwargs))
