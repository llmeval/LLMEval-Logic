"""Bench- and rubric-record helpers shared by `code.fl_eval.rubric_judge`.

Pure stdlib — no LLM calls, no I/O beyond reading the bench JSON. The
release relies on the rubrics that ship under `bench/base/rubrics/`, so the
LLM-driven rubric-generation code that originally lived alongside these
helpers is intentionally not part of the public package.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io_utils import load_json_list


ALLOWED_GROUPS = {"logical_relation", "stated_constraint", "query_alignment"}
FORBIDDEN_GROUPS = {"semantic_unit", "given_fact", "task_equivalence", "answer_correctness"}
GROUP_PREFIX = {
    "logical_relation": "LR",
    "stated_constraint": "SC",
    "query_alignment": "QA",
}
ITEM_ID_PATTERN = re.compile(r"^(LR|SC|QA)[1-9][0-9]*$")
Z3_CHECK_GROUPS = {"logical_relation", "stated_constraint", "query_alignment"}
Z3_OMITTED_MARKER = "z3_check omitted"
COMPAT_REASON_REQUIRED_LABEL = "必须满足"
COMPAT_REASON_OPTIONAL_LABELS = ("可接受变体", "不可接受")


@dataclass
class ValidationIssue:
    line: int
    problem_id: Optional[Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"line": self.line, "problem_id": self.problem_id, "message": self.message}


def public_formalization(sample: Dict[str, Any]) -> Dict[str, Any]:
    formalization = sample.get("formalization") or {}
    if not isinstance(formalization, dict):
        formalization = {}
    return {
        "parameters": formalization.get("parameters") or {},
        "translation": formalization.get("translation") or {},
        "premise": formalization.get("premise") or [],
        "question": formalization.get("question") or [],
    }


def source_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    original = sample.get("original") or {}
    if not isinstance(original, dict):
        original = {}
    return {
        "id": sample.get("id"),
        "title": sample.get("title") or "",
        "original": {
            "background": original.get("background") or "",
            "question": original.get("question") or "",
        },
        "formalization": public_formalization(sample),
    }


def load_bench(path: Path) -> Dict[int, Dict[str, Any]]:
    data = load_json_list(path)
    bench: Dict[int, Dict[str, Any]] = {}
    for row_no, sample in enumerate(data, start=1):
        item_id = sample.get("id")
        if isinstance(item_id, int) and not isinstance(item_id, bool):
            bench[item_id] = sample
        else:
            raise ValueError(f"Bench row {row_no} is missing integer id")
    return bench


def _extract_reason_section(reason: str, label: str) -> Optional[str]:
    pattern = re.compile(
        rf"{re.escape(label)}[ \t]*[:：][ \t]*(.*?)(?=\n\s*(?:必须满足|可接受变体|不可接受)[ \t]*[:：]|$)",
        re.S,
    )
    match = pattern.search(reason)
    if match is None:
        return None
    return match.group(1).strip()


def _validate_compat_reason(
    reason: Any,
    *,
    path: str,
    line_no: int,
    problem_id: Optional[Any],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(reason, str) or not reason.strip():
        issues.append(ValidationIssue(line_no, problem_id, f"{path}.reason must be non-empty"))
        return issues

    must = _extract_reason_section(reason, COMPAT_REASON_REQUIRED_LABEL)
    if not must:
        issues.append(
            ValidationIssue(
                line_no,
                problem_id,
                f"{path}.reason must include non-empty '{COMPAT_REASON_REQUIRED_LABEL}: ...'",
            )
        )
    for label in COMPAT_REASON_OPTIONAL_LABELS:
        if re.search(rf"{re.escape(label)}[ \t]*[:：]", reason) and not _extract_reason_section(reason, label):
            issues.append(
                ValidationIssue(
                    line_no,
                    problem_id,
                    f"{path}.reason has empty optional section '{label}: ...'",
                )
            )
    return issues


def validate_rubric_record(
    record: Dict[str, Any],
    line_no: int,
    bench: Dict[int, Dict[str, Any]],
    *,
    require_compat_reason: bool = False,
) -> Tuple[List[ValidationIssue], Dict[str, int]]:
    issues: List[ValidationIssue] = []
    group_counts = {group: 0 for group in ALLOWED_GROUPS}
    item_id = record.get("id")
    if not isinstance(item_id, int) or isinstance(item_id, bool):
        issues.append(ValidationIssue(line_no, item_id, "id must be an integer"))
        bench_sample = None
    else:
        bench_sample = bench.get(item_id)
        if bench_sample is None:
            issues.append(ValidationIssue(line_no, item_id, "id not found in bench"))

    title = record.get("title")
    if not isinstance(title, str) or not title.strip():
        issues.append(ValidationIssue(line_no, item_id, "title must be a non-empty string"))
    elif bench_sample is not None and title != (bench_sample.get("title") or ""):
        issues.append(ValidationIssue(line_no, item_id, "title does not match bench title"))

    items = record.get("items")
    if not isinstance(items, list):
        issues.append(ValidationIssue(line_no, item_id, "items must be a list"))
        return issues, group_counts
    if len(items) < 3:
        issues.append(ValidationIssue(line_no, item_id, "items length must be at least 3"))

    seen_ids: set = set()
    for item_index, item in enumerate(items, start=1):
        path = f"items[{item_index}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue(line_no, item_id, f"{path} must be an object"))
            continue
        atom_id = item.get("id")
        group = item.get("group")
        desc = item.get("desc")
        score = item.get("score")
        reason = item.get("reason")
        z3_check = item.get("z3_check")

        if not isinstance(atom_id, str) or not ITEM_ID_PATTERN.match(atom_id):
            issues.append(ValidationIssue(line_no, item_id, f"{path}.id has invalid format"))
        elif atom_id in seen_ids:
            issues.append(ValidationIssue(line_no, item_id, f"{path}.id is duplicated"))
        else:
            seen_ids.add(atom_id)

        if group in FORBIDDEN_GROUPS:
            issues.append(ValidationIssue(line_no, item_id, f"{path}.group is forbidden: {group}"))
        if group not in ALLOWED_GROUPS:
            issues.append(ValidationIssue(line_no, item_id, f"{path}.group is not allowed"))
        else:
            group_counts[group] += 1
            expected_prefix = GROUP_PREFIX[group]
            if isinstance(atom_id, str) and not atom_id.startswith(expected_prefix):
                issues.append(
                    ValidationIssue(
                        line_no,
                        item_id,
                        f"{path}.id prefix must be {expected_prefix} for {group}",
                    )
                )
        if not isinstance(desc, str) or not desc.strip():
            issues.append(ValidationIssue(line_no, item_id, f"{path}.desc must be non-empty"))
        if require_compat_reason:
            issues.extend(
                _validate_compat_reason(
                    reason,
                    path=path,
                    line_no=line_no,
                    problem_id=item_id,
                )
            )
        if not isinstance(score, int) or isinstance(score, bool) or score != 0:
            issues.append(ValidationIssue(line_no, item_id, f"{path}.score must be integer 0"))

        if group in Z3_CHECK_GROUPS:
            if z3_check is None:
                if not (isinstance(reason, str) and Z3_OMITTED_MARKER in reason):
                    issues.append(
                        ValidationIssue(
                            line_no,
                            item_id,
                            f"{path}.z3_check missing and reason lacks marker '{Z3_OMITTED_MARKER}: ...'",
                        )
                    )
            elif not isinstance(z3_check, dict):
                issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check must be an object"))
            else:
                atoms = z3_check.get("atoms")
                mode = z3_check.get("mode")
                if not isinstance(atoms, list) or not atoms:
                    issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.atoms must be a non-empty list"))
                if group == "query_alignment":
                    if mode != "query_equiv":
                        issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.mode must be query_equiv for query_alignment"))
                    qt = z3_check.get("query_type")
                    target = z3_check.get("target")
                    if qt not in ("possible", "necessary", "enumerate_models", "count_models"):
                        issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.query_type must be one of possible/necessary/enumerate_models/count_models"))
                    if not isinstance(target, list) or not target:
                        issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.target must be a non-empty list"))
                else:
                    formula = z3_check.get("formula")
                    if not isinstance(formula, str) or not formula.strip():
                        issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.formula must be a non-empty string"))
                    if mode not in ("entails", "equiv", "literal_in_premise"):
                        issues.append(ValidationIssue(line_no, item_id, f"{path}.z3_check.mode must be entails / equiv / literal_in_premise"))

    qa_count = group_counts.get("query_alignment", 0)
    if qa_count != 1:
        issues.append(ValidationIssue(line_no, item_id, f"query_alignment must contain exactly 1 item (found {qa_count})"))

    return issues, group_counts


def parse_slice_indices(raw: Optional[str]) -> List[int]:
    if raw is None or not raw.strip():
        return []
    values: List[int] = []
    seen: set = set()
    for token in re.split(r"[\s,]+", raw.strip()):
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid descending range: {token}")
            expanded: Iterable[int] = range(start, end + 1)
        else:
            expanded = [int(token)]
        for value in expanded:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values
