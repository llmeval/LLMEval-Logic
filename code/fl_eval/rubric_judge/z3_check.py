"""Z3-based satisfaction check for rubric `z3_check` items.

Used by `code.fl_eval.rubric_judge.core` as a fast path before falling back to LLM
judgement on `logical_relation`, `stated_constraint`, and `query_alignment`
rubric items.

For LR/SC items:
  The item carries `atoms` (NL gloss per placeholder) + `formula` written in
  those placeholders + `mode`. The caller substitutes placeholders into the
  candidate FL's actual symbols (see `substitute_atoms`) and calls
  `check_target` with the resulting candidate-symbol formula string.

For QA1 items (mode=query_equiv):
  The item carries `atoms` + `query_type` + `target`. The caller aligns atoms,
  substitutes placeholders, and calls `check_query_equiv` with both candidate
  and gold FL to verify that running gold's query against candidate's premises
  produces gold's expected answer (under projection onto gold target).

Modes:
- `entails`: candidate.premise conjunction implies target.
- `equiv`: some candidate.premise is logically equivalent to target.
- `literal_in_premise`: target string matches a candidate premise after operator
  normalization.
- `query_equiv`: candidate's query (after symbol alignment) is answer-equivalent
  to gold's query, i.e. running gold-style query on candidate FL produces the
  same set/cardinality/possible/necessary verdict as running gold's query on
  gold FL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from z3 import And, Implies, Not, Solver, Z3Exception, sat, unsat

from ...lib.z3_engine import (
    FormulaParseError,
    LogicEnvironment,
    LogicParser,
    check_necessary,
    check_possible,
    enumerate_bool_models,
    normalize_operator,
    parse_query,
    tokenize,
)


DEFAULT_TIMEOUT_MS = 5000
SUPPORTED_QUERY_TYPES = ("possible", "necessary", "enumerate_models", "count_models")
PARSE_EXCEPTIONS = (FormulaParseError, Z3Exception, TypeError, ValueError)


@dataclass
class CheckResult:
    passed: bool
    reason: str
    mode: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "mode": self.mode,
            "error": self.error,
        }


def _build_env(candidate_fl: Dict[str, Any]) -> LogicEnvironment:
    parameters = candidate_fl.get("parameters") or {}
    translation = candidate_fl.get("translation") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    if not isinstance(translation, dict):
        translation = {}
    return LogicEnvironment(parameters=parameters, translation=translation)


def _parse_premises(parser: LogicParser, premises: Sequence[Any]) -> List[Any]:
    parsed = []
    for raw in premises:
        if not isinstance(raw, str) or not raw.strip():
            continue
        parsed.append(parser.parse(raw))
    return parsed


def _normalize_formula_string(text: str) -> str:
    """Tokenize and re-emit using canonical ASCII ops, ignoring whitespace.

    Used by literal_in_premise mode to compare two formula strings while
    tolerating LaTeX/Unicode/ASCII variations.
    """
    out: List[str] = []
    for tok in tokenize(text):
        if tok.kind == "EOF":
            break
        if tok.kind in ("LPAREN", "RPAREN", "COMMA"):
            out.append({"LPAREN": "(", "RPAREN": ")", "COMMA": ","}[tok.kind])
            continue
        if tok.kind == "IDENT":
            out.append(tok.value)
            continue
        # Operator token: kind is the canonical name (AND, OR, IMPLIES, ...).
        out.append(tok.kind)
    # Compact: spaces only between IDENT/operator tokens to avoid identifier merge.
    compact = " ".join(out)
    compact = re.sub(r"\s*([(),])\s*", r"\1", compact)
    return compact


def substitute_atoms(formula: str, mapping: Dict[str, str]) -> str:
    """Replace placeholder atom keys with candidate symbols/predicate calls.

    `mapping` maps placeholder key -> candidate-side replacement string. Done as
    whole-word substitution so longer keys are replaced before shorter ones to
    avoid prefix collisions (e.g. `wind` vs `wind2`).
    """
    if not mapping:
        return formula
    result = formula
    for key in sorted(mapping.keys(), key=len, reverse=True):
        replacement = mapping[key]
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])")
        result = pattern.sub(replacement, result)
    return result


def check_target(
    candidate_fl: Dict[str, Any],
    target_formula: str,
    mode: str = "entails",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> CheckResult:
    """Run a Z3 entailment / equivalence / literal check.

    `target_formula` must already be in the candidate's symbol space (see
    `substitute_atoms`).
    """
    mode = (mode or "entails").lower()
    if mode not in ("entails", "equiv", "literal_in_premise"):
        return CheckResult(False, f"unknown mode: {mode}", mode, error="bad_mode")

    candidate_premises = candidate_fl.get("premise") or []
    if not isinstance(candidate_premises, list):
        return CheckResult(False, "candidate FL has no premise list", mode, error="no_premise")

    if mode == "literal_in_premise":
        try:
            target_norm = _normalize_formula_string(target_formula)
        except PARSE_EXCEPTIONS as exc:
            return CheckResult(False, f"target parse error: {exc}", mode, error="parse_error")
        for raw in candidate_premises:
            if not isinstance(raw, str):
                continue
            try:
                if _normalize_formula_string(raw) == target_norm:
                    return CheckResult(True, f"literal match in premise: {raw}", mode)
            except PARSE_EXCEPTIONS:
                continue
        return CheckResult(False, "target string not found in any premise", mode)

    try:
        env = _build_env(candidate_fl)
        parser = LogicParser(env)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(False, f"env build failed: {exc}", mode, error="env_error")

    try:
        target_expr = parser.parse(target_formula)
    except PARSE_EXCEPTIONS as exc:
        return CheckResult(False, f"target parse error: {exc}", mode, error="parse_error")

    try:
        premise_exprs = _parse_premises(parser, candidate_premises)
    except PARSE_EXCEPTIONS as exc:
        return CheckResult(False, f"premise parse error: {exc}", mode, error="parse_error")

    if mode == "entails":
        return _check_entails(premise_exprs, target_expr, timeout_ms)

    # equiv: target ⟺ p for some single premise p
    try:
        for raw, expr in zip(candidate_premises, premise_exprs):
            if _bidir_equiv(expr, target_expr, timeout_ms):
                return CheckResult(True, f"equivalent to premise: {raw}", "equiv")
    except Z3Exception as exc:
        return CheckResult(False, f"z3 equivalence error: {exc}", "equiv", error="z3_error")
    return CheckResult(False, "no premise is logically equivalent to target", "equiv")


def _check_entails(
    premise_exprs: Sequence[Any], target_expr: Any, timeout_ms: int
) -> CheckResult:
    solver = Solver()
    solver.set("timeout", timeout_ms)
    try:
        for premise in premise_exprs:
            solver.add(premise)
        solver.add(Not(target_expr))
        result = solver.check()
    except Z3Exception as exc:
        return CheckResult(False, f"z3 entailment error: {exc}", "entails", error="z3_error")
    if result == unsat:
        return CheckResult(True, "premises entail target (Not(target) is unsat)", "entails")
    if result == sat:
        return CheckResult(False, "premises do not entail target (counter-model exists)", "entails")
    return CheckResult(False, f"z3 returned {result}", "entails", error="timeout_or_unknown")


def _bidir_equiv(a: Any, b: Any, timeout_ms: int) -> bool:
    for lhs, rhs in ((a, b), (b, a)):
        solver = Solver()
        solver.set("timeout", timeout_ms)
        solver.add(lhs)
        solver.add(Not(rhs))
        if solver.check() != unsat:
            return False
    return True


def _entails(premises: Sequence[Any], target: Any, timeout_ms: int) -> str:
    """Tri-state entailment: 'entails' | 'not_entails' | 'unknown'."""
    solver = Solver()
    solver.set("timeout", timeout_ms)
    for p in premises:
        solver.add(p)
    solver.add(Not(target))
    result = solver.check()
    if result == unsat:
        return "entails"
    if result == sat:
        return "not_entails"
    return "unknown"


def _bidir_equiv_tri(a: Any, b: Any, timeout_ms: int) -> str:
    """Tri-state bidirectional equivalence: 'equiv' | 'not_equiv' | 'unknown'."""
    fwd = _entails([a], b, timeout_ms)
    if fwd == "unknown":
        return "unknown"
    if fwd == "not_entails":
        return "not_equiv"
    bwd = _entails([b], a, timeout_ms)
    if bwd == "unknown":
        return "unknown"
    if bwd == "not_entails":
        return "not_equiv"
    return "equiv"


def check_premise_equivalence(
    cand_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Dict[str, Any]:
    """fixedFL premise equivalence check.

    Stage A: bidirectional equivalence of conj(cand_premise) and conj(gold_premise).
    Stage C (only if Stage A finds them not equivalent): for each candidate premise,
    check whether gold entails it (else it is an "extra" constraint); for each gold
    premise, check whether candidate entails it (else it is "missing").

    Returns:
        {
            'equivalent': True | False | None,   # None on z3 unknown / parse error
            'extras':  [(idx, raw_text), ...],   # cand premises gold does not entail
            'missing': [(idx, raw_text), ...],   # gold premises cand does not entail
            'error':   None | str,
        }
    """
    cand_premises_raw = cand_fl.get("premise") or []
    gold_premises_raw = gold_fl.get("premise") or []
    if not isinstance(cand_premises_raw, list) or not isinstance(gold_premises_raw, list):
        return {"equivalent": None, "extras": [], "missing": [], "error": "premise field is not a list"}
    if not cand_premises_raw and not gold_premises_raw:
        return {"equivalent": True, "extras": [], "missing": [], "error": None}

    try:
        env = _build_env(cand_fl)  # fixedFL: cand and gold share params/translation
        parser = LogicParser(env)
    except Exception as exc:
        return {"equivalent": None, "extras": [], "missing": [], "error": f"env build failed: {exc}"}

    def _safe_parse(raw: Any):
        try:
            if not isinstance(raw, str) or not raw.strip():
                return None, None
            return parser.parse(raw), None
        except Exception as exc:
            return None, f"parse error: {exc}"

    cand_pairs = [(i, raw, *_safe_parse(raw)) for i, raw in enumerate(cand_premises_raw)]
    gold_pairs = [(j, raw, *_safe_parse(raw)) for j, raw in enumerate(gold_premises_raw)]

    cand_exprs = [t[2] for t in cand_pairs if t[2] is not None]
    gold_exprs = [t[2] for t in gold_pairs if t[2] is not None]

    cand_parse_errs = [t for t in cand_pairs if t[2] is None and t[3] is not None]
    gold_parse_errs = [t for t in gold_pairs if t[2] is None and t[3] is not None]
    if cand_parse_errs or gold_parse_errs:
        return {
            "equivalent": None,
            "extras": [],
            "missing": [],
            "error": f"parse error (cand={len(cand_parse_errs)}, gold={len(gold_parse_errs)})",
        }

    cand_conj = And(*cand_exprs) if cand_exprs else True
    gold_conj = And(*gold_exprs) if gold_exprs else True

    verdict = _bidir_equiv_tri(cand_conj, gold_conj, timeout_ms)
    if verdict == "equiv":
        return {"equivalent": True, "extras": [], "missing": [], "error": None}
    if verdict == "unknown":
        return {"equivalent": None, "extras": [], "missing": [], "error": "z3 unknown / timeout"}

    # Stage C: locate specific differences
    extras: List[tuple] = []
    for idx, raw, expr, _ in cand_pairs:
        if expr is None:
            continue
        verdict_i = _entails([gold_conj], expr, timeout_ms)
        if verdict_i == "not_entails":
            extras.append((idx, raw))

    missing: List[tuple] = []
    for jdx, raw, expr, _ in gold_pairs:
        if expr is None:
            continue
        verdict_j = _entails([cand_conj], expr, timeout_ms)
        if verdict_j == "not_entails":
            missing.append((jdx, raw))

    return {"equivalent": False, "extras": extras, "missing": missing, "error": None}


def _candidate_query_types(candidate_fl: Dict[str, Any]) -> List[str]:
    """Return the list of query types declared in candidate.question, in order."""
    qs = candidate_fl.get("question") or []
    if not isinstance(qs, list):
        return []
    types: List[str] = []
    for q in qs:
        if not isinstance(q, str):
            continue
        try:
            qt, _, _ = parse_query(q)
            types.append(qt)
        except Exception:
            continue
    return types


def _candidate_enum_target(candidate_fl: Dict[str, Any]) -> Optional[List[str]]:
    """Return the variable list of the candidate's first enumerate/count query, if any."""
    qs = candidate_fl.get("question") or []
    if not isinstance(qs, list):
        return None
    for q in qs:
        if not isinstance(q, str):
            continue
        try:
            qt, _, args = parse_query(q)
        except Exception:
            continue
        if qt in ("enumerate_models", "count_models"):
            return [a.strip() for a in args]
    return None


def _candidate_simple_target(candidate_fl: Dict[str, Any], qtype: str) -> Optional[str]:
    """Return the target formula string of the candidate's first matching possible/necessary query."""
    qs = candidate_fl.get("question") or []
    if not isinstance(qs, list):
        return None
    for q in qs:
        if not isinstance(q, str):
            continue
        try:
            qt, target, _ = parse_query(q)
        except Exception:
            continue
        if qt == qtype:
            return target
    return None


def check_query_equiv(
    candidate_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    query_type: str,
    target: Sequence[str],
    gold_target: Optional[Sequence[str]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> CheckResult:
    """Check that candidate's query is answer-equivalent to gold's query.

    `query_type` and `target` describe the gold-side query, with `target` already
    translated into candidate's symbol space via `substitute_atoms`. `gold_target`
    is the same target list in gold's original symbol space, used to enumerate
    gold's expected answer set. If `gold_target` is omitted, it defaults to
    `target` (works when gold and candidate happen to share symbol names).

    Steps:
      1. candidate.question must contain a query of `query_type` (else fail).
      2. For enumerate/count: candidate's enumerated variables must form a
         SUPERSET of `target` (else fail; subset = wrong).
      3. Compute gold's expected answer using `gold_fl` + `gold_target`.
      4. Compute candidate's projected answer on `target`.
      5. Compare; pass iff equal.
    """
    mode = "query_equiv"
    if query_type not in SUPPORTED_QUERY_TYPES:
        return CheckResult(False, f"unsupported query_type: {query_type}", mode, error="bad_query_type")

    cand_types = _candidate_query_types(candidate_fl)
    if query_type not in cand_types:
        return CheckResult(
            False,
            f"candidate question does not contain a {query_type} query (found: {cand_types})",
            mode,
            error="query_type_mismatch",
        )

    if gold_target is None:
        gold_target = list(target)

    try:
        env_c = _build_env(candidate_fl)
        parser_c = LogicParser(env_c)
        premises_c = _parse_premises(parser_c, candidate_fl.get("premise") or [])
    except PARSE_EXCEPTIONS as exc:
        return CheckResult(False, f"candidate parse error: {exc}", mode, error="parse_error")
    except Exception as exc:  # pragma: no cover
        return CheckResult(False, f"candidate env build failed: {exc}", mode, error="env_error")

    try:
        env_g = _build_env(gold_fl)
        parser_g = LogicParser(env_g)
        premises_g = _parse_premises(parser_g, gold_fl.get("premise") or [])
    except PARSE_EXCEPTIONS as exc:
        return CheckResult(False, f"gold parse error: {exc}", mode, error="parse_error")
    except Exception as exc:  # pragma: no cover
        return CheckResult(False, f"gold env build failed: {exc}", mode, error="env_error")

    if query_type in ("possible", "necessary"):
        # `target` is a single formula string in candidate symbol space.
        if len(target) != 1:
            return CheckResult(False, f"{query_type} expects 1 target formula, got {len(target)}", mode, error="bad_target")
        target_formula_c = target[0]
        target_formula_g = gold_target[0]
        try:
            target_expr_c = parser_c.parse(target_formula_c)
            target_expr_g = parser_g.parse(target_formula_g)
        except PARSE_EXCEPTIONS as exc:
            return CheckResult(False, f"target parse error: {exc}", mode, error="parse_error")

        try:
            if query_type == "possible":
                _, label_c = check_possible(premises_c, target_expr_c, timeout_ms)
                _, label_g = check_possible(premises_g, target_expr_g, timeout_ms)
            else:
                _, label_c = check_necessary(premises_c, target_expr_c, timeout_ms)
                _, label_g = check_necessary(premises_g, target_expr_g, timeout_ms)
        except Z3Exception as exc:
            return CheckResult(False, f"z3 query error: {exc}", mode, error="z3_error")

        if label_c == label_g:
            return CheckResult(True, f"{query_type} verdicts match: {label_c}", mode)
        return CheckResult(
            False,
            f"{query_type} verdict mismatch: candidate={label_c} gold={label_g}",
            mode,
            error="verdict_mismatch",
        )

    # enumerate_models / count_models
    cand_enum_target = _candidate_enum_target(candidate_fl)
    if cand_enum_target is None:
        return CheckResult(False, "candidate has no enumerate/count query", mode, error="query_type_mismatch")

    target_set = [t.strip() for t in target]
    if not set(target_set).issubset(set(cand_enum_target)):
        missing = set(target_set) - set(cand_enum_target)
        return CheckResult(
            False,
            f"candidate target missing gold variables: {sorted(missing)}",
            mode,
            error="target_subset",
        )

    # Verify each gold target name actually exists as a Bool symbol in candidate env.
    for name in target_set:
        if name not in (candidate_fl.get("parameters") or {}):
            return CheckResult(False, f"target {name!r} not declared in candidate parameters", mode, error="undeclared_symbol")

    try:
        status_c, models_c = enumerate_bool_models(env_c, premises_c, target_set, timeout_ms)
    except Exception as exc:
        return CheckResult(False, f"candidate enumerate error: {exc}", mode, error="enumerate_error")
    try:
        status_g, models_g = enumerate_bool_models(env_g, premises_g, list(gold_target), timeout_ms)
    except Exception as exc:
        return CheckResult(False, f"gold enumerate error: {exc}", mode, error="enumerate_error")

    if status_c != "success" or status_g != "success":
        return CheckResult(
            False,
            f"enumerate status candidate={status_c} gold={status_g}",
            mode,
            error="timeout_or_unknown",
        )

    if query_type == "count_models":
        if len(models_c) == len(models_g):
            return CheckResult(True, f"count match: {len(models_c)}", mode)
        return CheckResult(
            False,
            f"count mismatch: candidate={len(models_c)} gold={len(models_g)}",
            mode,
            error="count_mismatch",
        )

    # enumerate_models: compare projected sets.
    # Each `models_*` entry is a list of variable names that were True in the model.
    # Map gold names → candidate names by position so set comparison is symbol-agnostic.
    gold_to_cand = {g: c for g, c in zip(gold_target, target_set)}
    set_c = {frozenset(m) for m in models_c}
    set_g = {frozenset(gold_to_cand[v] for v in m) for m in models_g}
    if set_c == set_g:
        return CheckResult(True, f"enumerate sets match ({len(set_c)} models)", mode)
    only_c = set_c - set_g
    only_g = set_g - set_c
    return CheckResult(
        False,
        f"enumerate mismatch: only_in_candidate={[sorted(s) for s in only_c]} only_in_gold={[sorted(s) for s in only_g]}",
        mode,
        error="enumerate_mismatch",
    )


def candidate_answers_match_gold(
    cand_fl: Dict[str, Any],
    gold_fl: Dict[str, Any],
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> bool:
    """fixedFL gate: do candidate's z3-solved answers equal gold's, for every gold query?

    Iterates over `gold_fl.question`; for each, parses the gold-side query and
    delegates to `check_query_equiv` (which solves both sides under their
    respective premises and compares projected answer sets / labels). In
    fixedFL the candidate and gold share atom symbols, so we pass the same
    target on both sides.

    Returns True iff every gold query passes (i.e. candidate's answer matches
    gold's). Returns False on any single mismatch, parse error, or query-type
    incompatibility — these all imply we should NOT skip PE failure injection.
    """
    gold_questions = gold_fl.get("question") or []
    if not isinstance(gold_questions, list) or not gold_questions:
        return False
    for q in gold_questions:
        if not isinstance(q, str):
            return False
        try:
            qtype, _raw_target, args = parse_query(q)
        except Exception:
            return False
        if qtype in ("possible", "necessary"):
            target_list: List[str] = [_raw_target]
        else:
            target_list = [a.strip() for a in args]
        result = check_query_equiv(
            cand_fl, gold_fl, qtype, target_list, gold_target=target_list, timeout_ms=timeout_ms,
        )
        if not result.passed:
            return False
    return True

