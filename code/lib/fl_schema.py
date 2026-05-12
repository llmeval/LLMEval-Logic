"""Formal Language schema validation, normalization, candidate extraction.

Consolidates duplicated logic that previously lived in:
- code/generate/formalize.py (normalize_fl, extract_base_fl, validate_fl,
  validate_fl_for_solver, normalize_premise_question, normalize_string_list,
  normalize_parameters via re-import)
- code/generate/solve_from_formal.py (normalize_parameters, normalize_premise,
  normalize_question, extract_formal_candidate)

The single source of truth for parser/env construction lives in
`code.lib.z3_engine`; this module only handles JSON shape concerns.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .z3_engine import (
    FormulaParseError,
    LogicEnvironment,
    LogicParser,
    clean_symbol_name,
    infer_translation_symbols,
    parse_query,
)


SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
SUPPORTED_QUERY_TYPES = {"possible", "necessary", "enumerate_models", "count_models"}


def normalize_parameters(parameters: Any, translation: Dict[str, Any]) -> Dict[str, str]:
    inferred = infer_translation_symbols(translation)
    normalized: Dict[str, str] = {}
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            normalized[clean_symbol_name(str(key))] = str(value)
    elif isinstance(parameters, list):
        for item in parameters:
            key = clean_symbol_name(str(item))
            normalized[key] = inferred.get(key, "Domain" if key[:1].islower() else "Bool")
    for key, value in inferred.items():
        normalized.setdefault(key, value)
    return normalized


def normalize_string_list(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    return None


def normalize_premise(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def normalize_question(value: Any) -> Optional[List[str]]:
    return normalize_string_list(value)


def normalize_premise_question(data: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    if not isinstance(data, dict):
        return None
    premise = data.get("premise")
    question = data.get("question")
    if premise is None or question is None:
        reasoning = data.get("reasoning") or {}
        if isinstance(reasoning, dict):
            premise = premise if premise is not None else reasoning.get("premise")
            question = question if question is not None else reasoning.get("question")
    premise_values = normalize_string_list(premise)
    question_values = normalize_string_list(question)
    if premise_values is None or question_values is None:
        return None
    return {"premise": premise_values, "question": question_values}


def normalize_fl(data: Dict[str, Any], allow_partial: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    pc = normalize_premise_question(data)
    if pc is None:
        return None
    if allow_partial:
        return pc

    translation = data.get("translation")
    if not isinstance(translation, dict):
        return None
    translation_out = {str(k): str(v) for k, v in translation.items()}
    parameters = data.get("parameters")
    if parameters is None:
        language = data.get("language")
        if isinstance(language, dict):
            parameters = language.get("parameters")
    return {
        "parameters": normalize_parameters(parameters, translation_out),
        "translation": translation_out,
        "premise": pc["premise"],
        "question": pc["question"],
    }


def extract_base_fl(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull pre-existing parameters/translation from a bench sample (no premise/question)."""
    formalization = sample.get("formalization")
    if not isinstance(formalization, dict):
        return None
    translation = formalization.get("translation")
    if not isinstance(translation, dict):
        return None
    parameters = formalization.get("parameters")
    if parameters is None:
        language = formalization.get("language")
        if isinstance(language, dict):
            parameters = language.get("parameters")
    translation_out = {str(k): str(v) for k, v in translation.items()}
    return {
        "parameters": normalize_parameters(parameters, translation_out),
        "translation": translation_out,
    }


def extract_formal_candidate(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull a solver-ready FL candidate out of a formalize.jsonl record."""
    data = record.get("formal_language")
    if not isinstance(data, dict):
        data = record.get("FL")
    if not isinstance(data, dict):
        data = record.get("formalization")
    if not isinstance(data, dict):
        return None

    translation = data.get("translation")
    if not isinstance(translation, dict):
        return None

    premise = normalize_premise(data.get("premise"))
    question = normalize_question(data.get("question"))
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        premise = premise or normalize_premise(reasoning.get("premise"))
        question = question or normalize_question(reasoning.get("question"))

    if premise is None or question is None:
        return None

    parameters = data.get("parameters")
    if parameters is None:
        language = data.get("language")
        if isinstance(language, dict):
            parameters = language.get("parameters")

    return {
        "parameters": normalize_parameters(parameters, translation),
        "translation": {str(k): str(v) for k, v in translation.items()},
        "premise": premise,
        "question": question,
    }


def validate_fl(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    return all(key in data for key in ("parameters", "translation", "premise", "question"))


def validate_solver_query(env: LogicEnvironment, parser: LogicParser, query: str) -> None:
    query_type, _target, args = parse_query(query)
    if query_type not in SUPPORTED_QUERY_TYPES:
        raise FormulaParseError(f"unsupported query type: {query_type}")
    if query_type in ("possible", "necessary"):
        if len(args) != 1:
            raise FormulaParseError(f"{query_type} expects one formula: {query}")
        parser.parse(args[0])
        return

    if len(args) == 2 and re.search(r"\b" + re.escape(args[1]) + r"\b", args[0]):
        variable_name = clean_symbol_name(args[1])
        if not SYMBOL_PATTERN.fullmatch(variable_name):
            raise FormulaParseError(f"invalid variable name in query: {args[1]}")
        parser.parse_with_binding(args[0], variable_name, env.term_symbol(variable_name))
        return

    for raw_name in args:
        name = clean_symbol_name(raw_name)
        if not SYMBOL_PATTERN.fullmatch(name):
            raise FormulaParseError(f"invalid enumerate/count target: {raw_name}")
        env.bool_symbol(name)


def validate_fl_for_solver(data: Optional[Dict[str, Any]]) -> Optional[str]:
    if data is None:
        return "missing FL"
    if not validate_fl(data):
        return "FL must contain parameters, translation, premise, and question"
    try:
        parameters = normalize_parameters(data.get("parameters"), data["translation"])
        env = LogicEnvironment(parameters, data["translation"])
        parser = LogicParser(env)
        for premise in data["premise"]:
            parser.parse(premise)
        for query in data["question"]:
            validate_solver_query(env, parser, query)
    except Exception as exc:
        return str(exc)
    return None
