"""LogicBench's canonical Z3 engine.

Lifted verbatim from the previous code/generate/solve_from_formal.py
(lines 32-495 in the pre-refactor file). This is the only LaTeX/Unicode →
Z3 expression parser in the repo. Older parsers under code/verify/ and
code/logic_matcher/ have been retired in favor of this module.

Public API (everything imported by formalize/solve/lib.fl_schema):
- Token, FormulaParseError, QueryResult
- normalize_operator, tokenize, split_top_level, clean_symbol_name
- function_arity, is_bool_kind, infer_translation_symbols
- LogicEnvironment, LogicParser
- parse_query, solver_with_premises
- check_possible, check_necessary, enumerate_bool_models, enumerate_fol_objects
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from z3 import (
    And,
    Bool,
    BoolRef,
    BoolSort,
    Const,
    DeclareSort,
    Exists,
    ExprRef,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    is_bool,
    sat,
    unsat,
)


QUERY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*\((.*)\)\s*$", re.DOTALL)
FUNCTION_TYPE_PATTERN = re.compile(r"Function\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(
    r"\s+|"
    r"(\\[A-Za-z]+)|"
    r"(<->|->|==|!=)|"
    r"([()\[\],=])|"
    r"([A-Za-z_][A-Za-z_0-9]*)|"
    r"(¬|∧|∨|→|↔|∀|∃|≠|[&|~])"
)


class FormulaParseError(ValueError):
    pass


@dataclass
class Token:
    kind: str
    value: str


@dataclass
class QueryResult:
    query: str
    query_type: str
    target: str
    status: str
    token: str
    targets: Optional[List[str]] = None
    models: Optional[List[List[str]]] = None
    count: Optional[Any] = None


_OPERATOR_MAP: Dict[str, str] = {
    r"\neg": "NOT", r"\lnot": "NOT", "¬": "NOT", "~": "NOT",
    r"\wedge": "AND", r"\land": "AND", "∧": "AND", "&": "AND",
    r"\vee": "OR", r"\lor": "OR", "∨": "OR", "|": "OR",
    r"\to": "IMPLIES", r"\rightarrow": "IMPLIES", r"\Rightarrow": "IMPLIES",
    "->": "IMPLIES", "→": "IMPLIES",
    r"\leftrightarrow": "IFF", r"\Leftrightarrow": "IFF", r"\iff": "IFF",
    "<->": "IFF", "↔": "IFF",
    r"\forall": "FORALL", "∀": "FORALL",
    r"\exists": "EXISTS", "∃": "EXISTS",
    r"\neq": "NEQ", "!=": "NEQ", "≠": "NEQ",
    "==": "EQ", "=": "EQ",
}


def normalize_operator(raw: str) -> str:
    return _OPERATOR_MAP.get(raw, raw)


def tokenize(formula: str) -> List[Token]:
    text = formula.replace("$", "").strip()
    text = re.sub(r"\\(?:left|right|big|Big)(?=[()\[\]{}])", "", text)
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        match = TOKEN_PATTERN.match(text, pos)
        if not match:
            raise FormulaParseError(f"unexpected token near: {text[pos:pos + 20]!r}")
        pos = match.end()
        raw = match.group(0)
        if raw.isspace():
            continue
        if match.group(1) or match.group(2) or match.group(5):
            op = normalize_operator(raw)
            if op != raw:
                tokens.append(Token(op, raw))
            else:
                tokens.append(Token("IDENT", raw))
        elif match.group(3):
            punct = raw
            if punct == "[":
                punct = "("
            elif punct == "]":
                punct = ")"
            kind = {"(": "LPAREN", ")": "RPAREN", ",": "COMMA", "=": "EQ"}[punct]
            tokens.append(Token(kind, raw))
        elif match.group(4):
            tokens.append(Token("IDENT", raw))
    tokens.append(Token("EOF", ""))
    return tokens


def split_top_level(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def clean_symbol_name(name: str) -> str:
    return name.strip().strip("$")


def function_arity(type_text: Any) -> Optional[int]:
    if not isinstance(type_text, str):
        return None
    match = FUNCTION_TYPE_PATTERN.search(type_text)
    if not match:
        return None
    return int(match.group(1))


def is_bool_kind(type_text: Any) -> bool:
    return isinstance(type_text, str) and type_text.strip().lower() == "bool"


def infer_translation_symbols(translation: Dict[str, Any]) -> Dict[str, str]:
    inferred: Dict[str, str] = {}
    for raw_key in translation.keys():
        key = clean_symbol_name(str(raw_key))
        fn_match = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)\s*\((.*)\)$", key)
        if fn_match:
            name = fn_match.group(1)
            args = split_top_level(fn_match.group(2))
            inferred.setdefault(name, f"Function({len(args)})")
        elif key:
            if key[:1].islower():
                inferred.setdefault(key, "Domain")
            else:
                inferred.setdefault(key, "Bool")
    return inferred


class LogicEnvironment:
    def __init__(self, parameters: Dict[str, str], translation: Dict[str, Any]):
        self.parameters = parameters
        self.translation = translation
        self.domain = DeclareSort("Domain")
        self.bools: Dict[str, BoolRef] = {}
        self.constants: Dict[str, ExprRef] = {}
        self.predicates: Dict[str, Any] = {}
        self.bound_vars: Dict[str, ExprRef] = {}
        self._build()

    def _build(self) -> None:
        for name, kind in self.parameters.items():
            arity = function_arity(kind)
            if arity is not None:
                self.predicates[name] = Function(name, *([self.domain] * arity + [BoolSort()]))
            elif is_bool_kind(kind):
                self.bools[name] = Bool(name)
            else:
                self.constants[name] = Const(name, self.domain)

    def bool_symbol(self, name: str) -> BoolRef:
        if name in self.bools:
            return self.bools[name]
        if name in self.predicates:
            self.bools[name] = Bool(name)
            return self.bools[name]
        if name in self.constants:
            raise FormulaParseError(f"{name} is not a Bool symbol")
        self.bools[name] = Bool(name)
        return self.bools[name]

    def term_symbol(self, name: str) -> ExprRef:
        if name in self.bound_vars:
            return self.bound_vars[name]
        if name in self.constants:
            return self.constants[name]
        if name in self.bools:
            raise FormulaParseError(f"{name} is a Bool symbol, not a domain term")
        self.constants[name] = Const(name, self.domain)
        return self.constants[name]

    def predicate(self, name: str) -> Any:
        if name in self.predicates:
            return self.predicates[name]
        raise FormulaParseError(f"{name} is not declared as a predicate/function")

    def named_constants(self) -> List[str]:
        def sort_key(name: str) -> Tuple[int, str]:
            return (0 if len(name) == 1 else 1, name)

        return sorted(self.constants.keys(), key=sort_key)


class LogicParser:
    def __init__(self, env: LogicEnvironment):
        self.env = env
        self.tokens: List[Token] = []
        self.pos = 0

    def parse(self, formula: str) -> ExprRef:
        self.tokens = tokenize(formula)
        self.pos = 0
        expr = self.parse_iff()
        if self.current().kind != "EOF":
            raise FormulaParseError(f"unexpected trailing token: {self.current().value}")
        if not is_bool(expr):
            raise FormulaParseError(f"formula did not parse to a Bool expression: {formula}")
        return expr

    def parse_with_binding(self, formula: str, name: str, value: ExprRef) -> ExprRef:
        old = self.env.bound_vars.get(name)
        self.env.bound_vars[name] = value
        try:
            return self.parse(formula)
        finally:
            if old is None:
                self.env.bound_vars.pop(name, None)
            else:
                self.env.bound_vars[name] = old

    def current(self) -> Token:
        return self.tokens[self.pos]

    def accept(self, *kinds: str) -> Optional[Token]:
        if self.current().kind in kinds:
            token = self.current()
            self.pos += 1
            return token
        return None

    def expect(self, kind: str) -> Token:
        token = self.accept(kind)
        if token is None:
            raise FormulaParseError(
                f"expected {kind}, got {self.current().value or self.current().kind}"
            )
        return token

    def parse_iff(self) -> ExprRef:
        left = self.parse_implies()
        while self.accept("IFF"):
            right = self.parse_implies()
            left = left == right
        return left

    def parse_implies(self) -> ExprRef:
        left = self.parse_or()
        if self.accept("IMPLIES"):
            right = self.parse_implies()
            return Implies(left, right)
        return left

    def parse_or(self) -> ExprRef:
        terms = [self.parse_and()]
        while self.accept("OR"):
            terms.append(self.parse_and())
        if len(terms) == 1:
            return terms[0]
        return Or(*terms)

    def parse_and(self) -> ExprRef:
        terms = [self.parse_unary()]
        while self.accept("AND"):
            terms.append(self.parse_unary())
        if len(terms) == 1:
            return terms[0]
        return And(*terms)

    def parse_unary(self) -> ExprRef:
        if self.accept("NOT"):
            return Not(self.parse_unary())
        if self.current().kind in ("FORALL", "EXISTS"):
            return self.parse_quantifier()
        return self.parse_atom()

    def parse_quantifier(self) -> ExprRef:
        quantifier = self.current().kind
        self.pos += 1
        variables: List[Tuple[str, ExprRef, Optional[ExprRef]]] = []
        first = self.expect("IDENT").value
        variables.append((first, Const(first, self.env.domain), self.env.bound_vars.get(first)))
        while self.accept("COMMA"):
            name = self.expect("IDENT").value
            variables.append((name, Const(name, self.env.domain), self.env.bound_vars.get(name)))

        for name, expr, _old in variables:
            self.env.bound_vars[name] = expr
        try:
            body = self.parse_unary() if self.current().kind == "LPAREN" else self.parse_implies()
        finally:
            for name, _expr, old in reversed(variables):
                if old is None:
                    self.env.bound_vars.pop(name, None)
                else:
                    self.env.bound_vars[name] = old

        bound_exprs = [expr for _name, expr, _old in variables]
        return ForAll(bound_exprs, body) if quantifier == "FORALL" else Exists(bound_exprs, body)

    def parse_atom(self) -> ExprRef:
        if self.accept("LPAREN"):
            expr = self.parse_iff()
            self.expect("RPAREN")
            return expr

        if self.current().kind != "IDENT":
            raise FormulaParseError(
                f"expected atom, got {self.current().value or self.current().kind}"
            )

        if self.lookahead_kind() == "LPAREN":
            return self.parse_predicate_application()

        name = self.expect("IDENT").value
        if self.current().kind in ("EQ", "NEQ"):
            left = self.env.term_symbol(name)
            op = self.current().kind
            self.pos += 1
            right = self.parse_term()
            return left != right if op == "NEQ" else left == right
        return self.env.bool_symbol(name)

    def parse_predicate_application(self) -> ExprRef:
        name = self.expect("IDENT").value
        self.expect("LPAREN")
        args: List[ExprRef] = []
        if self.current().kind != "RPAREN":
            args.append(self.parse_term())
            while self.accept("COMMA"):
                args.append(self.parse_term())
        self.expect("RPAREN")
        return self.env.predicate(name)(*args)

    def parse_term(self) -> ExprRef:
        if self.current().kind != "IDENT":
            raise FormulaParseError(
                f"expected term, got {self.current().value or self.current().kind}"
            )
        if self.lookahead_kind() == "LPAREN":
            return self.parse_predicate_application()
        return self.env.term_symbol(self.expect("IDENT").value)

    def lookahead_kind(self) -> str:
        if self.pos + 1 >= len(self.tokens):
            return "EOF"
        return self.tokens[self.pos + 1].kind


def parse_query(query: str) -> Tuple[str, str, List[str]]:
    match = QUERY_PATTERN.match(query)
    if not match:
        raise FormulaParseError(f"invalid query syntax: {query}")
    query_type = match.group(1).strip().lower()
    args = split_top_level(match.group(2))
    target = match.group(2).strip()
    if not args:
        raise FormulaParseError(f"query has no arguments: {query}")
    return query_type, target, args


def solver_with_premises(premises: Sequence[ExprRef], timeout_ms: int) -> Solver:
    solver = Solver()
    solver.set("timeout", timeout_ms)
    for premise in premises:
        solver.add(premise)
    return solver


def check_possible(premises: Sequence[ExprRef], expr: ExprRef, timeout_ms: int) -> Tuple[str, str]:
    solver = solver_with_premises(premises, timeout_ms)
    solver.add(expr)
    result = solver.check()
    if result == sat:
        return "sat", "possible"
    if result == unsat:
        return "unsat", "impossible"
    return str(result), "unknown"


def check_necessary(premises: Sequence[ExprRef], expr: ExprRef, timeout_ms: int) -> Tuple[str, str]:
    solver = solver_with_premises(premises, timeout_ms)
    solver.add(Not(expr))
    result = solver.check()
    if result == unsat:
        return "unsat", "necessary"
    if result == sat:
        return "sat", "unnecessary"
    return str(result), "unknown"


def enumerate_bool_models(
    env: LogicEnvironment,
    premises: Sequence[ExprRef],
    target_names: Sequence[str],
    timeout_ms: int,
    *,
    max_assignments: Optional[int] = None,
    deadline: Optional[float] = None,
) -> Tuple[str, List[List[str]]]:
    bools = [env.bool_symbol(name) for name in target_names]
    assignment_count = 1 << len(bools)
    if max_assignments is not None and assignment_count > max_assignments:
        return f"unknown:enumeration_limit:{assignment_count}>{max_assignments}", []

    models: List[List[str]] = []
    saw_unknown = False
    for bits in itertools.product([False, True], repeat=len(bools)):
        if deadline is not None and time.monotonic() >= deadline:
            return "unknown:wall_timeout", models
        solver = solver_with_premises(premises, timeout_ms)
        for symbol, bit in zip(bools, bits):
            solver.add(symbol if bit else Not(symbol))
        result = solver.check()
        if deadline is not None and time.monotonic() >= deadline:
            return "unknown:wall_timeout", models
        if result == sat:
            models.append([name for name, bit in zip(target_names, bits) if bit])
        elif result != unsat:
            saw_unknown = True
    order = {name: idx for idx, name in enumerate(target_names)}
    models.sort(key=lambda model: [order[name] for name in model])
    return ("unknown" if saw_unknown else "success"), models


def enumerate_fol_objects(
    env: LogicEnvironment,
    parser: LogicParser,
    premises: Sequence[ExprRef],
    formula: str,
    variable_name: str,
    timeout_ms: int,
    *,
    max_objects: Optional[int] = None,
    deadline: Optional[float] = None,
) -> Tuple[str, List[List[str]]]:
    models: List[List[str]] = []
    saw_unknown = False
    const_names = env.named_constants()
    if max_objects is not None and len(const_names) > max_objects:
        return f"unknown:enumeration_limit:{len(const_names)}>{max_objects}", []

    for const_name in const_names:
        if deadline is not None and time.monotonic() >= deadline:
            return "unknown:wall_timeout", models
        expr = parser.parse_with_binding(formula, variable_name, env.term_symbol(const_name))
        solver = solver_with_premises(premises, timeout_ms)
        solver.add(expr)
        result = solver.check()
        if deadline is not None and time.monotonic() >= deadline:
            return "unknown:wall_timeout", models
        if result == sat:
            models.append([const_name])
        elif result != unsat:
            saw_unknown = True
    return ("unknown" if saw_unknown else "success"), models
