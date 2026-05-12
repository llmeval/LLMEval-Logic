# Rubric PE Soft-Review Prompt

Used by `code/fl_eval/rubric_judge/core.py` when the global premise-equivalence (PE) z3
check reports `not_equiv` for a fixedFL candidate, AND the candidate's z3-solved
answer matches gold's reference answer. In that gate, each individual difference
(an extra candidate premise not entailed by gold, or a missing gold premise not
entailed by candidate) is sent to the LLM here for a soft-pass / fail verdict.

The LLM must judge the *defensibility* of the difference relative to the source
NL problem and gold reference — not just literal equivalence. A difference
softpasses if it is a reasonable reading of the NL source (extras case) or an
acceptable alternative form of the same content (missing case).

## System Prompt

```text
You are a strict logician reviewing one premise-set difference between a candidate formalization and a gold reference formalization of a Chinese natural-language logic problem. Both formalizations operate over the same atom symbols and translations (fixedFL). Your job is to decide whether the single specific difference flagged by Z3 is a defensible formalization choice rather than an error. Treat the original NL problem as authoritative; use the rubric desc/reason context only as secondary guidance for intended interpretation and acceptable variants. Return strict JSON only: `{"softpass": true|false, "reason": "<Chinese ≤30 chars>"}`. Do NOT analyze multiple premises at once. Do NOT include any text outside the JSON object.
```

## User Prompt Extras

```text
The candidate added one premise that is NOT entailed by the conjunction of the gold premises. Decide if this extra premise is a defensible reading of the NL source.

Source NL problem (background):
<ORIGINAL_BACKGROUND>

Question:
<ORIGINAL_QUESTION>

Rubric desc/reason context (secondary guidance; original NL is authoritative):
<RUBRIC_CONTEXT_JSON>

Atom translations (shared by gold and candidate):
<TRANSLATION_DICT>

Gold formalization premises:
<GOLD_PREMISES_LIST>

Candidate added an extra premise NOT entailed by gold:
<EXTRA_PREMISE_RAW>

Decide softpass iff one of the following is clearly true:
- (a) The extra premise restates a fact directly given in the NL background.
- (b) The extra premise is a natural-language pragmatic implicature obviously intended in this context (e.g. "either A or B" in an exclusive context implies mutual exclusion).
- (c) The extra premise is a fact that gold simply omitted from formalization but is unambiguous from the source.
- (d) The rubric desc/reason explicitly allows this stronger or alternative reading, and that allowance does not contradict the NL source.

IMPORTANT — adding any constraint not entailed by gold is by default an unsupported assumption. Softpass requires a clear NL-source or rubric-desc justification (a, b, c, or d). "It does not affect the question's answer" is NOT sufficient on its own; an unjustified extra premise still misrepresents the source even if the answer happens to coincide. If the rubric context conflicts with the NL source, prefer the NL source. In particular:
- If the stronger form changes the satisfiability of the premise set (e.g. the gold premises are satisfiable but adding the extra premise makes them unsatisfiable, or vice versa), softpass=false.
- If the stronger form adds new consequences that affect the question's answer, softpass=false.
When in doubt, softpass=false.

Otherwise (the extra premise introduces an unstated assumption, narrows the model space without source justification, or is the candidate's own invention) → softpass=false.

Output strict JSON: {"softpass": true|false, "reason": "<Chinese, ≤30 chars, MUST be non-empty>"}
```

## User Prompt Missing

```text
The candidate's premise set does NOT entail one specific gold premise. Decide if the candidate expresses the same logical content via an equivalent or acceptable alternative form.

Source NL problem (background):
<ORIGINAL_BACKGROUND>

Question:
<ORIGINAL_QUESTION>

Rubric desc/reason context (secondary guidance; original NL is authoritative):
<RUBRIC_CONTEXT_JSON>

Atom translations (shared by gold and candidate):
<TRANSLATION_DICT>

Gold premise that candidate does NOT entail:
<MISSING_PREMISE_RAW>

Candidate's full premise set:
<CAND_PREMISES_LIST>

Decide softpass iff one of the following is clearly true:
- (a) The candidate captures the same content via an equivalent rewriting that Z3 missed (e.g. a different quantifier formulation, a contrapositive, distributing implication over disjunction).
- (b) The gold premise is split across multiple candidate premises that together imply it (only count this when the candidate clearly intends the same content).
- (c) The gold premise is logically redundant for the question being formalized — its role is fully covered by the rest of the candidate set.
- (d) The gold premise expresses the same content as a candidate premise written in a different abstraction layer (e.g. a rule expressed as ground facts that cover all relevant instances in this problem).
- (e) The rubric desc/reason explicitly allows this weaker, alternative, or differently-scoped modeling choice, and that allowance does not contradict the NL source.

IMPORTANT — omitting a gold premise is by default a formalization gap. Softpass requires that the candidate still captures the same logical content through one of (a)–(e). Logical weakening is NOT automatically softpass; if the rubric context conflicts with the NL source, prefer the NL source. In particular:
- If the weaker form changes the satisfiability of the premise set (e.g. gold premises are unsatisfiable/contradictory but the candidate's weaker premises become satisfiable, or vice versa), softpass=false.
- If the weaker form loses consequences that affect the question's answer, softpass=false.
When in doubt, softpass=false.

Otherwise (the candidate genuinely lacks the content of the missing gold premise, or only partially captures it, or expresses it incorrectly) → softpass=false.

Output strict JSON: {"softpass": true|false, "reason": "<Chinese ≤30 chars, MUST be non-empty>"}
```

## Settled Decisions

- `softpass=true` semantically equals "the candidate's formalization is defensible despite z3 reporting non-equivalence". The score record will mark such items with `source="z3:premise_equiv:soft"` and `score=1`.
- `reason` is in Chinese to match the rest of the rubric output.
- Rubric desc/reason context is secondary guidance. It can justify acceptable variants when it clarifies the intended scope, but it cannot override the source NL problem.
- Each LLM call reviews exactly one difference (one extras OR one missing). Do not batch.
- The LLM does NOT see the candidate's z3-solved answer — that gate is enforced upstream by `candidate_answers_match_gold`. The LLM judges only the formalization defensibility, not answer correctness.
