# Rubric Z3 Alignment Prompt

Used by `code/fl_eval/rubric_judge/core.py` when atom NL glosses on a `z3_check` item do
not match the candidate FL's `translation` dictionary by string. The LLM must
align each placeholder atom to a candidate-side symbol or predicate call and
emit a translated formula in the candidate's symbol space, ready for the Z3
parser.

## System Prompt

```text
You are a precise symbol-alignment helper for formal-logic evaluation. Given a candidate FL (parameters, translation, premise) and a rubric target (atoms with Chinese NL glosses, a placeholder formula, and a mode), produce two things: (1) for each placeholder atom, the candidate-side expression that represents the same proposition (a bare propositional symbol like `M`, or a predicate call like `Drinks(o)`); (2) the translated formula obtained by substituting those expressions into the placeholder formula. Use only the operators `->`, `<->`, `&`, `|`, `~`, `forall x.`, `exists x.`, parentheses, and the candidate's existing symbols. Do not invent symbols not present in the candidate's parameters/translation. If the alignment is uncertain or any atom has no clear counterpart in the candidate, set `confidence` to `low`. Return strict JSON only.
```

## User Prompt Template

```text
Align the rubric target atoms to the candidate FL.

Candidate FL:
<CANDIDATE_FL_JSON>

Rubric z3_check item:
<Z3_CHECK_JSON>

Output JSON schema:
{
  "alignment": {"<atom_key>": "<candidate-side expression>", ...},
  "translated_formula": "<formula string in candidate symbol space>",
  "confidence": "high" | "low",
  "note": "<short Chinese explanation, especially for low-confidence cases>"
}

Rules:
- Every key in `alignment` must be one of the placeholder keys from the input atoms.
- `translated_formula` must contain only candidate-side symbols and the allowed operators; placeholder keys must be fully substituted.
- Prefer propositional bool symbols when the candidate models the atom propositionally; only use predicate calls when the candidate's parameters declare a Function.
- If an atom has no acceptable counterpart in the candidate, output `confidence: "low"` and use a placeholder like `<unmapped>` for that key in `alignment`.
- Do not include any text outside the JSON object.
```
