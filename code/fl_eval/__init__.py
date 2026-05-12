"""Formalization evaluation track (Formalization Accuracy).

Pipeline (Base items only — Hard items have no gold FL or rubrics):

    bench/base/llmeval_logic_base.json
              │
              ▼
    1. formalize/        NL  →  candidate FL JSON
              │
              ▼
    2. z3_judge/         Z3 execution against the reference  →  "Z3" column
              │
              ▼
    3. rubric_judge/  per-problem rubric atoms from
                         bench/base/rubrics/<index>.json     →  "Rubric" column

The ``Both`` column is the per-item intersection of Z3 and Rubric, produced
by ``evaluate.py``. This package produces Table 3 of Zhang et al. (2026).
"""
