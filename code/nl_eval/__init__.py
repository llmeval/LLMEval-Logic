"""NL evaluation track (Answer Accuracy).

Consumes the natural-language items in ``bench/base/`` and ``bench/hard/`` and
scores model answers against the gold answers via LLM-as-Judge. Produces the
``Item Accuracy`` and ``Sub-Q Accuracy`` columns of the paper leaderboard
(Table 2 of Zhang et al., 2026).

Submodules
----------
- ``eval``       : NL eval driver (model generation + answer-side judging).
- ``llm_judge``  : LLM-as-Judge implementation used for both NL and FL tracks.
"""
