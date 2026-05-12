"""CLI: score candidate FL against a rubric.

The 5 ablation flags from the legacy CLI (--no-z3, --no-llm-align,
--no-premise-equiv, --no-pe-soft-review, --no-freefl-no-z3-lr-sc) are
collapsed into a single `--mode {fixed,free}`:

  fixed: candidate reuses gold's parameters/translation. Run PE gate,
         per-item z3 entails, LLM soft-review on PE diffs (when answer matches).
  free:  candidate has its own symbol space. Skip PE entirely. LR/SC are
         judged by the LLM (with the over-strength check). QA still goes
         through z3 query_equiv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .core import DEFAULT_RUBRIC_MODEL, _is_fixed_fl, run_score
from ...lib.bench_utils import public_formalization, load_bench
from ...lib.io_utils import load_jsonl


_MODE_PRESETS = {
    "fixed": {
        "enable_z3_prefilter": True,
        "enable_llm_align": True,
        "enable_premise_equiv": True,
        "enable_pe_soft_review": True,
        "enable_freefl_no_z3_lr_sc": False,
    },
    "free": {
        "enable_z3_prefilter": True,
        "enable_llm_align": True,
        "enable_premise_equiv": False,
        "enable_pe_soft_review": False,
        "enable_freefl_no_z3_lr_sc": True,
    },
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate FL against a rubric.")
    parser.add_argument(
        "--mode",
        choices=("fixed", "free"),
        required=True,
        help="fixed: candidate reuses gold parameters/translation; free: candidate uses its own symbol space.",
    )
    parser.add_argument("--bench", default="bench/final_clean.json")
    parser.add_argument("--candidates", required=True, help="Candidate FL JSONL (formalize.jsonl).")
    parser.add_argument("--model", default=DEFAULT_RUBRIC_MODEL)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--rubric-output", required=False, help="Output path for newly-generated rubric JSONL.")
    parser.add_argument("--rubric-summary", required=False, help="Summary JSON for rubric generation.")
    parser.add_argument(
        "--use-existing-rubric",
        default=None,
        help="Path to a previously generated rubric JSONL. If set, rubric generation is skipped and must cover every sample_id in --candidates.",
    )
    parser.add_argument(
        "--use-bench-rubric",
        action="store_true",
        default=False,
        help="Load per-problem rubrics from --rubric-dir (named {index:03d}.json). Mutually exclusive with --use-existing-rubric.",
    )
    parser.add_argument(
        "--rubric-dir",
        default="bench/base/rubrics",
        help="Directory of per-problem rubric JSON files used when --use-bench-rubric is set (default: bench/base/rubrics).",
    )
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--score-summary", required=True)
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Delete any existing score JSONL before scoring (default: resume, skipping sample_ids already present).",
    )
    return parser.parse_args(argv)


def _check_mode_consistency(mode: str, bench_path: Path, candidates_path: Path) -> None:
    """Sample a few candidates and warn (don't exit) if --mode contradicts the data.

    fixed: candidate parameters/translation should match gold.
    free:  candidate parameters/translation should NOT match gold.
    """
    try:
        bench = load_bench(bench_path)
        candidates = load_jsonl(candidates_path)
    except Exception:
        return
    sample = candidates[: min(5, len(candidates))]
    mismatches = 0
    checked = 0
    for cand in sample:
        sid = cand.get("sample_id")
        gold_sample = bench.get(sid)
        if gold_sample is None:
            continue
        gold_fl = public_formalization(gold_sample)
        cand_fl = cand.get("FL") if isinstance(cand.get("FL"), dict) else None
        is_fixed = _is_fixed_fl(cand_fl, gold_fl)
        checked += 1
        if mode == "fixed" and not is_fixed:
            mismatches += 1
        elif mode == "free" and is_fixed:
            mismatches += 1
    if checked > 0 and mismatches > 0:
        print(
            f"[warn] --mode {mode} but {mismatches}/{checked} sampled candidates look like the opposite mode. "
            f"Continuing anyway.",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.use_bench_rubric and args.use_existing_rubric:
        raise SystemExit("--use-bench-rubric and --use-existing-rubric are mutually exclusive")
    if args.fresh and Path(args.score_output).exists():
        Path(args.score_output).unlink()

    preset = _MODE_PRESETS[args.mode]
    _check_mode_consistency(args.mode, Path(args.bench), Path(args.candidates))

    run_score(
        bench_path=Path(args.bench),
        candidates_path=Path(args.candidates),
        rubric_path=Path(args.rubric_output) if args.rubric_output else None,
        score_output_path=Path(args.score_output),
        score_summary_path=Path(args.score_summary),
        reuse_rubric_path=Path(args.use_existing_rubric) if args.use_existing_rubric else None,
        reuse_rubric_dir=Path(args.rubric_dir) if args.use_bench_rubric else None,
        rubric_summary_path=Path(args.rubric_summary) if args.rubric_summary else None,
        model=args.model,
        timeout=args.timeout,
        concurrency=args.concurrency,
        **preset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
