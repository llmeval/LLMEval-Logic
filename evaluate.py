#!/usr/bin/env python3
"""LLMEval-Logic one-command evaluator.

Runs every stage of the benchmark for a single candidate model and prints a
final scoreboard. Defaults to OpenRouter via ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` in your `.env`; any OpenAI-compatible endpoint works.

Stages (skip any with --skip ...):

  nl-base    code.nl_eval.eval on bench/base/llmeval_logic_base.json   → Item / Sub-Q Acc
  nl-hard    code.nl_eval.eval on bench/hard/llmeval_logic_hard.json   → Item / Sub-Q Acc
  fl-free    code.fl_eval.{formalize,z3_judge,rubric_judge} on Base, candidate has its own symbol space
  fl-fixed   code.fl_eval.{formalize,z3_judge,rubric_judge} on Base, candidate reuses gold parameters/translation

Each stage is fully resumable: re-running with the same --output-dir picks
up where the previous run left off.

Examples
--------
# Evaluate any OpenAI-compatible model id you can call directly:
python evaluate.py --model openai/gpt-4o

# Evaluate a model registered in code/client.py via a friendly key:
python evaluate.py --model hy3-nothink --judge-model openai/gpt-4o

# Quick sanity run on the first 3 items only:
python evaluate.py --model openai/gpt-4o --limit 3
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH_BASE = ROOT / "bench" / "base" / "llmeval_logic_base.json"
DEFAULT_BENCH_HARD = ROOT / "bench" / "hard" / "llmeval_logic_hard.json"
DEFAULT_RUBRIC_DIR = ROOT / "bench" / "base" / "rubrics"

ALL_STAGES = ("nl-base", "nl-hard", "fl-free", "fl-fixed")


def _safe_model_name(model: str) -> str:
    """Mirror code.fl_eval.formalize.core.sanitize_name so filenames line up."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def _run(cmd: List[str], *, label: str) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[{label}] $ {pretty}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] exited with code {proc.returncode}")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def stage_nl(
    *,
    model: str,
    judge_model: str,
    bench_path: Path,
    out_dir: Path,
    limit: Optional[int],
    label: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "code.nl_eval.eval",
        "--input",
        str(bench_path),
        "--output-dir",
        str(out_dir),
        "--models",
        model,
        "--judge-model",
        judge_model,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    _run(cmd, label=label)
    summary = _read_json(out_dir / "eval_summary.json") or {}
    for entry in summary.get("model_metrics") or []:
        if entry.get("model") == model:
            return {
                "item_acc": (entry.get("question_exact_match_accuracy") or 0.0) * 100.0,
                "subq_acc": (entry.get("subquestion_accuracy") or 0.0) * 100.0,
                "n_items": entry.get("question_total"),
                "n_subq": entry.get("subquestion_total"),
                "parse_fail": entry.get("parse_fail_count"),
                "request_error": entry.get("request_error_count"),
            }
    return {"item_acc": None, "subq_acc": None}


def stage_fl(
    *,
    model: str,
    judge_model: str,
    bench_path: Path,
    rubric_dir: Path,
    out_root: Path,
    mode: str,
    limit: Optional[int],
) -> Dict[str, Any]:
    safe = _safe_model_name(model)
    formalize_dir = out_root / f"formalize_{mode}"
    formalize_dir.mkdir(parents=True, exist_ok=True)
    candidate_jsonl = formalize_dir / f"formalize.{safe}.jsonl"

    cmd = [
        sys.executable,
        "-m",
        "code.fl_eval.formalize.cli",
        "--input",
        str(bench_path),
        "--output-dir",
        str(formalize_dir),
        "--mode",
        mode,
        "--models",
        model,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    _run(cmd, label=f"fl-{mode}/formalize")

    if not candidate_jsonl.exists():
        for cand in formalize_dir.glob("formalize.*.jsonl"):
            if cand.name != "formalize.jsonl":
                candidate_jsonl = cand
                break

    judge_dir = out_root / f"judge_{mode}"
    judge_dir.mkdir(parents=True, exist_ok=True)
    solve_jsonl = judge_dir / f"solve.{safe}.jsonl"
    judged_jsonl = judge_dir / f"judged.{safe}.jsonl"
    score_jsonl = judge_dir / f"score.{safe}.jsonl"
    solve_summary = judge_dir / f"solve.{safe}.summary.json"
    judged_summary = judge_dir / f"judged.{safe}.summary.json"
    score_summary = judge_dir / f"score.{safe}.summary.json"

    _run(
        [
            sys.executable,
            "-m",
            "code.fl_eval.z3_judge.cli",
            "--input",
            str(candidate_jsonl),
            "--data",
            str(bench_path),
            "--output",
            str(solve_jsonl),
            "--summary",
            str(solve_summary),
            "--progress-every",
            "50",
        ],
        label=f"fl-{mode}/z3_judge",
    )

    _run(
        [
            sys.executable,
            "-m",
            "code.nl_eval.llm_judge.cli",
            "--input",
            str(solve_jsonl),
            "--output",
            str(judged_jsonl),
            "--summary",
            str(judged_summary),
            "--bench",
            str(bench_path),
            "--judge-model",
            judge_model,
            "--judge-concurrency",
            "10",
            "--judge-batch-size",
            "10",
            "--timeout",
            "300",
            "--overwrite",
        ],
        label=f"fl-{mode}/llm_judge",
    )

    _run(
        [
            sys.executable,
            "-m",
            "code.fl_eval.rubric_judge.cli",
            "--mode",
            mode,
            "--bench",
            str(bench_path),
            "--candidates",
            str(candidate_jsonl),
            "--use-bench-rubric",
            "--rubric-dir",
            str(rubric_dir),
            "--model",
            judge_model,
            "--concurrency",
            "10",
            "--timeout",
            "300",
            "--score-output",
            str(score_jsonl),
            "--score-summary",
            str(score_summary),
            "--fresh",
        ],
        label=f"fl-{mode}/rubric_judge",
    )

    return _summarize_fl(
        bench_path=bench_path,
        judged_jsonl=judged_jsonl,
        score_jsonl=score_jsonl,
    )


def _summarize_fl(*, bench_path: Path, judged_jsonl: Path, score_jsonl: Path) -> Dict[str, Any]:
    """Compute Z3 / Rubric / Both percentages from the judge JSONLs."""
    judged_match: Dict[int, bool] = {}
    if judged_jsonl.exists():
        with judged_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = rec.get("sample_id")
                if not isinstance(sid, int):
                    continue
                judged_match[sid] = rec.get("judge_match") is True

    rubric_pass: Dict[int, bool] = {}
    if score_jsonl.exists():
        with score_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                pid = rec.get("problem_id")
                if not isinstance(pid, int):
                    continue
                total = rec.get("total") or 0
                passed = rec.get("passed") or 0
                rubric_pass[pid] = total > 0 and passed == total

    total = max(len(judged_match), len(rubric_pass))
    if total == 0:
        return {"z3": None, "rubric": None, "both": None, "n": 0}

    z3 = sum(judged_match.values()) / total * 100
    rubric = sum(rubric_pass.values()) / total * 100
    both_n = sum(1 for sid in judged_match if judged_match.get(sid) and rubric_pass.get(sid))
    both = both_n / total * 100
    return {"z3": z3, "rubric": rubric, "both": both, "n": total}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command LLMEval-Logic evaluator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id passed to your OpenAI-compatible endpoint (e.g. "
        "openai/gpt-4o, anthropic/claude-3.5-sonnet, tencent/hy3-preview), "
        "OR a friendly key registered in code/client.py:MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o",
        help="LLM-as-judge model. Same routing as --model. Default: openai/gpt-4o.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Where to write generation + judging artifacts (default: ./output).",
    )
    parser.add_argument(
        "--bench-base",
        default=str(DEFAULT_BENCH_BASE),
        help=f"Base bench path (default: {DEFAULT_BENCH_BASE.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--bench-hard",
        default=str(DEFAULT_BENCH_HARD),
        help=f"Hard bench path (default: {DEFAULT_BENCH_HARD.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--rubric-dir",
        default=str(DEFAULT_RUBRIC_DIR),
        help=f"Per-problem rubric directory (default: {DEFAULT_RUBRIC_DIR.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of items per stage (for smoke tests).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=ALL_STAGES,
        help="Stages to skip. By default every stage runs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=ALL_STAGES,
        help="Run only the listed stages (overrides --skip).",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    safe = _safe_model_name(args.model)

    if args.only is not None:
        stages = list(args.only)
    else:
        stages = [s for s in ALL_STAGES if s not in args.skip]
    print(f"[plan] model={args.model!r}  judge={args.judge_model!r}  stages={stages}", flush=True)

    results: Dict[str, Any] = {}

    if "nl-base" in stages:
        results["nl-base"] = stage_nl(
            model=args.model,
            judge_model=args.judge_model,
            bench_path=Path(args.bench_base),
            out_dir=out_root / "eval_base" / safe,
            limit=args.limit,
            label="nl-base",
        )
    if "nl-hard" in stages:
        results["nl-hard"] = stage_nl(
            model=args.model,
            judge_model=args.judge_model,
            bench_path=Path(args.bench_hard),
            out_dir=out_root / "eval_hard" / safe,
            limit=args.limit,
            label="nl-hard",
        )
    if "fl-free" in stages:
        results["fl-free"] = stage_fl(
            model=args.model,
            judge_model=args.judge_model,
            bench_path=Path(args.bench_base),
            rubric_dir=Path(args.rubric_dir),
            out_root=out_root,
            mode="free",
            limit=args.limit,
        )
    if "fl-fixed" in stages:
        results["fl-fixed"] = stage_fl(
            model=args.model,
            judge_model=args.judge_model,
            bench_path=Path(args.bench_base),
            rubric_dir=Path(args.rubric_dir),
            out_root=out_root,
            mode="fixed",
            limit=args.limit,
        )

    print("\n" + "=" * 64, flush=True)
    print(f"  Final scoreboard for: {args.model}", flush=True)
    print("=" * 64, flush=True)

    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "  -- "
        return f"{x:5.1f}"

    if "nl-base" in results:
        s = results["nl-base"] or {}
        print(f"  NL Base   Item Acc   {_fmt_pct(s.get('item_acc'))}   "
              f"Sub-Q Acc   {_fmt_pct(s.get('subq_acc'))}", flush=True)
    if "nl-hard" in results:
        s = results["nl-hard"] or {}
        print(f"  NL Hard   Item Acc   {_fmt_pct(s.get('item_acc'))}   "
              f"Sub-Q Acc   {_fmt_pct(s.get('subq_acc'))}", flush=True)
    for mode in ("free", "fixed"):
        key = f"fl-{mode}"
        if key in results:
            s = results[key] or {}
            print(
                f"  FL {mode:<5} Z3 {_fmt_pct(s.get('z3'))}   "
                f"Rubric {_fmt_pct(s.get('rubric'))}   Both {_fmt_pct(s.get('both'))}   "
                f"(n={s.get('n', 0)})",
                flush=True,
            )

    summary_path = out_root / f"summary_{safe}.json"
    summary_path.write_text(
        json.dumps({"model": args.model, "judge_model": args.judge_model, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[saved] {summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
