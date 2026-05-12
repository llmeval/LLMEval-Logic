"""CLI for the judge stage."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .core import DEFAULT_JUDGE_MODEL, run


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-judge results.jsonl for semantic match.")
    parser.add_argument("--input", default="results.jsonl", help="Input results JSONL file.")
    parser.add_argument("--output", default="results_judged.jsonl", help="Output JSONL file.")
    parser.add_argument("--summary", default="summary.json", help="Summary JSON file.")
    parser.add_argument(
        "--bench",
        default=None,
        help="Optional bench JSON used to add title, background, and question context.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--judge-concurrency", type=int, default=4, help="Judge concurrency.")
    parser.add_argument("--judge-batch-size", type=int, default=5, help="Judge batch size.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds).")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End index (inclusive).")
    parser.add_argument("--no-json-mode", action="store_true", help="Disable response_format json_object.")
    parser.add_argument("--store-raw", action="store_true", help="Store raw judge responses.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output.")
    parser.add_argument("--overwrite", action="store_true", help="Re-judge even if judge_match exists.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        bench_path=Path(args.bench) if args.bench else None,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
        batch_size=args.judge_batch_size,
        timeout=args.timeout,
        start=args.start,
        end=args.end,
        use_json_mode=not args.no_json_mode,
        store_raw=args.store_raw,
        show_progress=not args.no_progress,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
