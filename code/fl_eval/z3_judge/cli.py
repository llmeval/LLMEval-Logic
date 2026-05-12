"""CLI for the Z3 solver stage. Drop-in replacement for the previous
`code/generate/solve_from_formal.py` entrypoint with a slimmer flag set
(legacy backward-compat flags removed)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .core import run


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve candidate Formal Language JSON with Z3. The candidate FL must provide "
            "parameters, translation, premise, and question; any answer field is ignored."
        )
    )
    parser.add_argument("--input", required=True, help="Input candidate formalization JSONL file.")
    parser.add_argument("--data", default=None, help="Optional original data JSON for reference answers.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive, by record index).")
    parser.add_argument("--end", type=int, default=None, help="End index (inclusive, by record index).")
    parser.add_argument("--z3-timeout-ms", type=int, default=5000, help="Z3 timeout per check in milliseconds.")
    parser.add_argument(
        "--z3-max-enumerate-assignments",
        type=int,
        default=4096,
        help=(
            "Maximum boolean assignments to enumerate for enumerate_models/count_models. "
            "Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--z3-record-timeout-s",
        type=float,
        default=60.0,
        help=(
            "Best-effort wall-clock timeout per record for enumeration-heavy queries. "
            "Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print Z3 solve progress every N records. Use 0 to disable progress logs.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    max_enumerate_assignments = (
        args.z3_max_enumerate_assignments
        if args.z3_max_enumerate_assignments > 0
        else None
    )
    record_timeout_s = args.z3_record_timeout_s if args.z3_record_timeout_s > 0 else None
    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        data_path=Path(args.data) if args.data else None,
        summary_path=Path(args.summary) if args.summary else None,
        start=args.start,
        end=args.end,
        max_samples=args.max_samples,
        z3_timeout_ms=args.z3_timeout_ms,
        max_enumerate_assignments=max_enumerate_assignments,
        record_timeout_s=record_timeout_s,
        progress_every=max(args.progress_every, 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
