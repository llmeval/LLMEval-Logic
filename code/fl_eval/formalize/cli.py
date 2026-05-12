#!/usr/bin/env python3
"""CLI for the formalize stage (NL -> solver-compatible Formal Language JSON).

This module is the entry point of the FL evaluation track and mirrors
``code.nl_eval.eval``'s style: multi-model, resumable, per-item checkpoints.
Two modes are supported:

* ``--mode free``  (default): the model generates the full FL
  (parameters / translation / premise / question / reason).
* ``--mode fixed``: the bench item's ``formalization.parameters`` and
  ``formalization.translation`` are injected as read-only context; the model
  only emits ``premise`` / ``question`` / ``reason``.

The set of model keys you can pass to ``--models`` is whatever you have
registered in ``code/client.py``; unregistered keys fall through to the
OpenAI-compatible default.

Examples
--------
::

    # Smoke test: 2 items, 1 model, free mode
    python -m code.fl_eval.formalize.cli \
        --input bench/base/llmeval_logic_base.json \
        --output-dir output/formalize/_smoke_free \
        --mode free \
        --models gpt-4o \
        --limit 2

    # Fixed mode: bench parameters/translation injected as read-only context
    python -m code.fl_eval.formalize.cli \
        --input bench/base/llmeval_logic_base.json \
        --output-dir output/formalize/base_fixed \
        --mode fixed \
        --models gpt-4o claude-sonnet-4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional

from ...client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    list_models,
    load_dotenv,
)
from .core import ALLOWED_MODES, MODE_FREE, run_async


class TeeLogger:
    """Write log lines to both stdout and a file (matches eval.py)."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate solver-compatible Formal Language JSON for LogicBench items.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input bench JSON list (e.g. bench/base/llmeval_logic_base.json).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write detail_results/, formalize.<model>.jsonl, and formalize_summary.json.")
    parser.add_argument("--mode", choices=list(ALLOWED_MODES), default=MODE_FREE,
                        help="free = full FL generation; fixed = use existing parameters/translation as read-only context.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model keys understood by code/client.py "
                             "(any registered key in MODEL_CONFIGS, or a model id "
                             "forwarded as-is to the OpenAI-compatible default backend).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N items (smoke test).")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Global in-flight cap across ALL models.")
    parser.add_argument("--per-model-concurrency", type=int, default=8,
                        help="Per-model in-flight cap.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-request timeout passed to call_model.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Per-request max_tokens passed to call_model.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Retries per request inside client.call_model.")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if a successful checkpoint already exists.")
    parser.add_argument("--dotenv", default=".env",
                        help="Path to .env containing API credentials (relative to CWD).")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    load_dotenv(args.dotenv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(output_dir / "run.log")
    try:
        logger.log(f"cmd: {' '.join(sys.argv)}")
        logger.log(f"input={args.input}  output_dir={output_dir}  mode={args.mode}")
        logger.log(f"models={args.models}  limit={args.limit}  force={args.force}")
        logger.log(
            f"concurrency={args.concurrency}  per_model_concurrency={args.per_model_concurrency}  "
            f"timeout={args.timeout}  max_tokens={args.max_tokens}  max_retries={args.max_retries}"
        )
        summary = await run_async(
            input_path=Path(args.input),
            output_dir=output_dir,
            models=list(args.models),
            mode=args.mode,
            concurrency=args.concurrency,
            per_model_concurrency=args.per_model_concurrency,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            limit=args.limit,
            force=args.force,
            log_fn=logger.log,
        )
        # Echo a compact one-line stats per model so the operator can sanity-check.
        for m in summary.get("model_metrics", []):
            logger.log(
                f"summary model={m['model']} fl_ok={m['fl_ok']}/{m['total']} "
                f"({m['fl_ok_rate']:.1%})  request_err={m['request_error_count']}  "
                f"parse_err={m['parse_error_count']}  avg_latency={m['avg_latency_ms']}ms"
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        logger.close()


def main() -> int:
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
