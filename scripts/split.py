#!/usr/bin/env python3
"""Reproducible 80/20 stratified split for LLMEval-Logic.

This script is what produced the public 80% release. Running it on the full
corpus (Base 246 + Hard 190 + 246 rubrics, kept internally by Fudan NLP Lab)
yields exactly the same public set that ships in ``bench/`` and the private
holdout that backs the official LLMEval-Logic leaderboard. It is published so
that the public split can be independently audited.

Strategy
========

Base (246 items)
    Stratify by a derived ``label_class`` from the ``label_type`` field:

        ``enum``    if ``enumerate_models`` in labels
        ``count``   if ``count_models``     in labels
        ``pos+nec`` if ``necessary`` and ``possible`` both in labels
        ``nec``     if ``necessary`` in labels
        ``pos``     if ``possible``  in labels
        otherwise   ``other``

    Strata with fewer than 5 items get pooled into ``__small_pool`` so we
    never produce a stratum whose private bucket would be zero.

    (Note: ``logictype`` is *not* part of the stratification key. The PL/FOL
    proportion is preserved as a byproduct — see ``bench_private/SPLIT_STATS.json``
    in the maintainers' working copy for the empirical breakdown.)

Hard (190 items)
    Stratify by the sub-question-count bucket of each item:

        2-4   ->  ``small``
        5     ->  ``medium``    (mode of the distribution)
        6-8   ->  ``large``

Rubrics (246 per-problem files under ``bench/base/rubrics/``)
    The rubric set follows the Base split: rubrics for public Base items go
    public, rubrics for private Base items go private. Rubrics are not
    independently sampled.

Seed: ``random.Random(2026)`` (year of release) → deterministic.

Input layout (run on the maintainers' full corpus)
==================================================

    bench_full/
    ├── llmeval_logic_base.json    246 items
    ├── llmeval_logic_hard.json    190 items
    └── rubrics/                   246 per-problem rubrics

Output layout
=============

    bench/                         the 80% public release this script writes
    ├── base/
    │   ├── llmeval_logic_base.json    197 items
    │   └── rubrics/                   197 per-problem rubrics
    └── hard/
        └── llmeval_logic_hard.json    152 items

    bench_private/                 the 20% private holdout (NEVER push)
    ├── llmeval_logic_base.json
    ├── llmeval_logic_hard.json
    ├── rubrics/
    └── SPLIT_STATS.json           per-stratum counts (size only, no item ids)

Usage
=====

    python scripts/split.py \\
        --full-corpus bench_full \\
        --public-out bench \\
        --private-out bench_private

The corresponding ``bench_full/`` directory is not part of the public
release; ask the maintainers if you need access for auditing.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

SEED = 2026
PRIVATE_FRACTION = 0.20


# -- Stratifier functions ---------------------------------------------------

def label_class(item: dict) -> str:
    labels = set(item.get("label_type", []) or [])
    if "enumerate_models" in labels:
        return "enum"
    if "count_models" in labels:
        return "count"
    if "necessary" in labels and "possible" in labels:
        return "pos+nec"
    if "necessary" in labels:
        return "nec"
    if "possible" in labels:
        return "pos"
    return "other"


def subq_bucket(item: dict) -> str:
    n = len(item.get("question", []) or [])
    if n <= 4:
        return "small"
    if n == 5:
        return "medium"
    return "large"


# -- Core split -------------------------------------------------------------

def stratified_split(
    items: List[dict],
    strat_fn: Callable[[dict], str],
    rng: random.Random,
    frac_private: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Deterministic stratified random sampling.

    Pools any stratum with fewer than 5 items into ``__small_pool`` so we
    never produce an empty private bucket. Within each stratum we shuffle
    once with ``rng`` and clamp ``n_private`` to ``[1, n-1]`` so neither
    side of the split is empty.
    """
    by_stratum: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_stratum[str(strat_fn(it))].append(it)

    too_small = [k for k, v in by_stratum.items() if len(v) < 5]
    if too_small:
        pooled = []
        for k in too_small:
            pooled.extend(by_stratum.pop(k))
        by_stratum["__small_pool"] = pooled

    public_items: List[dict] = []
    private_items: List[dict] = []
    stratum_stats: List[dict] = []
    for stratum, group in sorted(by_stratum.items()):
        rng.shuffle(group)
        n_private = round(len(group) * frac_private)
        n_private = max(1, min(len(group) - 1, n_private))
        private_part = group[:n_private]
        public_part = group[n_private:]
        public_items.extend(public_part)
        private_items.extend(private_part)
        stratum_stats.append({
            "stratum": stratum,
            "total": len(group),
            "public": len(public_part),
            "private": len(private_part),
        })

    public_items.sort(key=lambda it: it["id"])
    private_items.sort(key=lambda it: it["id"])
    return public_items, private_items, stratum_stats


# -- I/O orchestration ------------------------------------------------------

def _write_split(
    out_root: Path,
    base_items: List[dict],
    hard_items: List[dict],
    rubric_dir_src: Path,
    rubric_ids: set,
) -> None:
    """Lay out the new bench tree at ``out_root``: base/ + hard/."""
    base_dir = out_root / "base"
    hard_dir = out_root / "hard"
    rubric_dst = base_dir / "rubrics"
    base_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)
    rubric_dst.mkdir(parents=True, exist_ok=True)

    (base_dir / "llmeval_logic_base.json").write_text(
        json.dumps(base_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (hard_dir / "llmeval_logic_hard.json").write_text(
        json.dumps(hard_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for r in sorted(rubric_dir_src.iterdir()):
        if not r.name.endswith(".json"):
            continue
        try:
            pid = int(r.stem)
        except ValueError:
            continue
        if pid in rubric_ids:
            shutil.copy(r, rubric_dst / r.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full-corpus", required=True,
        help="Path to the maintainers' full bench tree "
             "(must contain llmeval_logic_base.json, llmeval_logic_hard.json, rubrics/).",
    )
    parser.add_argument(
        "--public-out", default="bench",
        help="Where to write the public 80%% (will create base/ and hard/ inside).",
    )
    parser.add_argument(
        "--private-out", default="bench_private",
        help="Where to write the private 20%% (will create base/ and hard/ inside).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--private-fraction", type=float, default=PRIVATE_FRACTION)
    args = parser.parse_args()

    src = Path(args.full_corpus)
    base = json.loads((src / "llmeval_logic_base.json").read_text(encoding="utf-8"))
    hard = json.loads((src / "llmeval_logic_hard.json").read_text(encoding="utf-8"))
    rubric_dir = src / "rubrics"
    if not rubric_dir.is_dir():
        raise SystemExit(f"Missing rubric directory: {rubric_dir}")

    rng = random.Random(args.seed)
    base_pub, base_priv, base_strata = stratified_split(
        base, label_class, rng, args.private_fraction,
    )
    hard_pub, hard_priv, hard_strata = stratified_split(
        hard, subq_bucket, rng, args.private_fraction,
    )

    pub_ids = {it["id"] for it in base_pub}
    priv_ids = {it["id"] for it in base_priv}

    pub_out = Path(args.public_out)
    priv_out = Path(args.private_out)
    _write_split(pub_out, base_pub, hard_pub, rubric_dir, pub_ids)
    _write_split(priv_out, base_priv, hard_priv, rubric_dir, priv_ids)

    stats = {
        "seed": args.seed,
        "private_fraction_target": args.private_fraction,
        "base": {
            "total": len(base),
            "public": len(base_pub),
            "private": len(base_priv),
            "private_fraction_actual": round(len(base_priv) / len(base), 4),
            "strata": base_strata,
        },
        "hard": {
            "total": len(hard),
            "public": len(hard_pub),
            "private": len(hard_priv),
            "private_fraction_actual": round(len(hard_priv) / len(hard), 4),
            "strata": hard_strata,
        },
        "rubrics": {
            "total": len(list(rubric_dir.glob("*.json"))),
            "public": len(pub_ids),
            "private": len(priv_ids),
        },
    }
    (priv_out / "SPLIT_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"public  -> {pub_out}/  ({len(base_pub)} base, {len(hard_pub)} hard, {len(pub_ids)} rubrics)")
    print(f"private -> {priv_out}/ ({len(base_priv)} base, {len(hard_priv)} hard, {len(priv_ids)} rubrics)")
    print(f"stats   -> {priv_out / 'SPLIT_STATS.json'}")


if __name__ == "__main__":
    main()
