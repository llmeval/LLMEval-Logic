"""JSON / JSONL IO and index slicing — collapses copies from batch_eval, formalize,
solve, judge, rubric, generate_rubric."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected JSON list at top level")
    return [item for item in data if isinstance(item, dict)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def write_jsonl(
    path: Path,
    items: Iterable[Dict[str, Any]],
    *,
    drop_underscore: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            if drop_underscore:
                item = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_indices(items: List[Dict[str, Any]]) -> None:
    """Backfill `id` field with positional value when absent or non-int."""
    for idx, item in enumerate(items):
        if not isinstance(item.get("id"), int):
            item["id"] = idx


def slice_by_index(
    items: List[Dict[str, Any]], start: Optional[int], end: Optional[int]
) -> List[Dict[str, Any]]:
    if start is None and end is None:
        return items
    ids = [item["id"] for item in items if isinstance(item.get("id"), int)]
    if not ids:
        return items
    lo = start if start is not None else min(ids)
    hi = end if end is not None else max(ids)
    if lo > hi:
        raise SystemExit(f"Invalid range: start={lo} > end={hi}")
    return [item for item in items if lo <= item.get("id", -1) <= hi]
