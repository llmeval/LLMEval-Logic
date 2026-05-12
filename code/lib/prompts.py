"""External-prompt loader and markdown section extractor.

Keeps system prompts out of Python source so they can be version-controlled
as prose. Each feature folder ships its own prompt markdown files next to its
`core.py`; callers pass their `__file__` as `caller_file` so we resolve the
prompt path locally:

    load_prompt("system.md", caller_file=__file__)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_cache: Dict[Path, str] = {}


def load_prompt(name: str, *, caller_file: str) -> str:
    """Read a prompt file from the caller's directory (cached).

    `caller_file` is the `__file__` of the module calling this function; we
    resolve `name` relative to that directory. This keeps each feature's
    prompts inside its own folder.
    """
    path = (Path(caller_file).resolve().parent / name).resolve()
    cached = _cache.get(path)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8")
    _cache[path] = text
    return text


def extract_prompt_section(markdown: str, heading: str) -> str:
    """Extract a ```text fenced block that follows `## {heading}`."""
    pattern = re.compile(rf"## {re.escape(heading)}\n\n```text\n(.*?)\n```", re.S)
    match = pattern.search(markdown)
    if not match:
        raise ValueError(f"Missing prompt section: {heading}")
    return match.group(1).strip()
