#!/usr/bin/env python3
"""Validate canonical diagram display width with Unicode-aware accounting."""

from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 100


def display_width(text: str) -> int:
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


failures: list[str] = []
for path in sorted((ROOT / "diagrams").glob("**/*.txt")):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        width = display_width(line)
        if width > LIMIT:
            failures.append(f"{path.relative_to(ROOT)}:{number}: display width {width} > {LIMIT}")

if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)

count = len(list((ROOT / "diagrams").glob("**/*.txt")))
print(f"diagram width check passed: {count} Unicode text diagrams, limit {LIMIT}")
