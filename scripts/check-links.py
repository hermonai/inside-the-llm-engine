#!/usr/bin/env python3
"""Check repository-relative Markdown links without external dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("#", "http://", "https://", "mailto:", "data:")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and "target" not in path.parts
    )


def target_path(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(SKIP_PREFIXES):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return None
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    return document.parent / target


def main() -> int:
    failures: list[str] = []
    for document in markdown_files():
        in_fence = False
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in LINK.finditer(line):
                resolved = target_path(document, match.group(1))
                if resolved is not None and not resolved.exists():
                    relative = document.relative_to(ROOT)
                    failures.append(
                        f"{relative}:{line_number}: missing link target {match.group(1)!r}"
                    )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"link check passed: {len(markdown_files())} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
