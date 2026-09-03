#!/usr/bin/env python3
"""Check canonical diagram inventory coverage and legacy connector regressions."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_ROOT = ROOT / "diagrams"
INDEX = DIAGRAM_ROOT / "INDEX.md"
LEGACY = (
    (re.compile(r"\+[-+]{2,}\+"), "legacy ASCII box border"),
    (re.compile(r"(?<![-=])(?:-->|->|<--|<-|<->|=>)"), "legacy ASCII arrow"),
    (re.compile(r"`[─━═┄┈]+[▶▷>]"), "backtick used as connector"),
)


paths = sorted(DIAGRAM_ROOT.glob("**/*.txt"))
inventory = INDEX.read_text(encoding="utf-8")
failures: list[str] = []

for path in paths:
    relative = path.relative_to(DIAGRAM_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    if f"({relative})" not in inventory:
        failures.append(f"diagrams/INDEX.md: missing {relative}")
    if "\t" in text:
        failures.append(f"{path.relative_to(ROOT)}: tabs are not portable")
    for number, line in enumerate(text.splitlines(), 1):
        for pattern, reason in LEGACY:
            if pattern.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{number}: {reason}")

for match in re.finditer(r"\(([^)]+\.txt)\)", inventory):
    path = DIAGRAM_ROOT / match.group(1)
    if not path.is_file():
        failures.append(f"diagrams/INDEX.md: missing target {match.group(1)}")

rows = re.findall(r"^\| D\d{3} \|", inventory, flags=re.MULTILINE)
if len(rows) != len(paths):
    failures.append(
        f"diagrams/INDEX.md: {len(rows)} inventory rows for {len(paths)} diagram files"
    )

if failures:
    raise SystemExit("\n".join(failures))

print(f"diagram style check passed: {len(paths)} canonical diagrams inventoried")
