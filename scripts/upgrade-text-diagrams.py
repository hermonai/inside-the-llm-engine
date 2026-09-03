#!/usr/bin/env python3
"""Upgrade legacy ASCII connectors to portable Unicode box drawing.

The converter is intentionally conservative: it changes connector runs only
when neighboring characters prove that they belong to a diagram. Prose and
mathematical operators outside canonical ``diagrams/**/*.txt`` files are not
touched.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONNECTORS = set("+-|─│┌┐└┘├┤┬┴┼")
BOX = {
    (False, True, False, True): "─",
    (True, False, True, False): "│",
    (False, True, True, False): "┌",
    (False, False, True, True): "┐",
    (True, True, False, False): "└",
    (True, False, False, True): "┘",
    (False, True, True, True): "┬",
    (True, True, False, True): "┴",
    (True, True, True, False): "├",
    (True, False, True, True): "┤",
    (True, True, True, True): "┼",
}

HORIZONTAL = set("+-─┌┐└┘├┤┬┴┼")
VERTICAL = set("+|│┌┐└┘├┤┬┴┼")


def connected(lines: list[str], row: int, column: int, direction: str) -> bool:
    deltas = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
    dr, dc = deltas[direction]
    other_row, other_column = row + dr, column + dc
    if not (0 <= other_row < len(lines) and 0 <= other_column < len(lines[other_row])):
        return False
    current = lines[row][column]
    other = lines[other_row][other_column]
    if direction in {"left", "right"}:
        return current in HORIZONTAL and other in HORIZONTAL
    return current in VERTICAL and other in VERTICAL


def upgrade(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    for row, line in enumerate(lines):
        converted: list[str] = []
        for column, character in enumerate(line):
            if character not in CONNECTORS:
                converted.append(character)
                continue
            directions = (
                connected(lines, row, column, "up"),
                connected(lines, row, column, "right"),
                connected(lines, row, column, "down"),
                connected(lines, row, column, "left"),
            )
            replacement = BOX.get(directions)
            if replacement is None and character in "-─" and (directions[1] or directions[3]):
                replacement = "─"
            if replacement is None and character in "|│" and (directions[0] or directions[2]):
                replacement = "│"
            converted.append(replacement or character)
        upgraded = "".join(converted)
        upgraded = upgraded.replace("<==", "◀══").replace("==>", "══▶")
        upgraded = upgraded.replace("<--", "◀──").replace("-->", "──▶")
        upgraded = upgraded.replace(" - - > ", " ┄┄▶ ")
        upgraded = upgraded.replace("->", "─▶").replace("<-", "◀─")
        if upgraded.strip() == "v":
            upgraded = upgraded.replace("v", "▼")
        elif upgraded.strip() == "^":
            upgraded = upgraded.replace("^", "▲")
        output.append(upgraded)
    return "\n".join(output) + "\n"


def main() -> None:
    for path in sorted((ROOT / "diagrams").glob("**/*.txt")):
        original = path.read_text(encoding="utf-8")
        upgraded = upgrade(original)
        if upgraded != original:
            path.write_text(upgraded, encoding="utf-8")


if __name__ == "__main__":
    main()
