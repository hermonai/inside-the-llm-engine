#!/usr/bin/env python3
"""Check structural parts of the mathematical editorial contract."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [
    ROOT / "manuscript" / "part-01" / "chapter-01-the-missing-half-of-ai.md",
    ROOT / "manuscript" / "part-01" / "chapter-02-from-text-to-tokens.md",
    ROOT / "manuscript" / "part-01" / "chapter-03-the-smallest-possible-language-model.md",
    ROOT / "manuscript" / "part-01" / "chapter-04-logits-sampling-autoregressive-loop.md",
    ROOT / "manuscript" / "part-02" / "chapter-05-tensors-without-magic.md",
    ROOT / "manuscript" / "part-02" / "chapter-06-matrix-multiplication-the-engine-room.md",
]
REQUIRED_IDS = {
    "LATENCY-TOTAL",
    "TOKEN-ROUNDTRIP",
    "OUTPUT-PROJECTION",
    "STABLE-SOFTMAX",
    "TENSOR-OFFSET",
    "GEMM-CONTRACTION",
    "ROOFLINE-BOUND",
}

failures: list[str] = []
blocks = 0
shape_blocks = 0
for path in CHAPTERS:
    text = path.read_text(encoding="utf-8")
    fences = [match.start() for match in re.finditer(r"(?m)^\$\$\s*$", text)]
    if len(fences) % 2:
        failures.append(f"{path.relative_to(ROOT)}: unmatched display-math fence")
    blocks += len(fences) // 2
    shape_blocks += text.count("\\in\\mathbb{R}")
    if not fences:
        failures.append(f"{path.relative_to(ROOT)}: no display mathematics")

index = (ROOT / "docs" / "MATH_INDEX.md").read_text(encoding="utf-8")
for equation_id in sorted(REQUIRED_IDS):
    if f"`{equation_id}`" not in index:
        failures.append(f"docs/MATH_INDEX.md: missing {equation_id}")

if "F_{\\mathrm{GEMM}}=2MKN" in CHAPTERS[-1].read_text(encoding="utf-8"):
    failures.append("Chapter 6: GEMM work estimate uses equality instead of approximation")

if failures:
    raise SystemExit("\n".join(failures))

print(
    f"math style check passed: {blocks} display blocks, "
    f"{shape_blocks} explicit real-valued shape declarations"
)
