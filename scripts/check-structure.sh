#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

required='README.md
BOOK.md
AGENTS.md
GLOSSARY.md
CONTRIBUTING.md
docs/OUTLINE.md
docs/BOOK_CONSTITUTION.md
docs/CHAPTER_CONTRACT.md
docs/SOURCE_POLICY.md
docs/CODE_POLICY.md
docs/MATH_STYLE.md
docs/STYLE_GUIDE.md
docs/TERMINOLOGY.md
docs/BENCHMARK_POLICY.md
docs/AUTHORING_WORKFLOW.md
docs/AI_AUTHORING.md
docs/ROADMAP.md
docs/STATUS.md
docs/LABS.md
research/README.md
research/hermon/README.md
diagrams/README.md'

printf '%s\n' "$required" | while IFS= read -r file; do
    if [ ! -s "$file" ]; then
        echo "missing or empty required file: $file" >&2
        exit 1
    fi
done

part=1
while [ "$part" -le 15 ]; do
    dir=$(printf 'manuscript/part-%02d' "$part")
    if [ ! -s "$dir/README.md" ]; then
        echo "missing part index: $dir/README.md" >&2
        exit 1
    fi
    part=$((part + 1))
done

chapter_count=$(grep -Ec '^### Chapter [0-9]+ — ' docs/OUTLINE.md)
if [ "$chapter_count" -ne 94 ]; then
    echo "expected 94 chapter specifications, found $chapter_count" >&2
    exit 1
fi

expected=1
grep -E '^### Chapter [0-9]+ — ' docs/OUTLINE.md | while IFS= read -r heading; do
    actual=$(printf '%s\n' "$heading" | sed -E 's/^### Chapter ([0-9]+).*/\1/')
    if [ "$actual" -ne "$expected" ]; then
        echo "chapter sequence error: expected $expected, found $actual" >&2
        exit 1
    fi
    expected=$((expected + 1))
done

for field in \
    'Purpose / key question' \
    'Prerequisites' \
    'Concepts' \
    'Mathematics / systems / hardware' \
    'Implementation / Hermon / external' \
    'Diagrams / experiments' \
    'Correctness / benchmark' \
    'Misconceptions / failures / deliverable / next'
do
    count=$(grep -Fc -- "**$field:**" docs/OUTLINE.md)
    if [ "$count" -ne 94 ]; then
        echo "expected 94 '$field' fields, found $count" >&2
        exit 1
    fi
done

if grep -R -n '```mermaid' README.md BOOK.md docs manuscript research diagrams; then
    echo "Mermaid found in Markdown-first book infrastructure" >&2
    exit 1
fi

echo "structure check passed: 15 parts, 94 complete chapter specifications"
