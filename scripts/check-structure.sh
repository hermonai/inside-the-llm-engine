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
research/part-01/README.md
research/part-01/chapter-01-the-missing-half-of-ai.md
research/part-01/chapter-02-from-text-to-tokens.md
research/part-01/tokenizer-comparison.md
diagrams/README.md'

printf '%s\n' "$required" | while IFS= read -r file; do
    if [ ! -s "$file" ]; then
        echo "missing or empty required file: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-01/chapter-02-from-text-to-tokens.md \
    code/reference/chapter-02-tokenizer-oracles.md \
    code/experiments/tokenizer-comparison/compare.py \
    code/experiments/tokenizer-comparison/README.md \
    code/mini-engine/fixtures/tokenizer/tiny-bpe.txt \
    code/mini-engine/fixtures/tokenizer/utf8-stream.txt \
    code/mini-engine/fixtures/tokenizer/chat-template.txt \
    labs/lab-02-tokenize-by-hand.md \
    labs/lab-03-stream-utf8-across-tokens.md \
    labs/lab-04-use-the-wrong-chat-template.md \
    diagrams/tokenizer/text-unicode-bytes-tokens.txt \
    diagrams/tokenizer/bpe-merge-process.txt \
    diagrams/tokenizer/chat-template-pipeline.txt \
    diagrams/tokenizer/token-to-byte-stream.txt \
    diagrams/tokenizer/utf8-partial-token-boundary.txt \
    diagrams/tokenizer/model-tokenizer-template-contract.txt \
    diagrams/tokenizer/engine0-token-ownership.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 2 artifact: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-01/chapter-01-the-missing-half-of-ai.md \
    code/mini-engine/Cargo.toml \
    code/mini-engine/crates/engine0/Cargo.toml \
    code/reference/engine-0-oracle.md \
    labs/lab-01-generate-one-token-manually.md \
    diagrams/runtime/request-to-token.txt \
    diagrams/runtime/model-vs-engine.txt \
    diagrams/runtime/inference-stack.txt \
    diagrams/runtime/token-byte-owner.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 1 artifact: $file" >&2
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

chapter_one_words=$(wc -w < manuscript/part-01/chapter-01-the-missing-half-of-ai.md)
if [ "$chapter_one_words" -lt 5000 ] || [ "$chapter_one_words" -gt 9000 ]; then
    echo "Chapter 1 must contain 5,000-9,000 words, found $chapter_one_words" >&2
    exit 1
fi

chapter_two_words=$(wc -w < manuscript/part-01/chapter-02-from-text-to-tokens.md)
if [ "$chapter_two_words" -lt 6000 ] || [ "$chapter_two_words" -gt 10000 ]; then
    echo "Chapter 2 must contain 6,000-10,000 words, found $chapter_two_words" >&2
    exit 1
fi

echo "structure check passed: 15 parts, 94 specifications, Chapters 1-2 artifacts present"
