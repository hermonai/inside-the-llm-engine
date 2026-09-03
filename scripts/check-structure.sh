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
docs/MATH_INDEX.md
docs/DIAGRAM_STYLE.md
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
research/part-01/chapter-03-the-smallest-possible-language-model.md
research/part-01/chapter-04-logits-sampling-autoregressive-loop.md
research/part-01/tokenizer-comparison.md
research/part-02/README.md
research/part-02/chapter-05-tensors-without-magic.md
research/part-02/chapter-06-matrix-multiplication-the-engine-room.md
diagrams/README.md
diagrams/INDEX.md'

printf '%s\n' "$required" | while IFS= read -r file; do
    if [ ! -s "$file" ]; then
        echo "missing or empty required file: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-02/chapter-06-matrix-multiplication-the-engine-room.md \
    code/mini-engine/crates/engine0/src/linear.rs \
    code/mini-engine/crates/engine0/tests/linear.rs \
    code/mini-engine/crates/engine0/examples/chapter06_bench.rs \
    code/reference/python/chapter06_matmul_oracle.py \
    research/benchmarks/chapter-06-loop-order.md \
    research/benchmarks/chapter-06-blocked-matmul.md \
    research/benchmarks/chapter-06-gemv-vs-gemm.md \
    labs/lab-22-dot-product.md \
    labs/lab-23-gemv-by-hand.md \
    labs/lab-24-gemm-by-hand.md \
    labs/lab-25-loop-order.md \
    labs/lab-26-blocked-gemm.md \
    labs/lab-27-break-the-kernel.md \
    labs/lab-28-kernel-equivalence.md \
    labs/lab-29-gemv-vs-gemm.md \
    diagrams/linear/dot-product-multiply-accumulate.txt \
    diagrams/linear/gemv-shape-contract.txt \
    diagrams/linear/gemm-shape-contract.txt \
    diagrams/linear/gemm-one-output-cell.txt \
    diagrams/linear/weight-orientation.txt \
    diagrams/linear/row-major-access.txt \
    diagrams/linear/loop-order-ijk-vs-ikj.txt \
    diagrams/linear/cache-reuse-and-tiling.txt \
    diagrams/linear/reference-vs-blocked-kernel.txt \
    diagrams/linear/optimization-ladder.txt \
    diagrams/linear/gemv-vs-gemm-reuse.txt \
    diagrams/linear/roofline-concept.txt \
    diagrams/linear/engine-2-kernel-stack.txt \
    diagrams/linear/follow-the-flop.txt \
    diagrams/linear/follow-the-byte.txt \
    diagrams/linear/follow-the-reuse.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 6 artifact: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-02/chapter-05-tensors-without-magic.md \
    code/mini-engine/crates/engine0/src/tensor.rs \
    code/mini-engine/crates/engine0/tests/tensor.rs \
    code/mini-engine/crates/engine0/examples/chapter05_traversal.rs \
    code/reference/python/chapter05_tensor_oracle.py \
    research/benchmarks/chapter-05-traversal-order.md \
    labs/lab-16-offset-by-hand.md \
    labs/lab-17-transpose-without-copy.md \
    labs/lab-18-reshape-view.md \
    labs/lab-19-non-contiguous-copy.md \
    labs/lab-20-break-shape-arithmetic.md \
    labs/lab-21-mutation-and-aliasing.md \
    diagrams/tensor/logical-vs-physical.txt \
    diagrams/tensor/row-major-offsets.txt \
    diagrams/tensor/shape-and-strides.txt \
    diagrams/tensor/transpose-view.txt \
    diagrams/tensor/view-vs-copy.txt \
    diagrams/tensor/tensor-ownership.txt \
    diagrams/tensor/tensor-memory-lifetime.txt \
    diagrams/tensor/contiguous-vs-strided.txt \
    diagrams/tensor/follow-the-element.txt \
    diagrams/tensor/follow-the-byte.txt \
    diagrams/tensor/follow-the-owner.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 5 artifact: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-01/chapter-04-logits-sampling-autoregressive-loop.md \
    code/mini-engine/crates/engine0/src/sampling.rs \
    code/mini-engine/crates/engine0/tests/sampling.rs \
    code/reference/python/chapter04_sampling_oracle.py \
    research/benchmarks/chapter-04-sampling-cost.md \
    labs/lab-09-stable-softmax-by-hand.md \
    labs/lab-10-temperature.md \
    labs/lab-11-fixed-categorical-draw.md \
    labs/lab-12-top-k-vs-top-p.md \
    labs/lab-13-build-the-autoregressive-loop.md \
    labs/lab-14-change-the-seed.md \
    labs/lab-15-break-the-sampler.md \
    diagrams/sampling/sampling-pipeline.txt \
    diagrams/sampling/autoregressive-state.txt \
    diagrams/sampling/categorical-intervals.txt \
    diagrams/sampling/part1-follow-token.txt \
    diagrams/sampling/part1-follow-byte.txt \
    diagrams/sampling/part1-follow-owner.txt \
    diagrams/sampling/model-two-requests.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 4 artifact: $file" >&2
        exit 1
    fi
done

for file in \
    manuscript/part-01/chapter-03-the-smallest-possible-language-model.md \
    code/mini-engine/crates/engine0/src/model.rs \
    code/reference/python/chapter03_oracle.py \
    code/experiments/chapter-03-projection-scaling.py \
    research/benchmarks/chapter-03-projection-scaling.md \
    labs/lab-05-forward-pass-by-hand.md \
    labs/lab-06-change-one-weight.md \
    labs/lab-07-same-last-token-same-output.md \
    labs/lab-08-break-the-shape.md \
    diagrams/model/token-id-to-logits.txt \
    diagrams/model/embedding-row-lookup.txt \
    diagrams/model/one-logit-dot-product.txt \
    diagrams/model/tiny-model-tensor-shapes.txt \
    diagrams/model/parameters-vs-activations.txt \
    diagrams/model/semantics-vs-execution.txt \
    diagrams/model/context-limitation.txt
do
    if [ ! -s "$file" ]; then
        echo "missing or empty Chapter 3 artifact: $file" >&2
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

chapter_three_words=$(wc -w < manuscript/part-01/chapter-03-the-smallest-possible-language-model.md)
if [ "$chapter_three_words" -lt 6000 ] || [ "$chapter_three_words" -gt 10000 ]; then
    echo "Chapter 3 must contain 6,000-10,000 words, found $chapter_three_words" >&2
    exit 1
fi

chapter_four_words=$(wc -w < manuscript/part-01/chapter-04-logits-sampling-autoregressive-loop.md)
if [ "$chapter_four_words" -lt 6000 ] || [ "$chapter_four_words" -gt 10000 ]; then
    echo "Chapter 4 must contain 6,000-10,000 words, found $chapter_four_words" >&2
    exit 1
fi

chapter_five_words=$(wc -w < manuscript/part-02/chapter-05-tensors-without-magic.md)
if [ "$chapter_five_words" -lt 6000 ] || [ "$chapter_five_words" -gt 10000 ]; then
    echo "Chapter 5 must contain 6,000-10,000 words, found $chapter_five_words" >&2
    exit 1
fi

chapter_six_words=$(wc -w < manuscript/part-02/chapter-06-matrix-multiplication-the-engine-room.md)
if [ "$chapter_six_words" -lt 7000 ] || [ "$chapter_six_words" -gt 11000 ]; then
    echo "Chapter 6 must contain 7,000-11,000 words, found $chapter_six_words" >&2
    exit 1
fi

echo "structure check passed: 15 parts, 94 specifications, Chapters 1-6 artifacts present"
