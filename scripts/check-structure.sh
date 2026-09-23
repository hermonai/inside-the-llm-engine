#!/bin/sh
# Repository structure, governance budget, plan consistency, preservation of
# first-edition artifacts, and a guard against committed secrets.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

fail() {
    echo "$1" >&2
    exit 1
}

# --- Second-edition governance ------------------------------------------------
for file in README.md BOOK.md AGENTS.md AUTHORING.md CONTRIBUTING.md \
    docs/STRUCTURE.md docs/STATUS.md archive/README.md research/FRONTIER.md \
    tex/inside-the-llm-engine.tex tex/preamble.tex; do
    [ -s "$file" ] || fail "missing or empty required file: $file"
done

# One governance document, capped at 3,000 words (see AUTHORING.md).
words=$(wc -w < AUTHORING.md | tr -d " ")
[ "$words" -le 3000 ] || fail "AUTHORING.md is $words words; the cap is 3,000"

# docs/ holds only the plan and the status; policy lives in AUTHORING.md.
extra=$(ls docs | grep -v -x -e STRUCTURE.md -e STATUS.md || true)
[ -z "$extra" ] || fail "unexpected files in docs/ (policy belongs in AUTHORING.md): $extra"

# --- The plan: 42 chapters, numbered in sequence, same titles everywhere ------
plan=$(mktemp)
book=$(mktemp)
trap 'rm -f "$plan" "$book"' EXIT
sed -n -E 's/^\| ([0-9]+) \| ([^|]+) \|.*/\1. \2/p' docs/STRUCTURE.md \
    | sed -E 's/ +$//' > "$plan"
sed -n -E 's/^([0-9]+)\. (.+)$/\1. \2/p' BOOK.md | sed -E 's/ +$//' > "$book"

count=$(wc -l < "$plan" | tr -d ' ')
[ "$count" -ge 1 ] || fail "no chapter rows found in docs/STRUCTURE.md"
[ "$count" -le 45 ] || fail "the plan has $count chapters; merge instead of growing past 45"

expected=1
while IFS= read -r line; do
    number=${line%%.*}
    [ "$number" -eq "$expected" ] || fail "chapter sequence error in STRUCTURE.md: expected $expected, found $number"
    expected=$((expected + 1))
done < "$plan"

if ! diff -u "$plan" "$book" > /dev/null; then
    diff -u "$plan" "$book" >&2 || true
    fail "chapter titles differ between docs/STRUCTURE.md and BOOK.md"
fi

# --- First-edition artifacts must survive (nothing is thrown away) -----------
for file in \
    code/mini-engine/Cargo.toml \
    code/mini-engine/crates/engine0/Cargo.toml \
    code/mini-engine/crates/engine0/src/lib.rs \
    code/mini-engine/crates/engine0/src/tokenizer.rs \
    code/mini-engine/crates/engine0/src/model.rs \
    code/mini-engine/crates/engine0/src/sampling.rs \
    code/mini-engine/crates/engine0/src/tensor.rs \
    code/mini-engine/crates/engine0/src/linear.rs \
    code/mini-engine/crates/engine0/src/embedding.rs \
    code/mini-engine/crates/engine0/src/normalization.rs \
    code/mini-engine/crates/engine0/src/qkv.rs \
    code/mini-engine/crates/engine0/src/rope.rs \
    code/mini-engine/crates/engine0/src/utf8.rs \
    code/reference/engine-0-oracle.md \
    code/reference/chapter-02-tokenizer-oracles.md \
    code/reference/python/chapter03_oracle.py \
    code/reference/python/chapter04_sampling_oracle.py \
    code/reference/python/chapter05_tensor_oracle.py \
    code/reference/python/chapter06_matmul_oracle.py \
    code/reference/python/chapter07_embedding_rmsnorm_oracle.py \
    code/reference/python/chapter08_qkv_oracle.py \
    code/reference/python/chapter09_rope_oracle.py \
    code/reference/fixtures/chapter08-qkv.json \
    code/reference/fixtures/chapter09-rope.json \
    code/experiments/tokenizer-comparison/compare.py \
    code/experiments/chapter-03-projection-scaling.py \
    research/README.md \
    research/hermon/README.md \
    research/benchmarks/chapter-03-projection-scaling.md \
    research/benchmarks/chapter-04-sampling-cost.md \
    research/benchmarks/chapter-05-traversal-order.md \
    research/benchmarks/chapter-06-loop-order.md \
    research/benchmarks/chapter-06-blocked-matmul.md \
    research/benchmarks/chapter-06-gemv-vs-gemm.md \
    labs/README.md \
    labs/lab-01-generate-one-token-manually.md \
    labs/lab-39-qkv-projection-workbench.md \
    labs/lab-49-rope-position-workbench.md \
    diagrams/README.md \
    diagrams/INDEX.md \
    figures/manifest.json \
    publication/build-tex.py
do
    [ -s "$file" ] || fail "missing or empty first-edition artifact: $file"
done

lab_count=$(ls labs | grep -c -E '^lab-[0-9]+-.*\.md$')
[ "$lab_count" -ge 40 ] || fail "expected the first edition's lab files, found $lab_count"

part=1
while [ "$part" -le 15 ]; do
    dir=$(printf 'manuscript/part-%02d' "$part")
    [ -s "$dir/README.md" ] || fail "missing frozen Markdown part index: $dir/README.md"
    part=$((part + 1))
done

if grep -R -n '```mermaid' README.md BOOK.md AUTHORING.md docs research; then
    fail "Mermaid found; figures are native TikZ"
fi

# --- Secrets guard: the repository has a public remote ------------------------
# Generic patterns only; naming real hosts here would itself leak them. The
# bracketed letters keep each pattern from matching its own source line.
ipv4='(^|[^0-9.])((25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])([^0-9.]|$)'
secrets='duckdn[s]\.org|/\.ss[h]/|id_(ed2551[9]|rs[a])\b|BEGIN [A-Z ]*PRIVAT[E] KEY|gh[p]_[A-Za-z0-9]{20}|github_pa[t]_|sk-an[t]-|AKI[A][0-9A-Z]{16}'
hits=$(git ls-files -z | xargs -0 grep -I -n -E "$ipv4|$secrets" 2>/dev/null || true)
[ -z "$hits" ] || fail "possible host, address or credential in tracked files:
$hits"

echo "structure check passed: $count chapters planned, AUTHORING.md $words words, first-edition artifacts present, no secrets"
