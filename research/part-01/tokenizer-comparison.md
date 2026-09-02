# Chapter 2 Experiment — Two Real Tokenizers

Run date: 2026-09-02. This is a token-count and segmentation observation, not a
speed benchmark or a quality ranking.

## Compared identities

1. OpenAI `tiktoken` 0.14.0, named encoding `cl100k_base`; project head observed
   at `4e71bbe0c078468e00fefbf94b39849389f346e5`.
2. Google SentencePiece 0.2.2 with the official repository fixture
   `data/wagahaiwa_nekodearu_ja_bpe_byte_2000.model` at commit
   `ac0f71dcbc85f94292266e55bf6cee1b4d6c9dc1`; model SHA-256
   `6f00a9995a025eab01394c94e6e6b73904ca172762dfc3ecd2fd7ce094587a25`.

The SentencePiece fixture is a 2,000-piece BPE test model trained on a small
Japanese corpus with byte fallback. It is useful primary-backed evidence for
algorithm behavior, but it is not representative of a production LLM's
vocabulary size or training corpus. `cl100k_base` and this fixture do not belong
to the same model. Their IDs are never interchangeable.

## Reproduction

The run used Python 3.9 in a temporary virtual environment; neither package nor
model was added to the book's Rust dependency graph. The complete probe is
[`compare.py`](../../code/experiments/tokenizer-comparison/compare.py).
Commands, abbreviated only by `$TMP`, were:

```sh
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install tiktoken==0.14.0 sentencepiece==0.2.2
git clone --filter=blob:none --no-checkout \
  https://github.com/google/sentencepiece.git "$TMP/sentencepiece"
git -C "$TMP/sentencepiece" checkout \
  ac0f71dcbc85f94292266e55bf6cee1b4d6c9dc1
```

For each valid UTF-8 input, byte count is `len(text.encode("utf-8"))` and scalar
count is Python's code-point length for these inputs (none contains a surrogate
code point). `tiktoken.decode_single_token_bytes` supplies piece bytes;
SentencePiece supplies IDs, piece labels, and decoded text.

## Summary counts

| Input | UTF-8 bytes | Unicode scalars | `cl100k_base` tokens | SentencePiece tokens |
| --- | ---: | ---: | ---: | ---: |
| `Tokenizers count spaces.` | 24 | 24 | 5 | 25 |
| two-line C-like loop | 39 | 39 | 21 | 38 |
| `模型把文本变成编号。` | 30 | 10 | 10 | 19 |
| `👩🏽‍💻🚀` | 19 | 5 | 13 | 20 |
| leading/tab/double-space/newline/trailing fixture | 34 | 34 | 9 | 30 |

These counts reflect different vocabularies, corpora, pre-processing, and
piece rules. The smaller number in a row is not “better” without a model and
workload objective.

## Piece observations

### English

```text
input:     Tokenizers count spaces.
cl100k:    "Token" | "izers" | " count" | " spaces" | "."
SP model:  ▁ | <0x54> | o | <0x6B> | e | n | i | <0x7A> | e | r | s |
           ▁ | c | o | u | n | t | ▁ | s | <0x70> | a | c | e | s | .
```

The small Japanese fixture falls back to individual bytes for several Latin
letters. Its decode still reconstructs the input exactly in this case.

### Code and line endings

```text
input:  for (i = 0; i < 3; i++) {\n  sum += i;\n}
counts: cl100k=21, SentencePiece=38
```

`cl100k_base` has pieces including `"for"`, `" ("`, `"++)"`, `" {\n"`, and
`";\n"`. The SentencePiece fixture uses many one-character or byte-fallback
pieces. More importantly, its configured normalizer decodes the two-line input
as one space-normalized line:

```text
for (i = 0; i < 3; i++) { sum += i; }
```

So `decode(encode(text)) == text` is false for this configured tokenizer even
though its vocabulary has byte fallback. Normalization changed the surface
before segmentation.

### Chinese

```text
input:   模型把文本变成编号。
cl100k:  模 | 型 | [E6 8A] | [8A] | 文 | 本 | 变 | 成 | 编号 | 。
SP:      ▁ | 模 | <E5> | <9E> | <8B> | <E6> | <8A> | <8A> | 文 | 本 |
         <E5> | <8F> | <98> | 成 | <E7> | <BC> | <96> | 号 | 。
```

The `把` bytes cross two `cl100k_base` token boundaries. Neither piece is valid
UTF-8 alone. The SentencePiece fixture also uses byte fallback for several
scalars. Both reconstruct the input after concatenating pieces and decoding the
complete byte stream.

### Emoji

```text
input:  👩🏽‍💻🚀
bytes:  19
scalars: 5
tokens: cl100k=13, SentencePiece=20 (including SentencePiece's dummy ▁)
```

Most `cl100k_base` pieces are incomplete UTF-8 byte fragments, such as `F0 9F`,
`91`, and `A9` for the first scalar. The SentencePiece fixture uses one byte
fallback token per UTF-8 byte. This directly demonstrates why a streamer must
buffer bytes across token boundaries.

### Whitespace-sensitive input

```text
input JSON: "  leading\tand  internal\ntrailing  "
cl100k decoded surface: exact input
SentencePiece decoded:  "leading and internal trailing"
```

The SentencePiece model's configured normalization removes leading/trailing
space, collapses repeated whitespace, and represents boundaries with `▁`.
Again, BPE and byte fallback alone do not determine round-trip behavior.

## Systems conclusions

- Token count is a function of exact tokenizer identity and configured input
  transforms, not merely input characters or language.
- Tokens can contain spaces, omit spaces, span many bytes, or represent byte
  fragments that are not independently valid UTF-8.
- Byte fallback guarantees vocabulary coverage after the tokenizer's configured
  preprocessing; it does not reverse an earlier lossy normalization.
- Context limits, prefill positions, KV growth, per-token accounting, and stop
  budgets must use the IDs produced by the tokenizer bound to the model.
- This experiment provides no throughput, latency, memory-allocation, or model
  quality evidence.
