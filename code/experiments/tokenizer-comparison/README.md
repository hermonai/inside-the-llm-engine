# Chapter 2 real-tokenizer comparison

This probe records segmentation and counts; it is not a performance benchmark.
It requires Python 3, `tiktoken==0.14.0`, `sentencepiece==0.2.2`, and the pinned
official SentencePiece fixture described below. Install them only in a scratch
virtual environment.

```sh
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install tiktoken==0.14.0 sentencepiece==0.2.2
git clone --filter=blob:none --no-checkout \
  https://github.com/google/sentencepiece.git "$TMP/sentencepiece"
git -C "$TMP/sentencepiece" sparse-checkout init --cone
git -C "$TMP/sentencepiece" sparse-checkout set data
git -C "$TMP/sentencepiece" checkout \
  ac0f71dcbc85f94292266e55bf6cee1b4d6c9dc1
"$TMP/venv/bin/python" compare.py \
  "$TMP/sentencepiece/data/wagahaiwa_nekodearu_ja_bpe_byte_2000.model"
```

The expected model SHA-256 is
`6f00a9995a025eab01394c94e6e6b73904ca172762dfc3ecd2fd7ce094587a25`.
The interpreted record is
[`research/part-01/tokenizer-comparison.md`](../../../research/part-01/tokenizer-comparison.md).

