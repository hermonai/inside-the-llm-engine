#!/usr/bin/env python3
"""Print the pinned Chapter 2 tokenizer comparison as JSON."""

import hashlib
import json
from pathlib import Path
import sys

import sentencepiece as spm
import tiktoken


EXPECTED_MODEL_SHA256 = (
    "6f00a9995a025eab01394c94e6e6b73904ca172762dfc3ecd2fd7ce094587a25"
)

CASES = {
    "english": "Tokenizers count spaces.",
    "code": "for (i = 0; i < 3; i++) {\n  sum += i;\n}",
    "chinese": "模型把文本变成编号。",
    "emoji": "👩🏽‍💻🚀",
    "whitespace": "  leading\tand  internal\ntrailing  ",
}


def render_piece(piece: bytes) -> dict:
    try:
        return {"utf8": piece.decode("utf-8"), "hex": piece.hex()}
    except UnicodeDecodeError:
        return {"utf8": None, "hex": piece.hex()}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: compare.py SENTENCEPIECE_MODEL", file=sys.stderr)
        return 2

    model_path = Path(sys.argv[1])
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        print(
            f"unexpected SentencePiece model SHA-256: {digest}",
            file=sys.stderr,
        )
        return 2

    sentencepiece = spm.SentencePieceProcessor(model_file=str(model_path))
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    rows = []
    for name, text in CASES.items():
        tiktoken_ids = tiktoken_encoding.encode(text)
        sentencepiece_ids = sentencepiece.encode(text, out_type=int)
        rows.append(
            {
                "case": name,
                "text": text,
                "utf8_bytes": len(text.encode("utf-8")),
                "unicode_scalars": len(text),
                "cl100k_base": {
                    "ids": tiktoken_ids,
                    "pieces": [
                        render_piece(tiktoken_encoding.decode_single_token_bytes(token_id))
                        for token_id in tiktoken_ids
                    ],
                },
                "sentencepiece": {
                    "ids": sentencepiece_ids,
                    "pieces": sentencepiece.encode(text, out_type=str),
                    "decoded": sentencepiece.decode(sentencepiece_ids),
                },
            }
        )

    print(
        json.dumps(
            {
                "tiktoken_version": tiktoken.__version__,
                "tiktoken_encoding": "cl100k_base",
                "sentencepiece_version": spm.__version__,
                "sentencepiece_model_sha256": digest,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
