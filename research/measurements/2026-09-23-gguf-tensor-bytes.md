# Tensor bytes of two GGUF files, and the bytes a token touches

**Date:** 2026-09-23.
**Used by:** Chapters 2, 3 and 5.
**Raw outputs and tool:** [`2026-09-23-gguf-tensor-bytes/`](2026-09-23-gguf-tensor-bytes/).

## Question

How many bytes does a model file hold, and how many of them does one decode
step at batch one actually read?

## Method

`gguf_info.py`, a standard-library GGUF v2/v3 reader, parses each file's
header: metadata, then every tensor's name, shape and type. Tensor bytes are
computed from the type's block size and bytes per block (the table in the
script, matching llama.cpp's K-quant definitions: Q4_K 144 bytes per 256
values, Q6_K 210 bytes per 256). No weights are loaded. This is a read of the
files, not a timing; it is exact for the files named.

| File | Model | Digest |
| --- | --- | --- |
| `llama3.2-gguf.json` | Llama 3.2 3B Instruct, Q4_K_M | `sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff` |
| `deepseek-r1-gguf.json` | DeepSeek-R1-0528-Qwen3-8B (Qwen3 8B architecture), Q4_K_M | `sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d` |

## Results

| | Llama 3.2 3B | Qwen3-8B distill |
| --- | ---: | ---: |
| Parameters | 3,212,749,888 | 8,190,735,360 |
| Tensor bytes | 2,011,539,712 | 5,219,418,112 |
| File bytes | 2,019,377,376 | 5,225,373,760 |
| Input embedding table | 323,205,120 (Q6_K), **tied**: also the output projection | 350,060,544 (Q4_K) |
| Output projection | — (uses the input table) | 510,504,960 (Q6_K) |
| Tensor types | Q4_K 1,362,493,440; Q6_K 648,345,600; F32 700,672 | Q4_K 3,662,512,128; Q6_K 1,253,683,200; F16 301,989,888; F32 1,232,896 |

**Parameter counts match the published shapes exactly** once norms are
counted: for Qwen3-8B, 36 layers × (41,943,040 attention + 150,994,944 FFN +
8,448 norm) + 2 × 622,329,856 (embedding, output) + 4,096 (final norm) =
8,190,735,360. For Llama 3.2 3B, the formula gives 3,212,749,824; the
remaining 64 values are a RoPE frequency-factor tensor.

## Derived: bytes read per decode token at batch one

- **Llama 3.2 3B:** all 2,011,539,712 bytes. The embedding lookup reads one
  row, but the tied output projection reads the whole table.
- **Qwen3-8B distill:** 5,219,418,112 − 350,060,544 + one embedding row
  (4,096 values × 4.5 bits = 2,304 bytes) ≈ **4.87 GB, 93% of the tensors**.
  The untied input table is gathered one row per token; everything else is
  streamed.

KV-cache reads are additional and grow with context (Chapter 5).

## Limits

Bytes *read* assumes a kernel reads each weight once per step. A kernel that
re-reads tiles, or the small-batch staircase of the companion record
(`2026-09-23-m1-llama32-3b.md`), moves more.
