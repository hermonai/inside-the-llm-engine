# Part I — What Actually Happens When an LLM Answers?

**Goal:** replace the API black box with a complete first-token mental model.
**Chapters:** 1–4, from the inference stack through tokenization, a tiny neural
model, logits, sampling, and the autoregressive loop.
**Prerequisites:** programming and basic algebra; no ML systems background.
**Code milestone:** ENGINE-0 token generator and ENGINE-1 tiny neural model.
**Conceptual milestone:** follow one token from input bytes to streamed output.
**Later parts:** supplies vocabulary, logits, and loop semantics used everywhere.

| Chapter | Status | Artifact |
| --- | --- | --- |
| 1. The Missing Half of AI | COMPLETE | [Chapter](chapter-01-the-missing-half-of-ai.md), [ENGINE-0](../../code/mini-engine/README.md), [Lab 1](../../labs/lab-01-generate-one-token-manually.md) |
| 2. From Text to Tokens | PLANNED | Tokenizer contract and byte/token oracles |
| 3. The Smallest Possible Language Model | PLANNED | ENGINE-1 numerical model |
| 4. Logits, Sampling, and the Autoregressive Loop | PLANNED | Complete generation loop |
