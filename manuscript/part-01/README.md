# Part I — What Actually Happens When an LLM Answers?

**Goal:** replace the API black box with a complete first-token mental model.
**Chapters:** 1–4, from the inference stack through tokenization, a tiny neural
model, logits, sampling, and the autoregressive loop.
**Prerequisites:** programming and basic algebra; no ML systems background.
**Code milestone:** ENGINE-0 token generator and ENGINE-1 tiny neural model.
**Conceptual milestone:** follow one token from input bytes to streamed output.
**Later parts:** supplies vocabulary, logits, and loop semantics used everywhere.
