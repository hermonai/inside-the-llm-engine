# Terminology System

`GLOSSARY.md` is the reader-facing definition catalog. This file governs how
authors introduce and maintain terms.

## Entry contract

Every important term records: term, short definition, precise definition,
first-introduced chapter, related terms, and common confusion. Add the glossary
entry before a chapter reaches TECH-REVIEW.

## Canonical distinctions

- A **token** is a model vocabulary identifier; it is not necessarily a word,
  character, or byte.
- **Prefill** evaluates prompt positions and creates reusable per-position
  state; **decode** advances active sequences with newly generated positions.
- A **sequence** is one logical token history. A **physical token batch** is the
  work assembled for one forward execution and may mix phases/sequences.
- The **KV cache** stores per-layer key/value vectors for prior positions. It
  does not cache logits or eliminate reading visible history during attention.
- A **logical block** is a sequence-relative range; a **physical block** is an
  allocator-owned storage object; a **block table** maps between them.
- A **prefix cache** indexes reusable computed state. A **radix tree** is one
  possible index; **sticky slots** reuse state bound to a context and are not a
  general page-sharing prefix cache.
- **MHA**, **GQA**, and **MQA** describe query-to-KV-head geometry. Do not use
  them interchangeably.
- A **provider** implements an execution capability for hardware; a **kernel**
  is a bounded computation. Neither term implies GPU execution.
- An **oracle** is an independent correctness reference; a **differential
  test** compares implementations under controlled inputs and tolerances.
- **Residency** says where bytes currently live; **ownership** says who controls
  lifetime/mutation; **pinning** temporarily prevents eviction.

## Naming rules

Use `ENGINE-N` for curriculum milestones, `part-NN` and
`chapter-NN-topic.md` for manuscript paths, and uppercase project status labels.
Use “Hermon-owned paged engine” only for the specific gated path verified in
Hermon; do not shorten a PREVIEW into an apparently CURRENT “Hermon engine.”

## Review

Terminology review checks first introduction, pluralization, hyphenation,
acronym expansion, diagram labels, code identifiers, and glossary cross-links.
When an external project uses a conflicting term, retain its name while stating
the book's canonical equivalent.
