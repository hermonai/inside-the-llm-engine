# Part IX — Modern Decode Optimization

**Goal:** reuse prefixes and safely reduce target-model decode steps.
**Chapters:** 49–53, general prefix caching, sticky slots, speculative decoding,
prompt lookup, and losing workloads.
**Prerequisites:** batching, KV ownership, sampling, and provider cost models.
**Code milestone:** cache/speculation extensions with rollback and adaptive gates.
**Conceptual milestone:** derive acceptance and overhead break-even.
**Later parts:** reinforces evidence-driven optimization and state rollback.
