# Part IV — Why Naive Inference Is Slow

**Goal:** use measurement to derive prefill/decode differences and KV reuse.
**Chapters:** 19–22, profiling, workload phases, KV necessity, and memory math.
**Prerequisites:** a working real-model runner and basic measurement literacy.
**Code milestone:** ENGINE-4 cached decoder with uncached equivalence.
**Conceptual milestone:** separate compute, memory, and repeated work.
**Later parts:** motivates scheduling and paged allocation.
