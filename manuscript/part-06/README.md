# Part VI — Paged Inference Memory

**Goal:** derive explicit logical-to-physical KV ownership and prefix sharing.
**Chapters:** 28–34, slot failure, paging, block pool, radix index, COW, eviction,
admission, and paged attention.
**Prerequisites:** KV geometry, request lifecycle, and continuous batching.
**Code milestone:** ENGINE-7 paged-KV runtime.
**Conceptual milestone:** prove when a physical block is mutable or reusable.
**Later parts:** supplies the memory contract consumed by native kernels.
