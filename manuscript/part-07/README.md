# Part VII — Native Kernel Engineering

**Goal:** move hot mechanisms behind a narrow, deterministic native boundary.
**Chapters:** 35–41, boundary placement, ABI, arenas, refcounts, bulk writes,
online softmax, split-K, and planning.
**Prerequisites:** ENGINE-7, C/Rust FFI basics, numerical testing.
**Code milestone:** ENGINE-8 native kernel runtime.
**Conceptual milestone:** separate host policy from planned kernel mechanism.
**Later parts:** provides a stable surface for ISA and accelerator providers.
