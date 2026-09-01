# Part X — Mixture-of-Experts Inference

**Goal:** manage models whose inactive weights exceed fast memory.
**Chapters:** 54–58, routing economics, VRAM limits, expert storage/paging,
residency/pinning/I/O, and unified inference memory.
**Prerequisites:** packed weights, paging, providers, and storage fundamentals.
**Code milestone:** expert pager and host compute path with correctness oracle.
**Conceptual milestone:** follow an expert byte through tiers and leases.
**Later parts:** supplies difficult lifetime cases and future memory design.
