# Part V — From Model Runner to Inference Server

**Goal:** turn one synchronous generation into a bounded multi-user service.
**Chapters:** 23–27, request state, concurrency, continuous batching, fairness,
backpressure, cancellation, and streaming.
**Prerequisites:** ENGINE-4 plus async/concurrency fundamentals.
**Code milestone:** ENGINE-5 server and ENGINE-6 continuous-batched runtime.
**Conceptual milestone:** follow request ownership through every terminal path.
**Later parts:** creates the allocator and scheduler pressure paged KV must handle.
