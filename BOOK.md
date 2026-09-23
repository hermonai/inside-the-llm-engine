# Inside the LLM Engine

## The Systems Engineering of Large Language Model Inference

Every inference engine is a negotiation between four scarce resources — memory
bandwidth, memory capacity, compute and latency — for two outcomes: a correct
answer and a low cost per token. This book introduces every technique as a
response to a measured constraint and charges it for what it spends.

This is the reader-facing table of contents of the second edition. Chapter
intents are in [docs/STRUCTURE.md](docs/STRUCTURE.md); progress is in
[docs/STATUS.md](docs/STATUS.md).

### Part I — The Physics of Inference

1. Why Inference Is Its Own Discipline
2. The Forward Pass, Compressed
3. The Roofline: Why Decode Leaves Your GPU Idle
4. Prefill and Decode: Two Workloads, One Model
5. The KV Cache: Inference's Central Data Structure
6. Napkin Math: Latency, Throughput and Cost per Token

### Part II — Making One Request Fast

7. Matrix Multiplication on Real Hardware
8. Attention as a Memory Problem: From Online Softmax to FlashAttention
9. Decode Attention: FlashDecoding, Split-K and Paged Kernels
10. Quantizing Weights: INT8, INT4, GPTQ, AWQ and K-Quants
11. Below 16 Bits in Floating Point: FP8, MXFP4 and NVFP4
12. Quantizing the KV Cache
13. How Kernels Are Written Now: CUDA, Triton, Tile DSLs and Graph Capture

### Part III — Serving Many Requests

14. The Request Lifecycle: Streaming, Cancellation and Terminal Ownership
15. Continuous Batching
16. Paged KV Memory
17. Prefix Caching and RadixAttention
18. Chunked Prefill and Phase Interference
19. Scheduling Under Pressure: Preemption, Fairness and SLO-Aware Admission
20. Structured Output at Engine Speed
21. One Server, Many Models: LoRA, Adapters and Routing

### Part IV — Beyond One Token per Step

22. Speculative Decoding: The Draft–Verify Contract
23. Modern Speculation: The EAGLE Family, Medusa and Multi-Token Prediction
24. When Speculation Loses

### Part V — Architectures That Reshape the Engine

25. Shrinking the KV Cache: MQA, GQA and Multi-Head Latent Attention
26. Mixture of Experts
27. Hybrid Models: State Spaces and Linear Attention
28. Long Context: Windows, Sinks and Sparse Attention
29. Reasoning Models and Test-Time Compute
30. Multimodal Inference

### Part VI — Scaling Out

31. Tensor, Pipeline and Context Parallelism
32. Expert Parallelism at Scale
33. Disaggregated Serving
34. The KV Cache as a Distributed Storage Tier

### Part VII — Inference on the Hardware You Have

35. Offloading Across VRAM, RAM and NVMe
36. Streaming Experts From Disk: A Measured Case Study
37. Unified Inference Memory

### Part VIII — Correctness and Measurement

38. Fast Wrong Answers: Oracles, Differential Testing and Determinism
39. Benchmarking Without Lying to Yourself

### Part IX — Production and Capstone

40. Operating Inference in Production
41. Anatomy of Real Engines: vLLM, SGLang, llama.cpp and Hermon
42. Capstone: Build a Mini Engine End to End

### Appendices

- A. The Transformer From Scratch
- B. GPU Architecture for Inference Engineers
- C. CPU and SIMD
- D. Model File Formats: GGUF and safetensors
- E. Numerics: Floating Point and Quantization Formats
- F. Notation
- G. Reproducing the Book's Measurements
