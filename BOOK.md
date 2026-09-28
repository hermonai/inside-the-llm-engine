# Inside the LLM Engine

## The Systems Engineering of Large Language Model Inference

Build a small model, understand its computation, then turn it into a fast and
reliable inference service. Part I is the main learning path, not optional
background. Later parts preserve the measured systems and frontier material.

This is the reader-facing outline. Source IDs, migration details and chapter
intents are in [docs/STRUCTURE.md](docs/STRUCTURE.md); actual completion is in
[docs/STATUS.md](docs/STATUS.md).

### Part I — Build and Understand a Language Model

1. From Text to Tokens
2. The Smallest Possible Language Model
3. Logits, Sampling, and the Autoregressive Loop
4. Tensors Without Magic
5. Matrix Multiplication: The Engine Room
6. Embeddings and Normalization
7. Queries, Keys, and Values
8. Position: RoPE From First Principles
9. Causal Attention, One Row at a Time
10. The Feed-Forward Network and the Residual Stream
11. Assembling a Decoder That Generates Tokens
12. Model File Formats: GGUF and safetensors
13. Numerics: Floating Point and Quantization Formats
14. CPU and SIMD
15. GPU Architecture for Inference Engineers
16. Reading Shapes, Bytes and Performance Bounds
17. Reproducing the Book's Measurements

### Part II — The Physics of Inference

18. Why Inference Is Its Own Discipline
19. The Forward Pass, Compressed
20. The Roofline: Why Decode Leaves Your GPU Idle
21. Prefill and Decode: Two Workloads, One Model
22. The KV Cache: Inference's Central Data Structure
23. Napkin Math: Latency, Throughput and Cost per Token

### Part III — Making One Request Fast

24. Matrix Multiplication on Real Hardware
25. Attention as a Memory Problem: From Online Softmax to FlashAttention
26. Decode Attention: FlashDecoding, Split-K and Paged Kernels
27. Quantizing Weights: INT8, INT4, GPTQ, AWQ and K-Quants
28. Below 16 Bits in Floating Point: FP8, MXFP4 and NVFP4
29. Quantizing the KV Cache
30. How Kernels Are Written Now: CUDA, Triton, Tile DSLs and Graph Capture

### Part IV — Serving Many Requests

31. The Request Lifecycle: Streaming, Cancellation and Terminal Ownership
32. Continuous Batching
33. Paged KV Memory
34. Prefix Caching and RadixAttention
35. Chunked Prefill and Phase Interference
36. Scheduling Under Pressure: Preemption, Fairness and SLO-Aware Admission
37. Structured Output at Engine Speed
38. One Server, Many Models: LoRA, Adapters and Routing

### Part V — Beyond One Token per Step

39. Speculative Decoding: The Draft–Verify Contract
40. Modern Speculation: The EAGLE Family, Medusa and Multi-Token Prediction
41. When Speculation Loses

### Part VI — Architectures That Reshape the Engine

42. Shrinking the KV Cache: MQA, GQA and Multi-Head Latent Attention
43. Mixture of Experts
44. Hybrid Models: State Spaces and Linear Attention
45. Long Context: Windows, Sinks and Sparse Attention
46. Reasoning Models and Test-Time Compute
47. Multimodal Inference

### Part VII — Scaling Out

48. Tensor, Pipeline and Context Parallelism
49. Expert Parallelism at Scale
50. Disaggregated Serving
51. The KV Cache as a Distributed Storage Tier

### Part VIII — Inference on the Hardware You Have

52. Offloading Across VRAM, RAM and NVMe
53. Streaming Experts From Disk: A Measured Case Study
54. Unified Inference Memory

### Part IX — Correctness and Measurement

55. Fast Wrong Answers: Oracles, Differential Testing and Determinism
56. Benchmarking Without Lying to Yourself

### Part X — Production and Capstone

57. Operating Inference in Production
58. Anatomy of Real Engines: vLLM, SGLang, llama.cpp and Hermon
59. Capstone: Build a Mini Engine End to End
