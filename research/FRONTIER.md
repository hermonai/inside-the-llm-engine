# Frontier ledger

The dated, sourced record of the state of the art that the book's chapters
rely on. `AUTHORING.md` §6 makes it mandatory: no frontier claim is written
from memory. Search, read the primary source, and log the finding here before
the claim enters a chapter.

## Entry format

| Field | Meaning |
| --- | --- |
| Technique | Name as the primary source uses it |
| What it is | One line |
| Primary source | Paper, specification, official documentation, or source at a pinned commit — with a link |
| Checked | Date the source was read (YYYY-MM-DD) |
| Maturity | *research*, *shipped in* a named engine or product, or *industry standard* |
| Chapter | Where the book uses it |

Blog posts and vendor benchmark claims may appear as leads, never as the
primary source. Vendor peak numbers are specifications, not measurements. When
an entry is superseded, keep it and add the successor as a new row that says
what it supersedes. "Read" means the abstract or the named section was read
on the date given; a claim from deeper in a paper cites the section.

---

## Hardware (specifications, not measurements)

Tensor-core peaks on NVIDIA datasheets are quoted **with 2:4 structured
sparsity** unless marked dense. Dense throughput is half. The book always
quotes dense numbers and says so.

| Device | Memory | Bandwidth | Dense peak (per GPU) | Primary source | Checked | Chapter |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | BF16 989 TFLOP/s; FP8 1,979 TFLOP/s (datasheet: 1,979 / 3,958 "with sparsity") | [nvidia.com H100](https://www.nvidia.com/en-us/data-center/h100/) | 2026-09-23 | 3, 6, App B |
| NVIDIA H200 SXM | 141 GB HBM3e | 4.8 TB/s | Same compute as H100 SXM (BF16 1,979 sparse) | [nvidia.com H200](https://www.nvidia.com/en-us/data-center/h200/) | 2026-09-23 | 3, 6 |
| NVIDIA B200 (DGX/HGX B200) | 180 GB HBM3e (1,440 GB per 8) | 8 TB/s (64 TB/s per 8) | BF16 2.25 PFLOP/s; FP8 4.5; FP4 9 (HGX: FP8/BF16 listed sparse, FP4 listed sparse \| dense) | [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [HGX](https://www.nvidia.com/en-us/data-center/hgx/) | 2026-09-23 | 3, 6, 11 |
| NVIDIA B300 (HGX B300) | ~2.1 TB per 8 GPUs | not listed on HGX page | FP4 13.5 PFLOP/s dense per GPU (108 per 8); FP8 and BF16 as B200; INT8 and FP64 sharply reduced (3 POPS, 10 TFLOPS per 8) | [HGX](https://www.nvidia.com/en-us/data-center/hgx/) | 2026-09-23 | 11, App B |
| NVIDIA GB300 NVL72 (rack of 72) | 20 TB HBM | up to 576 TB/s (8 TB/s per GPU) | FP4 720 PFLOPS dense per rack; "all Tensor Core specifications are with sparsity unless otherwise noted"; claims 2× attention performance vs Blackwell | [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) | 2026-09-23 | 11, 31 |
| AMD Instinct MI355X | 288 GB HBM3E | 8 TB/s | Dense per GPU: FP16/BF16 2.52 PFLOP/s; INT8 5.03 POPS; MXFP8 and OCP-FP8 5.03; **MXFP6 and MXFP4 both 10.07** PFLOP/s (sparsity doubles FP16, BF16, INT8, OCP-FP8 only); FP64 vector 78.6 TFLOP/s; 256 CUs, 2.4 GHz; 1,400 W TBP | [AMD MI355X GPU datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf) (PDF dated 10/25; read with `pdftotext`, the web page times out) | 2026-09-24 | 3, 11, App B |
| NVIDIA GeForce RTX 5060 Ti | 8 or 16 GB GDDR7, 128-bit | 448 GB/s (derived: 28 Gbps × 128 bit ÷ 8) | 759 "AI TOPS" (precision and sparsity unstated) | [nvidia.com RTX 5060 family](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/), [ASUS spec: 28 Gbps](https://www.asus.com/motherboards-components/graphics-cards/dual/dual-rtx5060ti-16g/techspec/) | 2026-09-23 | 3 |
| Apple M1 | 8/16 GB LPDDR4X unified | ≈68 GB/s (derived: Apple states M1 Max's 400 GB/s is "nearly 6x that of M1"; 4,266 MT/s × 128 bit = 68.3 GB/s) | — | [Apple newsroom, Oct 2021](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/) | 2026-09-23 | 3 |
| Blackwell `tcgen05.mma` | Tensor Memory (TMEM) holds the accumulator; 2-SM MMA | — | "2x to 4x faster" than Hopper WGMMA; block-scaled MMA with a scale per 32 K-elements (UE8M0, MX types) or per 16 (UE4M3, NVFP4) | [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | 2026-09-23 | 7, 11, 13 |

GPU rental price used for illustrative cost arithmetic (volatile; re-check):

| Item | Value | Source | Checked | Chapter |
| --- | --- | --- | --- | --- |
| On-demand H100 SXM, 8× instance | $3.99 per GPU-hour | [Lambda pricing](https://lambda.ai/pricing) | 2026-09-23 | 6 |
| On-demand B200, 8× instance | $6.69 per GPU-hour | [Lambda pricing](https://lambda.ai/pricing) | 2026-09-23 | 6 |

## Model configurations used in napkin math

Read from the published `config.json` (or the GGUF file itself for local
measurements). Bytes per KV token assume BF16/F16 cache unless stated.

| Model | Shape (from config) | KV per token (derived) | Source | Checked | Chapter |
| --- | --- | --- | --- | --- | --- |
| Llama 3 8B / 70B / 405B | L 32/80/126; d 4,096/8,192/16,384; FFN 14,336/28,672/53,248; 32/64/128 heads; 8 KV heads; RoPE θ 500,000 | 128 / 320 / 504 KiB | *The Llama 3 Herd of Models*, Table 3, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) | 2026-09-23 | 2, 5, 6 |
| Llama 3.2 3B Instruct (Q4_K_M GGUF, local) | L 28, d 3,072, FFN 8,192, 24 heads, 8 KV heads, head 128, vocab 128,256, **tied** embeddings; 3.213 B params; tensors 2,011,539,712 bytes | 112 KiB | GGUF metadata of the measured file (Ollama `llama3.2:3b`) | 2026-09-23 | 3 |
| DeepSeek-R1-0528-Qwen3-8B (Q4_K_M GGUF, local) | Qwen3 8B shape: L 36, d 4,096, FFN 12,288, 32 heads, 8 KV heads, **untied** embeddings (token_embd 350 MB Q4_K, output 511 MB Q6_K); tensors 5,219,418,112 bytes | 144 KiB | GGUF metadata of the measured file (Ollama `deepseek-r1:8b`) | 2026-09-23 | 3 |
| Llama 3.2 1B | L 16, d 2,048, FFN 8,192, 32 heads, 8 KV heads, head 64, vocab 128,256, tied; 1.24 B params, 2.47 GB in BF16 (derived) | 32 KiB | [config.json via unsloth mirror of Meta's release](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json) (Meta's repository is gated) | 2026-09-24 | 13 |
| Qwen3-8B | L 36, d 4,096, FFN 12,288, 32 heads, 8 KV heads, head 128, vocab 151,936, untied | 144 KiB | [config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) | 2026-09-23 | 2, 5 |
| Qwen3-30B-A3B | L 48, d 2,048, 32 heads, 4 KV heads, head 128; 128 experts, 8 per token, expert FFN 768 | 96 KiB | [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | 2026-09-23 | 2, 26, 36 |
| DeepSeek-V3 | L 61 (first 3 dense), d 7,168, 128 heads; MLA `kv_lora_rank` 512 + `qk_rope_head_dim` 64; 256 routed + 1 shared expert, 8 per token; 1 MTP layer; FP8 E4M3 weights, 128×128 blocks; 671 B total, 37 B active | 68.6 KiB (576 elements × 61 layers × 2 bytes); MHA with the same heads would be ≈4.8 MiB | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | 2026-09-23 | 2, 5, 11, 25 |
| gpt-oss-120b checkpoint bytes | 14 safetensors shards totalling 65,248,893,184 bytes for 117 B parameters: 4.46 bits per parameter (derived); consistent with ~114.7 B MoE parameters in MXFP4 (4.25 bits) plus ~2.1 B in BF16 | — | [Hugging Face file listing](https://huggingface.co/api/models/openai/gpt-oss-120b/tree/main) | 2026-09-24 | 11 |
| gpt-oss-120b | L 36, d 2,880, 64 heads, 8 KV heads, head 64; sliding window 128 on alternate layers; 128 experts, 4 per token; MXFP4 MoE weights; 117 B total, 5.1 B active; fits one 80 GB GPU | 36 KiB for the 18 full-attention layers; windowed layers capped at 128 tokens | [config.json](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json), [model card](https://huggingface.co/openai/gpt-oss-120b) | 2026-09-23 | 2, 11, 26, 28 |
| Qwen3.5-397B-A17B | L 60: 45 linear-attention + 15 full-attention (`full_attention_interval` 4); full attention 32 heads, **2 KV heads**, head 256, output gate; linear attention 16 key / 64 value heads of 128, conv kernel 4, FP32 state; 512 experts, 10 per token + shared; 1 MTP layer; 262,144 context; vision tower | 30 KiB (full layers only), plus a fixed recurrent state per sequence | [config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json) | 2026-09-23 | 1, 25, 26, 27 |
| Kimi K3 | 2.8 T total, 104 B active; 93 layers = 69 Kimi Delta Attention + 24 gated MLA; 896 experts, 16 per token; 1,048,576 context; MXFP4 weights / MXFP8 activations via quantization-aware training | not stated on card | [model card](https://huggingface.co/moonshotai/Kimi-K3) | 2026-09-23 | 1, 11, 26, 27 |
| DeepSeek-V4-Pro / -Flash | 1.6 T / 49 B active and 284 B / 13 B active; 1 M context; hybrid Compressed Sparse Attention + Heavily Compressed Attention; mHC residuals; Pro at 1 M context uses 27% of V3.2's per-token FLOPs and 10% of its KV cache | — | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) (abstract) | 2026-09-23 | 1, 28 |
| Gemma 4 | E2B, E4B, 12B, 26B A4B (MoE: 128 experts, 8 active + 1 shared; 3.8 B active), 31B dense; interleaved sliding-window (512 or 1,024) and global layers, final layer global; up to 256K context; Apache 2.0 | — | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) (page updated 2026-07-30) | 2026-09-23 | 1, 28 |

## Part I — the physics of inference

| Technique | What it is | Primary source | Checked | Maturity | Chapter |
| --- | --- | --- | --- | --- | --- |
| Roofline model | Attainable FLOP/s = min(peak FLOP/s, bandwidth × arithmetic intensity) | Williams, Waterman, Patterson, CACM 52(4):65–76, 2009, [doi:10.1145/1498765.1498785](https://dl.acm.org/doi/abs/10.1145/1498765.1498785) | 2026-09-23 | industry standard | 3 |
| Forward FLOPs ≈ 2N per token | $C_{\mathrm{forward}} \approx 2N + 2 n_{\mathrm{layer}} n_{\mathrm{ctx}} d_{\mathrm{model}}$; the 2 is the multiply–accumulate | Kaplan et al., *Scaling Laws for Neural Language Models*, §2.1 eq. 2.2, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | 2026-09-23 | industry standard | 2, 3, 6 |
| Incremental decoding is bandwidth-bound; MQA | Sharing K/V across heads cuts the bytes each decode step loads | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | 2026-09-23 | industry standard | 3, 5, 25 |
| Inference partitioning and MFU/latency trade-off | Analytical model of latency vs utilisation; 29 ms/token and 76% MFU on PaLM 540B (TPU v4) | Pope et al., *Efficiently Scaling Transformer Inference*, [arXiv:2211.05102](https://arxiv.org/abs/2211.05102) | 2026-09-23 | research (MLSys 2023) | 3, 6, 31 |
| TTFT, TPOT and goodput | Goodput = max request rate served within both TTFT and TPOT objectives | Zhong et al., *DistServe*, OSDI 2024, [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) | 2026-09-23 | research; the metrics are industry standard | 4, 6, 33, 39 |

## Part II — making one request fast

| Technique | What it is | Primary source | Checked | Maturity | Chapter |
| --- | --- | --- | --- | --- | --- |
| Online softmax | One-pass softmax normaliser; basis of tiled attention | Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) | 2026-09-23 | industry standard | 8 |
| FlashAttention | IO-aware exact attention: tiles in SRAM, no N×N matrix in HBM | Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | 2026-09-23 | industry standard | 8 |
| FlashAttention-2 | Better work partitioning; 50–73% of A100 peak | Dao, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | 2026-09-23 | industry standard | 8 |
| FlashAttention-3 | Hopper: warp specialisation, TMA/WGMMA overlap, FP8; 740 TFLOP/s FP16 (75%) on H100 | Shah et al., [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) | 2026-09-23 | shipped (FlashAttention library) | 8 |
| FlashAttention-4 | Blackwell: software exponential on FMA units, conditional rescaling, TMEM and 2-CTA MMA; up to 1,613 TFLOP/s BF16 (71%) on B200; written in CuTe DSL | Zadouri et al., [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) (MLSys 2026) | 2026-09-23 | research; engine adoption to verify | 8, 13 |
| Flash-Decoding | Split K/V along the sequence, attend in parallel, merge with log-sum-exp; up to 8× end-to-end on long sequences at batch 1 | Dao, Haziza, Massa, Sizov, [CRFM post, Oct 2023](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | 2026-09-23 | shipped (FlashAttention, xFormers per post) | 9 |
| FlashInfer | Attention engine: block-sparse KV formats, JIT templates, load-balanced CUDA-graph-compatible scheduling | Ye et al., MLSys 2025, [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) | 2026-09-23 | shipped in SGLang, vLLM, MLC (per paper) | 9, 16 |
| GPTQ | One-shot second-order post-training quantisation to 3–4 bits | Frantar et al., ICLR 2023, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | 2026-09-23 | industry standard | 10 |
| AWQ | Activation-aware per-channel scaling protects ~1% salient weights | Lin et al., MLSys 2024, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | 2026-09-23 | industry standard | 10 |
| SmoothQuant | W8A8 by migrating activation outliers into weights | Xiao et al., ICML 2023, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) | 2026-09-23 | research; engine adoption to verify | 10 |
| K-quants | llama.cpp super-block formats: Q4_K 4.5 bpw, Q6_K 6.5625 bpw, mixed per-tensor (Q4_K_M) | [llama.cpp PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684), merged 2023-06-05 | 2026-09-23 | shipped in llama.cpp | 10, App D |
| Marlin | FP16×INT4 GEMM near the ideal 4× up to batch 16–32; up to 2.8× end-to-end in vLLM | Frantar et al., [arXiv:2408.11743](https://arxiv.org/abs/2408.11743) | 2026-09-23 | shipped in vLLM (per paper) | 10 |
| FP8 E4M3 / E5M2 | Two FP8 encodings; E4M3 drops infinities for range | Micikevicius et al., [arXiv:2209.05433](https://arxiv.org/abs/2209.05433) | 2026-09-23 | industry standard | 11 |
| Microscaling (MX) formats | Blocks of 32 elements share an E8M0 scale; MXFP8, MXFP6, MXFP4 (E2M1), MXINT8 (Table 1) | Rouhani et al., [arXiv:2310.10537](https://arxiv.org/abs/2310.10537); OCP MX v1.0 spec (PDF not retrievable here) | 2026-09-23 | industry standard (OCP) | 11, App E |
| NVFP4 | E2M1 values, E4M3 scale per 16 values plus FP32 per-tensor scale; 4.5 bits/value; ≤1% accuracy loss vs FP8 on DeepSeek-R1-0528 | NVIDIA, [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), 2025-06-24 | 2026-09-23 | shipped (Blackwell) | 11 |
| MXFP4 checkpoints | Models released with MXFP4 MoE weights | gpt-oss model card, [arXiv:2508.10925](https://arxiv.org/abs/2508.10925); Kimi K3 card (above) | 2026-09-23 | shipped models | 1, 11 |
| KIVI | 2-bit KV cache: keys per channel, values per token; up to 4× batch | Liu et al., ICML 2024, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) | 2026-09-23 | research | 12 |
| KVQuant | Per-channel, pre-RoPE, non-uniform KV quantisation with outliers; <0.1 perplexity loss at 3 bits | Hooper et al., NeurIPS 2024, [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) | 2026-09-23 | research | 12 |
| FP8 KV cache | `fp8`, `fp8_e4m3`, `fp8_e5m2`; per-tensor or per-head scales | [vLLM docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) (page updated 2026-08-03) | 2026-09-23 | shipped in vLLM | 12 |
| Triton | Tile-based language and compiler for DNN kernels | Tillet, Kung, Cox, MAPL 2019, [doi:10.1145/3315508.3329973](https://dl.acm.org/doi/10.1145/3315508.3329973) | 2026-09-23 | industry standard | 13 |
| CuTe DSL | Python DSL in CUTLASS 4.x mirroring CuTe layouts, copy and MMA atoms; JIT; Ampere–Blackwell (`tcgen05`, TMEM) | [CUTLASS docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) | 2026-09-23 | shipped (CUTLASS 4.x) | 13 |
| CUDA Tile, cuTile Python | Tile programming model, CUDA Tile IR virtual ISA, cuTile Python DSL; compute capability 8.x/10.x/11.x/12.x at launch | NVIDIA, [CUDA 13.1 post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/), 2025-12-04 | 2026-09-23 | shipped (CUDA 13.1) | 13 |
| TileLang | Tile-level dataflow with a separate scheduling space | Wang et al., [arXiv:2504.17577](https://arxiv.org/abs/2504.17577) | 2026-09-23 | research, open source | 13 |
| ThunderKittens | 16×16 tile primitives; matches cuBLAS and FlashAttention-3 on H100 | Spector et al., [arXiv:2410.20399](https://arxiv.org/abs/2410.20399) | 2026-09-23 | research, open source | 13 |
| CUDA graphs in vLLM | Capture to remove launch overhead; default `FULL_AND_PIECEWISE` in V1 | [vLLM design doc](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) | 2026-09-23 | shipped in vLLM | 13, 41 |

## Part II additions (checked 2026-09-24)

| Technique | What it is | Primary source | Checked | Maturity | Chapter |
| --- | --- | --- | --- | --- | --- |
| Hopper TMA, clusters, distributed shared memory | Asynchronous bulk tensor copies issued by one thread; thread-block clusters (portable 8, up to 16 on H100) that read each other's shared memory; 228 KB shared memory per SM; 50 MB L2 | [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) §1.4 | 2026-09-24 | shipped (H100) | 7, App B |
| Hopper `wgmma` | Warpgroup (128 threads) asynchronous MMA; M = 64, N = 8–256 in steps of 8, K = 16 (F16/BF16) or 32 (FP8/INT8); B from shared memory, A from shared memory or registers; accumulator in registers; fence/commit/wait protocol | [CUTLASS WGMMA programming guide](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/wgmma_programming.html) | 2026-09-24 | shipped (H100) | 7, 13 |
| Stream-K | Work-centric GEMM decomposition: each SM gets an equal share of inner-loop iterations, fixing tile-quantization waste; up to 14× over tile-based kernels across 32,824 shapes | Osama et al., [arXiv:2301.03598](https://arxiv.org/abs/2301.03598) | 2026-09-24 | shipped in CUTLASS (per paper authors' affiliation; verify) | 7 |
| FlashAttention-4 in engines | FA4 (CuTe DSL) ships as `flash-attn-4` for Hopper and Blackwell; vLLM defaults to FA4 on SM100+, FA3 on SM90, FA2 otherwise | [flash-attention README](https://github.com/Dao-AILab/flash-attention); [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends.html) (updated 2026-09-23) | 2026-09-24 | shipped in vLLM | 8, 41 |
| vLLM attention backends | FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT and MLA backends (FLASHMLA, CUTLASS_MLA, FLASHINFER_MLA, TRITON_MLA, sparse DeepSeek-V4 variants); Blackwell prefers FlashInfer, Hopper FlashAttention | [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends.html) | 2026-09-24 | shipped | 9, 41 |
| FlashMLA | DeepSeek's MLA attention kernels (SM90, SM100): dense decode up to 3,000 GB/s memory-bound and 660 TFLOP/s compute-bound on H800; paged KV; BF16 and FP8 KV; sparse kernels for DeepSeek-V3.2 (Sept 2025) | [github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) (vendor numbers) | 2026-09-24 | shipped (DeepSeek, vLLM backend) | 9, 25 |
| QuaRot | Hadamard rotations remove activation outliers so weights, activations and KV cache all go to 4 bits; ≤0.47 perplexity loss on Llama-2-70B | Ashkboos et al., [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | 2026-09-24 | research | 10, 12 |
| SpinQuant | Learned rotations for W4A4KV4; closes up to 45% more of the gap than QuaRot on Llama-3-8B | Liu et al., ICLR 2025, [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) | 2026-09-24 | research | 10, 12 |
| Machete | Mixed-input (W4A16/W8A16) GEMM for Hopper using TMA and WGMMA; vLLM backend since 0.6.2 | [vLLM PR #7174](https://github.com/vllm-project/vllm/pull/7174); lead: Red Hat developer blog 2024-10-14 | 2026-09-24 | shipped in vLLM | 10 |
| FP8 value ranges | E4M3: bias 7, no infinities, one NaN pattern, max 448, min normal 2⁻⁶, min subnormal 2⁻⁹; E5M2: bias 15, IEEE-style specials, max 57,344, min normal 2⁻¹⁴ | Micikevicius et al., [arXiv:2209.05433](https://arxiv.org/abs/2209.05433), Table 1 | 2026-09-24 | industry standard | 11, App E |
| DeepSeek-V3 FP8 recipe | E4M3 for all tensors; activations scaled per 1×128 tile, weights per 128×128 block; H800 FP8 tensor-core accumulation keeps ~14 bits, so partial sums are promoted to FP32 on CUDA cores every 128 elements | [arXiv:2412.19437](https://arxiv.org/html/2412.19437) §3.3 | 2026-09-24 | shipped (training recipe; FP8 checkpoint) | 11 |
| NVFP4 pretraining | 12B model trained on 10T tokens in NVFP4 with random Hadamard transforms, 2-D scaling, stochastic rounding and some high-precision layers; matches an FP8 baseline | NVIDIA, [arXiv:2509.25149](https://arxiv.org/abs/2509.25149) | 2026-09-24 | research | 11 |
| TurboQuant | Online vector quantization: random rotation plus Lloyd–Max scalar quantization, with a 1-bit residual correction for unbiased inner products; KV cache quality-neutral at 3.5 bits per channel | Zandieh et al., [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (venue reported as ICLR 2026 by secondary sources; unverified) | 2026-09-24 | research → shipped | 12 |
| TurboQuant in vLLM | `--kv-cache-dtype turboquant_k8v4` (FP8 keys, 4-bit values, ~2.6× smaller), plus 3- and 4-bit variants; merged 2026-04-15; on Qwen3-4B, GSM8K 0.900 → ~0.860 for k8v4, 79–100% of baseline throughput | [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) | 2026-09-24 | shipped in vLLM | 12 |
| vLLM `torch.compile` | Compiles the forward pass with attention wrapped as a custom op and the graph split at attention; piecewise CUDA graphs between attention calls; all compilation before serving; cache reusable across deployments | [vLLM design doc](https://docs.vllm.ai/en/latest/design/torch_compile.html) (updated 2025-11-28) | 2026-09-24 | shipped in vLLM | 13 |
| Megakernels | Whole forward pass in one persistent kernel: about 100 kernels per Llama-1B forward pass normally; launch ≈2.1 µs (≈1.3 µs with CUDA graphs) on H100; megakernel reached 78% of H100 bandwidth vs ≤50% for engines, <1 ms per forward pass (May 2025) | Spector et al., [Hazy Research, 2025-05-27](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (research blog, measurements by the authors) | 2026-09-24 | research; active 2026 line (ForgeMegakernel, arXiv:2609.12379) | 13 |
| ForgeMegakernel | Coding agents generate a decode megakernel per model through ten staged milestones, gated by an oracle that reads the mid-step KV cache and checks bytes, float64-referenced error and a precision contract; 14 decode configurations of models from 0.6B to 13B parameters; geometric mean 1.21× over SGLang 0.5.18 and 1.54× over Mirage Persistent Kernel; 50.5–85.9% of bandwidth; one H100 80GB | [arXiv 2609.12379](https://arxiv.org/abs/2609.12379) (11 Sep 2026; abstract and HTML body read) | 2026-09-24 | research | 13 |

## Part III additions (checked 2026-09-24)

Engine behaviour is read at pinned commits: llama.cpp `d006858` (the source of
the Homebrew build 8660 measured in `research/measurements/2026-09-24-m1-part3.md`)
and vLLM `main` at `bcdacfc` (2026-09-23).

| Technique | What it is | Primary source | Checked | Maturity | Chapter |
| --- | --- | --- | --- | --- | --- |
| llama.cpp server scheduling | Per iteration: one token per decoding slot first, then prompt tokens up to the logical batch size (`-b`, default 2048); new prompts join only if continuous batching is on (default) or nothing is decoding; slots batch together only if same task type and equal LoRA settings (`can_batch_with`) | `tools/server/server-context.cpp` at `d006858` | 2026-09-24 | shipped (llama.cpp) | 15, 18, 21 |
| llama.cpp KV layout and host prompt cache | `-kvu` unified KV is the default only when the slot count is automatic; otherwise each slot gets `n_ctx / n_parallel` cells; evicted slot prompts saved to a host-memory prompt cache (`--cache-ram`, default 8,192 MiB) and restored by prefix similarity | `server-context.cpp`, `llama-server --help` at build 8660; PR 16391 cited by the help text | 2026-09-24 | shipped (llama.cpp) | 16, 17, 34 |
| llama.cpp KV exhaustion | On decode failure: try clearing idle slots, else halve the batch; at batch 1, "Context size has been exceeded." is sent to **every** processing slot and all are released; a TODO proposes terminating only the largest | `server-context.cpp` at `d006858` | 2026-09-24 | shipped (llama.cpp) | 19 |
| llama.cpp disconnect detection | HTTP threads poll `is_connection_closed` only when a 1 s wait for results times out; every result for any request notifies all waiters and restarts the wait, so a disconnected non-streamed request is not cancelled while other results flow (measured) | `server-queue.cpp` (`recv_with_timeout`), `server-context.cpp` (`HTTP_POLLING_SECONDS = 1`) at `d006858`; cpp-httplib `is_socket_alive` | 2026-09-24 | shipped (llama.cpp); bug not yet reported upstream | 14 |
| llama.cpp grammar sampling | Sample without the grammar, check only the chosen token, apply the full grammar mask and resample only if it is invalid; llguidance optional at build time | `common/sampling.cpp` at `d006858` | 2026-09-24 | shipped (llama.cpp) | 20 |
| vLLM V1 scheduler and engine | Scheduling decision = {request_id: num_tokens}; chunked prefill and prefix caching on by default; EngineCore in its own process (ZeroMQ); persistent batch; prefix caching costs <1% at 0% hits | [vLLM V1 blog, 2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | 2026-09-24 | shipped (vLLM) | 15, 17, 18 |
| vLLM preemption and priority | Preemption mode RECOMPUTE by default in V1; chunked prefill decode-first; `max_num_batched_tokens` trades ITL (2,048) against TTFT/throughput (>8,192 for small models on large GPUs); victim = last running request (FCFS) or max(priority, arrival) (priority policy); victim's blocks freed, computed tokens reset, prepended to waiting queue | [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.html); `vllm/v1/core/sched/scheduler.py`, `vllm/config/scheduler.py` at `bcdacfc` | 2026-09-24 | shipped (vLLM) | 16, 18, 19 |
| vLLM Model Runner V2 | Async scheduling as a design constraint: step N+1 prepared while N runs; inputs built on the GPU; opt-in `VLLM_USE_V2_MODEL_RUNNER=1`; 6.3% lower TPOT on 4×GB200 (GLM-4.7-FP8, speculative decoding) | [vLLM blog, 2026-03-24](https://vllm.ai/blog/2026-03-24-mrv2) (developers' measurement) | 2026-09-24 | shipped (vLLM, opt-in) | 15 |
| vLLM automatic prefix caching | Block hash = f(parent hash, block tokens, extra keys: LoRA, multimodal, cache salt); sha256 default; LRU free-block queue doubles as eviction order; `cache_salt` isolates reuse against timing attacks | [Prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) | 2026-09-24 | shipped (vLLM) | 16, 17 |
| vLLM hybrid KV cache manager | Layers grouped into KV-cache groups with one page size; sliding-window groups keep only recent blocks; prefix hits intersected across groups | [Hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html) | 2026-09-24 | shipped (vLLM) | 16, 27 |
| vLLM disconnect handling | `with_cancellation` races each handler against an ASGI `http.disconnect` listener; `AsyncLLM.generate` aborts the engine request on cancellation | `vllm/entrypoints/serve/utils/api_utils.py`, `vllm/v1/engine/async_llm.py` at `bcdacfc` | 2026-09-24 | shipped (vLLM) | 14 |
| SGLang v0.4 | Overlap ("zero-overhead") scheduler one batch ahead, 1.1× over v0.3; cache-aware load balancer with approximate radix tree per worker, hit rate 20%→75%, 1.9× throughput; XGrammar up to 10× faster JSON decoding | [LMSYS blog, 2024-12-04](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) (developers' measurements) | 2026-09-24 | shipped (SGLang) | 15, 17, 20 |
| vAttention | Contiguous virtual KV memory with on-demand physical pages via CUDA VMM APIs; unmodified kernels; up to 1.23× throughput vs paged kernels in FlashAttention/FlashInfer | Prabhu et al., ASPLOS 2025, [arXiv:2405.04437](https://arxiv.org/abs/2405.04437) | 2026-09-24 | research | 16 |
| Virtual Token Counter (fairness) | Token-weighted service counter; serve least-served backlogged client; \|W_f − W_g\| ≤ 2·max(w_p·L_input, w_q·M), tight, work-conserving | Sheng et al., [arXiv:2401.00588](https://arxiv.org/abs/2401.00588) (HTML body, Theorem 4.4) | 2026-09-24 | research | 19 |
| Mooncake early rejection | Reject at arrival if decode is predicted to be overloaded; naive early rejection oscillates; prediction-based fix; replay of 23,000 requests at 2×: 4,183 / 3,771 / 3,589 rejected (baseline / early / predicted) | Qin et al., [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) v4 (HTML body) | 2026-09-24 | shipped (Moonshot AI) | 19, 33 |
| LoRA | Frozen weights plus trainable low-rank updates; 10,000× fewer trainable parameters, 3× less GPU memory than full fine-tuning GPT-3 175B; mergeable, so no added latency when served alone | Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | 2026-09-24 | industry standard | 21 |
| S-LoRA | Adapters in host memory, fetched on demand; Unified Paging of adapter weights and KV cache; heterogeneous-batching kernels; up to 4× throughput vs HF PEFT and vLLM (then), orders of magnitude more adapters | Sheng et al., [arXiv:2311.03285](https://arxiv.org/abs/2311.03285) | 2026-09-24 | research | 21 |
| Punica | SGMV kernel batches requests for different LoRA models over one base model; 12× throughput, +2 ms per token | Chen et al., [arXiv:2310.18547](https://arxiv.org/abs/2310.18547) | 2026-09-24 | research | 21 |
| vLLM multi-LoRA | Adapters batched with base-model requests up to `max_loras`; `max_lora_rank` sizes memory; runtime load/unload endpoints behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) | 2026-09-24 | shipped (vLLM) | 21 |
| RouteLLM | Router between a strong and a weak model trained on preference data; cost down by over 2× in some cases without quality loss | Ong et al., [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) | 2026-09-24 | research | 21 |
| ServerlessLLM | Multi-tier checkpoint loading, live migration, locality-aware scheduling; 10–200× lower latency than serverless baselines | Fu et al., OSDI 2024, [arXiv:2401.14351](https://arxiv.org/abs/2401.14351) | 2026-09-24 | research | 21 |
| Grammar-aligned decoding | Masking distorts the distribution; ASAp samples from the model's distribution conditioned on the grammar | Park et al., NeurIPS 2024, [arXiv:2405.21047](https://arxiv.org/abs/2405.21047) | 2026-09-24 | research | 20 |
| vLLM structured outputs | Backends xgrammar and guidance (outlines, lm-format-enforcer also named); `auto` picks per request; JSON schema, regex, choice, EBNF, structural tags | [Structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) | 2026-09-24 | shipped (vLLM) | 20 |
| Prompt caching (Anthropic) | Explicit or automatic breakpoints; 5-minute (1.25× write) or 1-hour (2× write) lifetimes refreshed on use; reads 0.1× (0.05× or 0.025× on some models); minimum 512–4,096 tokens by model; isolated per organization and workspace | [Prompt caching docs](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching) | 2026-09-24 | shipped product | 17 |
| Prompt caching (OpenAI) | Automatic; ≥1,024 tokens (GPT-5.6+), 128-token increments on earlier models; ≥30 min retention (GPT-5.6+); cached input 0.1× (GPT-5.6+, "up to 90%" earlier); routed by a hash of initial tokens; not shared across organizations | [Prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) | 2026-09-24 | shipped product | 17 |
| Prompt-cache timing audit | Statistical timing tests detected cross-user cache sharing at seven API providers including OpenAI (2025) | Gu et al., ICML 2025, [arXiv:2502.07776](https://arxiv.org/abs/2502.07776) | 2026-09-24 | research | 17 |
| Responses API streaming and background mode | Typed SSE events (`response.created`, `response.output_text.delta`, `response.completed`, `error`); background responses polled at `/v1/responses/{id}`, cancelled idempotently at `/v1/responses/{id}/cancel`, streams resumable with `sequence_number` and `starting_after` | [Streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [Background mode](https://developers.openai.com/api/docs/guides/background) | 2026-09-24 | shipped product | 14 |

## Seeds for Parts III–VIII (verified; chapters not yet drafted)

| Technique | What it is | Primary source | Checked | Maturity | Chapter |
| --- | --- | --- | --- | --- | --- |
| Iteration-level scheduling (Orca) | Schedule per iteration, batch selectively; 36.9× GPT-3 175B throughput vs FasterTransformer | Yu et al., [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | 2026-09-23 | industry standard | 15 |
| PagedAttention / vLLM | Block-mapped KV cache, near-zero waste; 2–4× throughput | Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | 2026-09-23 | industry standard | 16 |
| RadixAttention (SGLang) | Radix tree over KV for automatic prefix reuse; compressed FSM decoding | Zheng et al., [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | 2026-09-23 | shipped in SGLang | 17, 20 |
| Chunked prefill, stall-free scheduling | Equal-size prefill chunks batched with decodes; 2.6–5.6× capacity | Agrawal et al., Sarathi-Serve, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | 2026-09-23 | research; engine adoption to verify | 18 |
| XGrammar | Context-independent token prechecks, persistent stack, overlap with GPU; near-zero serving overhead | Dong et al., MLSys 2025, [arXiv:2411.15100](https://arxiv.org/abs/2411.15100) | 2026-09-23 | research; engine adoption to verify | 20 |
| Speculative decoding | Draft then verify; output distribution unchanged; 2–3× on T5-XXL | Leviathan, Kalman, Matias, ICML 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | 2026-09-23 | industry standard | 22 |
| Speculative sampling | Modified rejection sampling preserves the target distribution; 2–2.5× on Chinchilla 70B | Chen et al., [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) | 2026-09-23 | industry standard | 22 |
| EAGLE-3 | Direct token prediction with multi-layer feature fusion ("training-time test"); up to 6.5× | Li et al., [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | 2026-09-23 | shipped in SGLang and vLLM | 23 |
| P-EAGLE | Parallel drafting: all K draft tokens in one pass via a learnable shared hidden state; 1.10–1.36× over EAGLE-3; gains shrink at high concurrency | Hui et al., [arXiv:2602.01469](https://arxiv.org/abs/2602.01469); vLLM ≥ 0.16.0 (`parallel_drafting`), [vLLM post 2026-03-13](https://vllm.ai/blog/2026-03-13-p-eagle) | 2026-09-23 | shipped in vLLM | 23, 24 |
| Multi-token prediction module | Extra layer(s) predicting further tokens, usable as a drafter | DeepSeek-V3 config `num_nextn_predict_layers: 1`; Qwen3.5 `mtp_num_hidden_layers: 1` | 2026-09-23 | shipped models | 23 |
| GQA | Intermediate number of KV heads; uptrain with 5% of pre-training compute | Ainslie et al., EMNLP 2023, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | 2026-09-23 | industry standard | 25 |
| Multi-head latent attention | Low-rank latent KV; 93.3% smaller cache, 5.76× max throughput vs DeepSeek 67B | DeepSeek-V2, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | 2026-09-23 | shipped models (DeepSeek-V2/V3, Kimi K3 gated MLA) | 25 |
| Native Sparse Attention | Hierarchical compression plus selection, natively trained | Yuan et al., [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | 2026-09-23 | research | 28 |
| DeepSeek Sparse Attention | Lightning indexer selects top-k tokens under MLA | DeepSeek-V3.2 report, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) (read via search excerpt; full read pending) | 2026-09-23 | shipped model | 28 |
| Gated DeltaNet | Gating plus delta-rule linear attention; hybrids with attention | Yang, Kautz, Hatamizadeh, ICLR 2025, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | 2026-09-23 | shipped in Qwen3.5 (per Qwen; config shows the 3:1 linear/full layout) | 27 |
| Batch invariance | Nondeterminism comes from batch-size-dependent kernels; batch-invariant RMSNorm, matmul, attention make 1,000/1,000 completions identical at 1.6–2.1× cost | He / Thinking Machines, [post 2025-09-10](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (lead: blog, with code) | 2026-09-23 | research; engine adoption to verify | 38 |
| Mooncake | KV-cache-centric disaggregation using CPU, DRAM, SSD; serves Kimi | Qin et al., [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) | 2026-09-23 | shipped (Moonshot AI) | 33, 34 |
| DeepSeek V3/R1 production statistics | Operator-reported: prefill EP32 over 4 nodes, decode EP144 over 18 nodes; ~73.7k input tok/s (incl. cache hits) and ~14.8k output tok/s per 8-GPU H800 node; 608B input tokens (56.3% from an on-disk KV cache) and 168B output tokens in 24 h; 20–22 tok/s per user; 226.75 nodes average; $2/GPU-hour assumed | [open-infra-index day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (operator's own report, not independently measured) | 2026-09-23 | shipped (DeepSeek service) | 1, 6, 32, 34 |
| DeepEP | All-to-all MoE dispatch/combine kernels, FP8, low SM use; README reports V2 | [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) (README numbers are vendor claims) | 2026-09-23 | shipped (open source) | 32 |

## Leads to verify before the chapters that need them

Not yet read at a primary source; do not cite until they are.

- SGLang's scheduler, KV manager and overlap loop at a pinned commit, and the
  vLLM V1 internals beyond those logged above (Chapter 41).
- NVIDIA Dynamo, llm-d, LMCache, SGLang HiCache, NIXL (Chapters 33–34).
- Medusa, lookahead decoding, *Speculative Speculative Decoding*
  ([arXiv:2603.03251](https://arxiv.org/abs/2603.03251)) (Chapters 23–24).
- Kimi Linear / Kimi Delta Attention paper; Mamba-2; Qwen's own description of
  Qwen3.5's linear layers (Chapter 27).
- StreamingLLM attention sinks, H2O, SnapKV, YaRN, DeepSeek-V4 CSA/HCA details
  (Chapter 28).
- FlexGen and PowerInfer (Chapter 35);
  NVLink-C2C and unified-memory systems (Chapter 37).
- MLPerf Inference v6.0 rules and results; InferenceMAX methodology (Chapter 39).
- FlashAttention-4 and FlashMLA adoption in SGLang (vLLM checked 2026-09-24);
  FlashAttention-4's forward-pass numbers beyond the abstract (Chapters 8, 25).
