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
| AMD Instinct MI355X | 288 GB HBM3E | 8 TB/s | MXFP4/MXFP6 supported; compute peaks not yet read from a primary table (product page and brochure timed out) | [amd.com MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) (via search excerpt) | 2026-09-23 | 3, App B |
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
| Qwen3-8B | L 36, d 4,096, FFN 12,288, 32 heads, 8 KV heads, head 128, vocab 151,936, untied | 144 KiB | [config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) | 2026-09-23 | 2, 5 |
| Qwen3-30B-A3B | L 48, d 2,048, 32 heads, 4 KV heads, head 128; 128 experts, 8 per token, expert FFN 768 | 96 KiB | [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | 2026-09-23 | 2, 26, 36 |
| DeepSeek-V3 | L 61 (first 3 dense), d 7,168, 128 heads; MLA `kv_lora_rank` 512 + `qk_rope_head_dim` 64; 256 routed + 1 shared expert, 8 per token; 1 MTP layer; FP8 E4M3 weights, 128×128 blocks; 671 B total, 37 B active | 68.6 KiB (576 elements × 61 layers × 2 bytes); MHA with the same heads would be ≈4.8 MiB | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | 2026-09-23 | 2, 5, 11, 25 |
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

- Current vLLM V1 and SGLang scheduler, KV manager and overlap architectures,
  at pinned commits (Chapters 15–19, 41).
- NVIDIA Dynamo, llm-d, LMCache, SGLang HiCache, NIXL (Chapters 33–34).
- Medusa, lookahead decoding, *Speculative Speculative Decoding*
  ([arXiv:2603.03251](https://arxiv.org/abs/2603.03251)) (Chapters 23–24).
- Kimi Linear / Kimi Delta Attention paper; Mamba-2; Qwen's own description of
  Qwen3.5's linear layers (Chapter 27).
- StreamingLLM attention sinks, H2O, SnapKV, YaRN, DeepSeek-V4 CSA/HCA details
  (Chapter 28).
- S-LoRA and Punica kernels (Chapter 21); FlexGen and PowerInfer (Chapter 35);
  NVLink-C2C and unified-memory systems (Chapter 37).
- MLPerf Inference v6.0 rules and results; InferenceMAX methodology (Chapter 39).
- FlashAttention-4 and FlashMLA adoption in vLLM/SGLang; FlashAttention-4's
  forward-pass numbers beyond the abstract (Chapters 8, 25).
- AMD MI355X dense compute peaks from AMD's datasheet (Chapter 3, Appendix B).
