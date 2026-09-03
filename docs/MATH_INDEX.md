# Mathematical Index

This index names the reusable equations established in completed Chapters 1–7.
The identifiers are semantic and stable even if manuscript line numbers move.

| ID | Equation or contract | Chapter | Meaning |
| --- | --- | ---: | --- |
| `LATENCY-TOTAL` | $T_{total}$ decomposition | 1 | Separates network, queue, preparation, prefill, per-token decode, and stream time. |
| `LATENCY-ENDPOINTS` | TTFT, ITL, request latency | 1 | Defines latency as differences between named timestamps. |
| `THROUGHPUT-RATES` | requests/s and tokens/s | 1 | Defines aggregate rates with explicit units. |
| `TOKEN-ROUNDTRIP` | $\operatorname{decode}(\operatorname{encode}(x))$ | 2 | States the conditional byte-round-trip contract. |
| `LM-CONDITIONAL` | $P(x_{t+1}\mid x_{0:t})$ | 3 | Defines next-token language-model output. |
| `EMBEDDING-LOOKUP` | $\mathbf{x}=\mathbf{E}_{t,:}$ | 3 / 7 | Maps one token identity to one owned model-width activation. |
| `EMBEDDING-SEQUENCE` | $X_{ij}=E_{t_i,j}$ | 7 | Gathers `[T]` token identities from `[V,D]` parameters into `[T,D]` activations. |
| `OUTPUT-PROJECTION` | $\mathbf{z}=\mathbf{W}\mathbf{h}+\mathbf{b}$ | 3 | Produces one logit per vocabulary identity. |
| `AUTOREGRESSIVE-FACTORIZATION` | $P(x_{0:T})$ product | 4 | Factors a sequence into next-token conditionals. |
| `STABLE-SOFTMAX` | shifted exponential normalization | 4 | Produces finite probabilities without changing the distribution. |
| `TEMPERATURE` | $\mathbf{z}'=\mathbf{z}/\tau$ | 4 | Rescales logits before filtering and sampling. |
| `CATEGORICAL-CDF` | cumulative interval selection | 4 | Maps one uniform draw to one token identity. |
| `TENSOR-ELEMENT-COUNT` | $N(\mathbf d)=\prod d_a$ | 5 | Counts logical elements with checked arithmetic. |
| `TENSOR-OFFSET` | $o(\mathbf i)=b+\sum i_as_a$ | 5 | Maps a logical index to physical element offset. |
| `TENSOR-STORAGE-EXTENT` | $L_{min}=1+b+\sum(d_a-1)s_a$ | 5 | Bounds storage for a nonnegative-stride view. |
| `GEMM-CONTRACTION` | $C_{ij}=\sum_k A_{ik}B_{kj}$ | 6 | Defines matrix multiplication and its contracted dimension. |
| `GEMM-WORK` | $F_{GEMM}\approx2MKN$ | 6 | Counts multiply-add work under the stated FLOP convention. |
| `ARITHMETIC-INTENSITY` | $I=F/Q$ | 6 | Relates arithmetic work to bytes moved. |
| `ROOFLINE-BOUND` | $P\le\min(P_{peak},B_{memory}I)$ | 6 | Gives an analytical throughput ceiling, not a measurement. |
| `NUMERICAL-TOLERANCE` | absolute-plus-relative bound | 6 | Defines the differential comparison criterion. |
| `ROOT-MEAN-SQUARE` | $\sqrt{D^{-1}\sum_i x_i^2}$ | 7 | Defines the uncentered magnitude statistic over one model-width vector. |
| `RMSNORM` | $y_i=x_iw_i/\sqrt{D^{-1}\sum_jx_j^2+\epsilon}$ | 7 | Defines epsilon-inside-root rescaling followed by learned element-wise scale. |
| `RMSNORM-PAYLOAD` | $Q\approx16D$ bytes | 7 | Models two input reads, one weight read, and one output write for F32 payload. |

Local symbol tables in each chapter remain authoritative for scope-specific
meaning. See [`MATH_STYLE.md`](MATH_STYLE.md) for notation and review rules.
