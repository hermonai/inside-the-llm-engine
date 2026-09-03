# Reference Implementations

Independent clarity-first Python/Rust oracles and hand-computable fixtures live
here. They define semantics for differential tests; they are not performance
baselines unless measured under the benchmark policy.

- [`engine-0-oracle.md`](engine-0-oracle.md) independently predicts ENGINE-0's
  first token, terminal outcome, and semantic event order.
- [`chapter03_oracle.py`](python/chapter03_oracle.py) independently computes
  ENGINE-1's embedding and full logit vector.
- [`chapter04_sampling_oracle.py`](python/chapter04_sampling_oracle.py)
  independently computes stable softmax, temperature, top-k/top-p,
  renormalization, and fixed-draw categorical selection.
- [`chapter05_tensor_oracle.py`](python/chapter05_tensor_oracle.py)
  independently derives canonical strides, exact offsets, transpose logical
  order, reshape offsets, and contiguous materialization order.
- [`chapter06_matmul_oracle.py`](python/chapter06_matmul_oracle.py)
  independently computes dot, GEMV, GEMM, transpose-view logical products, and
  binary32-rounded fractional fixtures.
