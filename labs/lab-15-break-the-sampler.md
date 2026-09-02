# Lab 15 — Break the Sampler

Chapter: 4. Artifact: typed-error table and exactly-once terminal evidence.

## CHECK

Predict the category for each case: negative temperature, `top_k=0`,
`top_p=0`, `top_p>1`, empty logits, non-finite probability, all-zero candidate
mass, and a draw outside `[0,1)`.

## BUILD

```sh
cd code/mini-engine
cargo test -p engine0 invalid_sampling_configuration_is_rejected_without_fallback
cargo test -p engine0 categorical_selection_rejects_invalid_draws_and_distributions
cargo test -p engine0 invalid_sampler_config_fails_once_without_admission_or_model_work
```

## BREAK

Temporarily replace one typed failure with greedy fallback. Show which test
fails and why silently changing policy is an inference-correctness bug.

## EXTEND

Add one deterministic malformed fixture not already covered. Assert error
category, zero post-terminal events, and exactly one terminal outcome. Revert
duplicate experiments; retain a new case only if it expands the matrix.
