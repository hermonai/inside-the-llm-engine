# Benchmark Policy

No naked performance claim enters the manuscript.

## Required record

Every result identifies:

- repository commit and date;
- build profile, compiler, relevant flags, and runtime version;
- machine, CPU, GPU, RAM, storage where relevant, OS, and driver;
- model, exact artifact/revision, quantization, and model configuration;
- prompt/context and requested/actual output length;
- concurrency, arrival pattern, repetitions, warmup, and statistic;
- runtime mode, provider, cache state, and warm/cold state;
- control/baseline, harness, command, and raw result location.

If any material field is unknown, label the result exploratory and do not use
it for a headline comparison.

## Measurement rules

Correctness gates precede timing. Separate time-to-first-token, inter-token
latency, request latency, throughput, fairness, memory, and energy rather than
collapsing them into one “speed.” Distinguish model demand, logical bytes,
physical bytes, compute, transfer, launch, synchronization, storage I/O, and
queue delay.

Use distributions and tail percentiles when serving behavior matters. Repeat
enough to expose variance and report the chosen statistic. Control thermal,
frequency, competing load, compilation, model load, allocator, and cache state
where they can change the result.

Never multiply independent benchmark ratios and call the product a measured
end-to-end result. Label projections and analytic break-even models as
estimates. Storage bandwidth without model execution is not tokens per second.

## Comparisons and negative results

Compare the same model semantics, quantization, output stopping rules, workload,
and effective concurrency. Disclose feature differences. Preserve negative
results with their hypotheses and controls; a failed optimization is evidence
when the experiment is reproducible.

## Durable layout

Store methodology and analysis under `research/benchmarks/`; store small raw
text/JSON/CSV outputs when licensing and size permit. Large artifacts belong in
an external durable store with checksums and retrieval instructions. Chapter
prose links to the record rather than copying an untraceable number.
