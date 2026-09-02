# Lab 14 — Change the Seed

Chapter: 4. Artifact: compared stochastic traces and a written reproduction
contract.

## CHECK

List every input besides seed that must stay fixed for a repeatable run:
engine version, executable/toolchain/target, model parameters, tokenizer,
prompt tokens, sampling policy, and floating-point path.

## BUILD

Run the same stochastic command twice:

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace --sample --temperature 1 \
  --top-k 3 --top-p .9 --seed 42 --max-tokens 3 'I like'
```

Compare RNG draws and token IDs, not timing fields.

## BREAK

Change only the seed, then only temperature. A different seed may—not must—
change a short sequence. A different distribution can change how the same draw
maps to a token.

## EXTEND

Run two request samplers in alternating order. Prove neither consumes values
from the other's RNG stream. Explain why one global RNG would fail under
concurrent scheduling.

