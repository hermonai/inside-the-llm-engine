# Lab 10 — Change Temperature

Chapter: 4. Artifact: a three-row probability table. Oracle: Chapter 4 Python
oracle and Rust sampling tests.

## CHECK

Predict which token remains largest when `[1,2,3]` is divided by `T=0.5`,
`T=1`, and `T=2`. Explain why positive temperature cannot change ordering.

## BUILD

Run the oracle three times with `--temperature 0.5`, `1`, and `2`. Compare:

```text
0.5 -> [0.0158762400, 0.1173104278, 0.8668133322]
1.0 -> [0.0900305732, 0.2447284711, 0.6652409558]
2.0 -> [0.1863237232, 0.3071958857, 0.5064803911]
```

## BREAK

Try zero, negative, infinity, and NaN through ENGINE-1's API. Require typed
validation; do not replace the error with greedy behavior.

## EXTEND

Plot or tabulate the largest probability over more positive temperatures.
Do not call temperature “randomness”: repeat with a fixed artificial draw and
identify the separate distribution-shape and selection stages.

