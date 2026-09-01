# ENGINE-0 Independent Oracle

This oracle is intentionally a table, not a call into ENGINE-0. It gives the
expected result by inspection.

## Vocabulary

| ID | Piece | Kind |
| ---: | --- | --- |
| 0 | `<eos>` | end-of-sequence |
| 1 | `blue` | text |
| 2 | `green` | text |

## Candidate source and greedy rule

At generation step 0:

| Token | Score |
| --- | ---: |
| `blue` | 9 |
| `green` | 4 |
| `<eos>` | 1 |

Greedy selection chooses the greatest score, so the first emitted token is
`blue` (ID 1).

At generation step 1:

| Token | Score |
| --- | ---: |
| `<eos>` | 10 |
| `blue` | 1 |

The selected end marker is not emitted as text. It produces the completed
terminal outcome `EndOfSequence`.

## Expected semantic stream

```text
Token { request_id: 1, index: 0, token: Token(1, "blue") }
Terminal { request_id: 1, outcome: Completed(EndOfSequence) }
```

Expected trace order, ignoring machine-dependent timestamps:

```text
Admitted
ExecutionStarted
ModelInvoked(step=0)
TokenSelected(step=0, token_id=1)
TokenEmitted(index=0, token_id=1)
ModelInvoked(step=1)
TokenSelected(step=1, token_id=0)
Terminal(Completed(EndOfSequence))
```

These integer scores are pedagogical candidates, not neural-network logits.
Chapter 3 supplies a numerical model; Chapter 4 supplies complete logit and
sampling semantics.
