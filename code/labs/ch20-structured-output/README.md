# Chapter 37 lab — Structured Output

## Predict
At the state immediately after `{"ok":`, decide which vocabulary pieces can begin a
valid continuation. Remember that a token's *entire* byte string must be legal.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
The parser's committed bytes are the source of truth. `CachedMask` memoizes by parser
state; the independent test oracle concatenates complete token bytes and directly
checks the two permitted strings with a prefix comparison. It does not call the
parser transition or use its precomputed state table.

## Break
1. Validate only the first byte of a token: `whole_token_suffix_is_checked` must fail.
2. Treat every prefix as EOS-valid: `eos_only_at_accepting_state` must fail.
3. Mutate committed state while trying a candidate: rollback/parity tests must fail.
4. Key the cache by token position instead of parser state: add true/false branches
   reaching different states at equal positions and observe the wrong mask.

## Limits
This is an inspectable regular-language teaching fixture, not a CFG/PDA implementation
or performance benchmark. Production systems need Unicode/tokenizer edge handling,
persistent stacks, sparse/dense mask strategies, device integration and concurrency.
