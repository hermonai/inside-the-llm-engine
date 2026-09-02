# Chapter 2 Independent Tokenizer Oracles

These fixtures are written independently of the Rust implementation. They are
small enough to verify with pencil and paper.

## Byte oracle

For every byte `b` in `0..=255`:

```text
encode([b]) = [TokenId(b)]
decode(TokenId(b)) = [b]
```

Concatenation extends the rule to any byte sequence, including malformed UTF-8.
Interpreting the decoded bytes as text is a separate operation.

## BPE oracle: `lower`

Start with byte identities:

```text
[l, o, w, e, r]
```

Apply the lowest-rank available adjacent merge, one occurrence at a time:

```text
rank 0 (l,o)->lo       [lo, w, e, r]
rank 1 (lo,w)->low     [low, e, r]
rank 2 (e,r)->er       [low, er]
rank 3 (low,er)->lower [lower]
```

Expected final ID: `[259]`. Expected decoded bytes: `lower`.

For `lolo`, rank 0 occurs twice. The contract chooses the leftmost occurrence,
then repeats, producing `[lo, lo]` or `[256, 256]`. For `xyz`, no pair has a
rule, so the expected IDs are the byte values `[120, 121, 122]`.

## Special-token oracle

Ordinary encoding and trusted control insertion are disjoint:

```text
encode("<|assistant|>") != [ASSISTANT]
insert_special(ASSISTANT) = [1006]
decode_ordinary(1006) = error
```

## UTF-8 stream oracle

The scalar `世` is `E4 B8 96` in UTF-8. If two generated IDs decode to
`[E4 B8]` and `[96]`, the first ID produces a token event but no text event.
The second completes the scalar and emits exactly `世`.

`C3 28` is definitively malformed and must fail without U+FFFD replacement.
`F0 9F` is a possible prefix but is incomplete; reaching a successful stop
with those bytes pending is a failed terminal outcome under the Chapter 2
policy.

## Chat-template oracle

For system `be exact`, user `lower`, and `add_generation_prompt=true`:

```text
BOS SYSTEM encode("be exact") END_TURN
    USER encode("lower")    END_TURN ASSISTANT
```

The naive bytes `system: be exact\nuser: lower\n` contain no role/control IDs
and therefore cannot equal the correct input sequence.

