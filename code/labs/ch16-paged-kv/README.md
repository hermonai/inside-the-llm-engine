# Chapter 33 lab — paged KV ownership

This is a deterministic CPU model of the allocator and address-translation
rules. An `i32` is a stand-in for one token's KV payload. The lab does not
claim GPU performance.

## Predict

Block size is four. Append `[10,11,12,13,14,15]` to A.

Predict:

- how many blocks A owns;
- which in-block offset contains logical position 5;
- what must happen if B forks A at length 6 and then appends `99`.

A needs two blocks. Position 5 is offset 1 in logical block 1. Because the
second block is a shared partial tail after the fork, B must copy that tail
before writing.

## Run

```bash
cargo fmt --check
cargo test
cargo run -- trace
```

The executable prints physical block handles and logical histories. Tests
compare the paged representation with a flat logical vector.

## Explain

A handle is `(physical_block_id, generation)`. The request table is ordered by
logical block number. Physical IDs may be in any order.

The pool preserves four invariants:

1. a logical position maps through exactly one live generation-qualified handle;
2. shared blocks are read-only until a writer obtains a private COW tail;
3. OOM leaves the existing logical history unchanged;
4. a block is reusable only after its reference count reaches zero.

## Break

Meaningful bugs to try:

- remove COW and observe B's append mutate A's shared tail;
- ignore OOM and append a table entry with no physical backing;
- free a block when one fork terminates even though another still references it;
- validate only physical block ID and accept an old handle after reuse.

The unit tests are intended to reject these changes.

## Not demonstrated

This lab does not implement GPU attention, block hashing, automatic prefix
caching, eviction policy, CUDA stream/event reclamation, offload, distributed
KV transfer, or a production cache layout.
