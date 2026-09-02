# Chapter 2 Research — From Text to Tokens

Inspection date: 2026-09-02.

## Question

What exact contract turns input text into the vocabulary identifiers consumed
by a model, and how can an inference runtime turn generated identifiers back
into a valid UTF-8 stream without confusing tokens, characters, bytes, or chat
messages?

Chapter 2 must replace ENGINE-0's opaque prompt and one-token/one-string
simplification. It must not introduce learned weights, embeddings, logits, or
Transformer computation. The durable boundary is:

```text
input bytes -> tokenizer semantics -> token IDs -> request/model boundary
generated token IDs -> token bytes -> UTF-8 framing -> output text pieces
```

The tokenizer is model semantics, not a generic convenience. Vocabulary,
normalization, merge/model rules, byte fallback, special-token identities, and
chat formatting must agree with the model artifact and training convention.

## Scope and truth categories

- **CURRENT** describes the reachable Hermon path at commit `472a44c`.
- **LIBRARY** describes Hermon's standalone tokenizer skeleton, which is not
  reached by the current real-model paths.
- **EXTERNAL** describes primary specifications, source, and official project
  documentation at the versions recorded below.
- **INFERENCE** marks conclusions derived by this book from those sources.
- **ENGINE-0/Chapter 2** describes the teaching implementation. Its BPE model,
  vocabulary, special tokens, and chat template are deliberately tiny.

This chapter does not claim that one tokenizer algorithm is universally best.
It makes no tokenizer-speed comparison. Chapter 3, not this chapter, introduces
genuine numerical inference.

## Primary sources and recorded versions

Repository heads were resolved with `git ls-remote ... HEAD` on the inspection
date. Mutable documentation links are paired with the recorded source head or
standard version used for the research pass.

| Subject | Recorded version | Primary evidence |
| --- | --- | --- |
| Unicode and UTF-8 | Unicode 17.0.0; RFC 3629 | Unicode Standard Chapter 3 definitions and well-formed UTF-8 tables; Unicode Standard Annex #15 normalization forms; IETF UTF-8 transformation format |
| Subword BPE | `rsennrich/subword-nmt` `92d6139`; Sennrich, Haddow, and Birch (2016) | Official implementation and paper linked from the repository |
| GPT-2 byte-level BPE | `openai/gpt-2` `9b63575` | `src/encoder.py`, especially byte-to-Unicode mapping, merge ranks, `bpe`, `encode`, and `decode` |
| SentencePiece | `google/sentencepiece` `ac0f71d` | Official README, `sentencepiece_model.proto`, normalization and special-symbol documentation |
| Hugging Face Tokenizers | `huggingface/tokenizers` `d582781` | Official component documentation for normalizer, pre-tokenizer, model, post-processor, and decoder |
| Chat templates | `huggingface/transformers` `e15d467` | Official chat-template documentation and source conventions |
| llama.cpp | Hermon pin `389ff61`; upstream head observed `b81c99b` | Pinned `include/llama.h`, `src/llama-vocab.cpp`, and chat-template implementation reached through Hermon's shim |
| OpenAI comparison tokenizer | `tiktoken` 0.14.0; repository head `4e71bbe` | Official package, repository, and named `cl100k_base` encoding |
| SentencePiece comparison tokenizer | SentencePiece 0.2.2; official fixture at `ac0f71d` | `data/wagahaiwa_nekodearu_ja_bpe_byte_2000.model`, SHA-256 `6f00a9995a025eab01394c94e6e6b73904ca172762dfc3ecd2fd7ce094587a25` |

The external comparison record is
[`tokenizer-comparison.md`](tokenizer-comparison.md). It compares token counts
and pieces only. The SentencePiece fixture is an official test artifact trained
on a small Japanese corpus, not a production-model tokenizer; that limitation
is part of the result.

## Text has several non-interchangeable units

**[EXTERNAL] Unicode.** A Unicode code point is an integer in the codespace
`U+0000..U+10FFFF`. A Unicode scalar value excludes the surrogate range used by
UTF-16. UTF-8 is a variable-length encoding form: one scalar value occupies
one to four bytes. A user-perceived grapheme can contain multiple scalar
values—for example, a base letter followed by a combining mark, or an emoji
sequence joined by U+200D ZERO WIDTH JOINER.

Therefore these counts need not agree:

```text
visible graphemes != Unicode scalar values != UTF-8 bytes != model tokens
```

For valid Rust `str` input, the string is already well-formed UTF-8. The
teaching tokenizer nevertheless accepts `&[u8]` because model vocabularies and
decoded token pieces are byte-oriented, and because the runtime needs an
explicit policy for arbitrary bytes. The toy byte-fallback vocabulary maps all
256 possible byte values, so any byte sequence can be encoded and decoded
losslessly as bytes. Interpreting those bytes as text is a later UTF-8 framing
operation.

**[EXTERNAL] UTF-8 conformance.** Unicode requires ill-formed sequences to be
treated as an error condition when interpreted as UTF-8; they must not be
silently interpreted as characters. RFC 3629 describes the one-to-four-octet
encoding and excludes overlong forms, surrogate code points, and values beyond
U+10FFFF. The Chapter 2 stream decoder therefore has three states:

1. complete valid buffer: emit it;
2. valid prefix followed by a possibly incomplete suffix: emit the prefix and
   retain the suffix;
3. definitively invalid sequence: preserve the buffer, return a typed error,
   and do not emit a new prefix or substitute replacement text.

At terminal, a non-empty incomplete suffix is an error. This policy preserves
raw bytes for diagnostics and never uses lossy replacement. Other production
protocols may deliberately choose U+FFFD replacement, but the choice must be
named and tested.

## Normalization changes the tokenization input

Unicode allows canonically equivalent sequences. NFC composes where possible;
NFD decomposes. NFKC and NFKD additionally apply compatibility mappings that
can change distinctions meaningful to some applications. UAX #15 defines
these forms and their stability rules.

Normalization is not an automatic precondition for every tokenizer. It is a
configured semantic transform. Two canonically equivalent strings may produce
different IDs under an identity-normalizing tokenizer, while a tokenizer that
applies NFC may collapse them before segmentation. Exact byte round trip is
possible only if no irreversible normalization, whitespace rewrite, unknown
replacement, special-token removal, or cleanup decoder changed the input.

**[EXTERNAL] Hugging Face Tokenizers.** Its documented pipeline separates
normalizer, pre-tokenizer, model, post-processor, and decoder. Only the model is
mandatory; the other components are configured. This separation is useful
because “BPE tokenizer” alone does not specify lowercasing, Unicode
normalization, whitespace boundaries, byte-level mapping, special insertion,
or decode cleanup.

**[EXTERNAL] SentencePiece.** The serialized `ModelProto` carries trainer,
normalizer, and vocabulary information. SentencePiece supports multiple model
types, including Unigram and BPE; therefore *SentencePiece* does not name one
segmentation algorithm. Its common normalizer may use NFKC-derived rules and a
metaspace marker for spaces. Configuration can collapse or strip whitespace.
Byte fallback is optional, and when enabled the model contains 256 byte pieces.

## Vocabulary and token identity

A **vocabulary** is a finite mapping between token IDs and token definitions.
An ID such as `42` has no portable meaning by itself. Its meaning depends on the
exact tokenizer artifact and configuration. Even if two vocabularies contain a
visually identical piece, their IDs can differ. A token is therefore a model
vocabulary identity, not necessarily a word, character, scalar value, byte, or
displayable string.

The Chapter 2 code uses a `TokenId(u32)` newtype. The toy vocabulary reserves:

- IDs `0..=255` for literal byte values;
- a small documented range for learned BPE merge results;
- a disjoint range for BOS, EOS, PAD, UNK, role, and end-of-turn controls.

An ordinary token decodes to bytes. A control token does not silently decode as
ordinary user text. Rendering special-token spellings is a diagnostic choice,
not the inverse of ordinary text encoding.

The 256-byte base makes `<unk>` unnecessary for the toy tokenizer's raw byte
domain, but an UNK identity is still present so the chapter can explain
artifacts that use it. In other tokenizers, unknown input may map to UNK unless
byte fallback or complete character coverage exists. UNK decoding does not
recover the lost source bytes.

## BPE from first principles

BPE training and BPE application are different operations. Training observes
a corpus and constructs an ordered merge list or equivalent ranks. Encoding
applies the fixed learned rules; it does not count pairs in the user's prompt
and learn new merges.

The Chapter 2 hand fixture begins with bytes for `lower`:

```text
initial:  l  o  w  e  r
rank 0:  (l,o)   -> lo
rank 1:  (lo,w)  -> low
rank 2:  (e,r)   -> er
rank 3:  (low,er)-> lower
```

At each encoding iteration, inspect adjacent pairs that have rules, select the
lowest rank, and merge one occurrence. For equal ranks, the toy contract uses
the leftmost occurrence as a deterministic tie-break. Repeat until no adjacent
pair has a rule. The merge order matters: applying a later rule before its
prerequisites can strand a different segmentation.

This tiny procedure teaches the invariant but is not an efficient production
implementation. A production encoder may use heaps, linked structures,
caches, pre-tokenization boundaries, or model-specific algorithms while
preserving the same fixed-rank result.

**[EXTERNAL] GPT-2.** OpenAI's reference `encoder.py` first maps UTF-8 bytes to
a reversible Unicode alphabet, applies a regex pre-tokenization, and then uses
ranked BPE merges. Decode reverses vocabulary pieces through the byte mapping
and UTF-8 decoding. The reversible byte mapping is why GPT-2-style byte-level
BPE can cover arbitrary input bytes without an ordinary UNK path; it should not
be confused with “one token per byte.” Merges frequently combine many mapped
bytes into one token.

**[EXTERNAL] Sennrich et al.** The subword NMT implementation applies learned
merge operations to smaller symbols. Its word-boundary conventions differ
from GPT-2's byte-level mapping. “Uses BPE” does not make two tokenizers
interchangeable.

## SentencePiece Unigram is not BPE

SentencePiece's Unigram model assigns scores to a vocabulary of candidate
pieces and finds a likely segmentation of the input, commonly with dynamic
programming. Training begins with a larger seed vocabulary and removes pieces
according to the model objective. BPE instead applies an ordered merge process.
Both yield subword IDs, but their learned artifacts and encoding algorithms are
different.

This distinction belongs in Chapter 2 because real model metadata may say
SentencePiece while the embedded model type says BPE or Unigram. An engine must
bind to the artifact's actual type instead of dispatching from the container
brand name.

## Special tokens are control identities

Common names describe roles, not universal IDs or universal behavior:

- **BOS** marks a sequence beginning when the model configuration requires it.
- **EOS** or another end-of-generation identity can terminate generation.
- **PAD** fills shape positions in systems that need padding; it is not
  necessarily a valid stop marker.
- **UNK** represents input the ordinary vocabulary cannot encode; information
  has already been lost.
- Role and end-of-turn identities structure chat transcripts for models trained
  with them.

Special tokens require a separate insertion surface. If ordinary user text
contains the literal bytes `<|assistant|>`, `encode_text` must encode those
bytes as ordinary data. Only a trusted template or explicit caller request may
insert the assistant control ID. Otherwise user text can cross a control-plane
boundary merely by spelling a marker.

llama.cpp exposes this distinction as `parse_special`: special/control text is
ordinary plaintext when false and may resolve to control IDs when true. Hermon
passes false for its raw prompt path and true after applying a chat template.
The Chapter 2 teaching API represents the distinction structurally rather than
asking callers to remember a Boolean at every call.

## Chat templates are serialized model input

A chat API accepts structured messages, but a causal language model consumes a
token sequence. A **chat template** serializes roles, content, separators,
turn endings, and possibly an assistant-generation prefix into the exact
sequence convention used during training.

**[EXTERNAL] Hugging Face.** Chat templates are stored with tokenizer/model
configuration and commonly expressed as Jinja. Official guidance warns that a
template may already insert required BOS/EOS tokens; adding them again during a
second tokenization step can duplicate control tokens. `add_generation_prompt`
may append the control sequence that begins an assistant turn, but its effect
is model-specific.

The teaching implementation does not embed Jinja. `TinyChatTemplate` accepts a
typed list of `(role, content)` messages and emits a sequence of two kinds of
segments: ordinary byte text and explicit special-token insertions. That is
enough to prove the ownership boundary:

```text
messages -> template segments -> ordinary encode + explicit specials -> IDs
```

The model identity object binds a model name, tokenizer identity, chat-template
identity, and special-token semantics. Swapping only the tokenizer or only the
template is a contract mismatch even if every component parses successfully.

## Raw completion and chat completion are different inputs

A raw completion input is encoded as the caller's ordinary text plus only the
special insertions explicitly required by that model surface. A chat
completion first serializes structured messages with the model's template.
Flattening `role: content` by intuition changes the token sequence, and hence
the model computation, before any learned weight is read.

The wrong-template experiment will render the same two messages with the tiny
contract and with a deliberately naive `role: content` flattening. It compares
bytes and IDs, not output quality: ENGINE-0 still uses a fake candidate source,
so it cannot truthfully demonstrate a model-quality change.

## Streaming decode is byte framing

The inverse vocabulary lookup for one ID returns a byte piece. That piece may
be empty, valid UTF-8, a prefix of a scalar value, a suffix that completes bytes
held from a prior ID, several scalar values, or definitively invalid when
combined with the pending prefix.

The output framer owns a bounded pending byte vector. UTF-8 requires at most
four bytes for one scalar, so after emitting every complete prefix, a valid
incomplete suffix is at most three bytes. The implementation checks this
invariant and exposes pending length in trace events.

Token events and text events are deliberately separate. The runtime can record
that token ID `x` was selected even when it cannot emit text until token ID `y`
completes the scalar. Time to first selected token and time to first emitted
text can therefore differ.

## Hermon trace at `472a44c`

Hermon `main`, local `HEAD`, and `origin/main` all resolved to
`472a44cdb511b2dae6c9569e59543db8f8350b25` after fetch. An unrelated untracked
`docs/core/.DS_Store` was left untouched.

### Current real-model owner

**[CURRENT]** The default batched route builds a message view in
`crates/hermon-runtime/src/batched.rs`, calls
`Model::apply_chat_template(..., true)`, falls back to a naive flattener only
when the model lacks an embedded template, then calls
`Context::tokenize(&prompt, true, true)`. The `true` flags request configured
special insertion and parsing for the template-formatted prompt.

`hermon-engine` and `hermon-runtime` reach the safe wrapper in
`hermon-llamacpp/src/linked.rs`. The C shim forwards to the pinned llama.cpp
`llama_tokenize`, `llama_token_to_piece`, and chat-template APIs. Hermon's
submodule is pinned to llama.cpp `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.
The upstream head observed during this inspection was `b81c99b...`; Hermon does
not claim to execute that newer head. “CURRENT” here means the pinned revision
owns Hermon's current path, not that the pin equals upstream HEAD.

### Streaming decode and buffer ownership

**[CURRENT]** Each active batched sequence owns `utf8_buf: Vec<u8>`. After
sampling, the worker asks llama.cpp for token bytes with special rendering
disabled, appends them, finds the longest prefix accepted by `str::from_utf8`,
and sends that prefix as a `StreamItem::Piece(String)`. This correctly
demonstrates why output pieces need not align one-to-one with tokens.

**[CURRENT caveat]** The code uses `Utf8Error::valid_up_to()` without branching
on `error_len()`. A definitively malformed suffix therefore remains buffered
instead of failing immediately. Finalization converts any leftover bytes with
`String::from_utf8_lossy`. The blocking engine convenience path has the same
lossy terminal flush. Chapter 2 records this behavior but does not modify
Hermon. ENGINE-0 chooses a stricter typed-error policy so the invariant is
executable and the tradeoff visible.

### Standalone tokenizer crate

**[LIBRARY]** `crates/hermon-tokenizer/src/lib.rs` defines `TokenizerKind`, a
`Tokenize` trait, and a prefix-cache skeleton. Its own status comment says the
current version ships interface skeletons and describes BPE, SentencePiece,
Tiktoken, and Hugging Face ingestion as future work. Repository search found no
current real-model route importing this crate. It must not be described as
Hermon's production tokenizer.

### Relevant test evidence

- `hermon-llamacpp/tests/batched_decode.rs` exercises real tokenization and
  concatenated token-byte decoding when an external model fixture is present.
- `hermon-runtime/tests/gguf_paged_differential.rs` applies a real embedded chat
  template, tokenizes through llama.cpp, and compares generated token IDs
  between the preview paged path and the pinned llama.cpp reference. It is
  ignored unless a real model path and linked CPU backend are supplied.
- No focused unit test was found for incomplete versus definitively malformed
  UTF-8 buffer behavior in the default batched streamer. The Chapter 2 teaching
  tests therefore should not be presented as verification of Hermon.

## ENGINE-0 Chapter 2 design

### Tokenizer contract

The public contract is byte-first and fallible:

```rust
pub trait Tokenizer {
    fn identity(&self) -> TokenizerIdentity;
    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError>;
    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError>;
    fn special_id(&self, special: SpecialToken) -> Option<TokenId>;
}
```

Ordinary encoding cannot synthesize special identities. A separate helper
encodes `TemplateSegment::Text` through `encode` and resolves
`TemplateSegment::Special` explicitly. Decode validates IDs and rejects
control identities on the ordinary-byte path.

### Reference and production-shaped implementations

`ByteTokenizer` is the independent clarity oracle: one byte maps to the token
ID with the same numeric value. `TinyBpeTokenizer` begins from the same byte
alphabet and applies a fixed, validated merge table. Both are standard-library
only. The BPE implementation is intentionally quadratic for small teaching
inputs; the chapter names that scaling limit rather than optimizing it early.

### Request and runtime evolution

`Request` owns encoded prompt IDs and the source byte count, not an opaque
`String`. `Model` still receives the request and runtime-owned generation
state. The fake `DemoModel` still returns the same hand-ranked candidates, now
using IDs from the tiny tokenizer contract. This is not ENGINE-1 and the scores
are still not logits.

The runtime additionally owns the output tokenizer/framer boundary:

```text
selected ordinary ID -> token bytes -> UTF-8 buffer -> zero or one text event
```

A token identity event remains observable separately from any valid text
event. EOS completes without ordinary byte emission. Decode/framing failure is
a failed terminal outcome and cannot also complete.

### Model contract object

A small `ModelContract` records model, tokenizer, and chat-template identities.
It validates that the concrete teaching components match those names before
preparing input. This is not a model loader. It exists to make an invariant
typed and testable: configuration pieces that jointly define model input must
travel together.

## Fixtures, experiments, and tests

Small text fixtures live under `code/mini-engine/fixtures/tokenizer/`; the
independent hand oracle lives under `code/reference/`. No model weights or
external tokenizer artifacts are committed.

Required correctness coverage:

- empty bytes; ASCII; multibyte UTF-8; Chinese; emoji; combining sequences;
- leading/trailing spaces, tabs, and newlines;
- repeated input and deterministic encoding;
- invalid token IDs and control-token decode attempts;
- ordinary marker-like text versus explicit special insertion;
- byte-tokenizer and BPE round trips for arbitrary bytes;
- BPE merge rank, leftmost tie, unmergeable input, repeats, and empty input;
- partial UTF-8 across token boundaries, several complete scalars in one
  piece, definite malformed bytes, and incomplete terminal suffix;
- correct versus wrong chat-template ID sequences;
- lifecycle success, cancellation, model failure, decode failure, and exactly
  one terminal event.

The real-tokenizer comparison uses five fixed inputs: English, code, Chinese,
emoji, and whitespace/newline-sensitive text. It records package/model
identity, byte count, scalar count, IDs, byte/piece renderings, token count, and
SentencePiece decoded output. Counts are systems inputs for context budgets,
prefill work, cache geometry, billing/accounting, and stop limits. They are not
a language-quality ranking.

## Planned diagrams

1. Text, scalar values, UTF-8 bytes, and token IDs.
2. Fixed-rank BPE merge process.
3. Structured chat messages through template segments to IDs.
4. Generated token identity to byte piece to text stream.
5. Partial UTF-8 scalar spanning token boundaries.
6. Model/tokenizer/template/special-token contract.
7. Input/output ownership through ENGINE-0.

All canonical diagrams remain plain ASCII and at most 100 columns.

## Chapter contract metadata

- **Purpose:** make the text/model boundary exact and executable.
- **Prerequisites:** Chapter 1's request, stream, byte, token, and owner map.
- **Key question:** which configured transforms determine the IDs seen by the
  model and the bytes seen by the client?
- **Mathematics:** sequence length `n`, vocabulary size `V`, deterministic
  mappings, BPE merge rank, byte/scalar/token counts, and bounded UTF-8 suffix.
- **Systems concepts:** configuration binding, trusted control insertion,
  byte/text boundaries, streaming framing, typed errors, and context budgets.
- **Hardware concepts:** token count controls later work and memory demand;
  cache locality is named only, with no tokenizer performance claim.
- **Implementation:** `TokenId`, byte oracle, toy byte-fallback BPE, explicit
  specials, tiny chat template, model contract, UTF-8 stream decoder, CLI
  probes, fixtures, and lifecycle integration.
- **Hermon connection:** CURRENT llama.cpp ownership and buffering; LIBRARY
  status for `hermon-tokenizer`; lossy terminal-flush caveat.
- **External connection:** Unicode, UTF-8, BPE, SentencePiece, Hugging Face
  tokenizer components/templates, GPT-2/tiktoken, and pinned llama.cpp APIs.
- **Deliverable:** Chapter 2 prose, updated ENGINE-0, Labs 2–4, seven diagrams,
  independent fixtures/oracles, comparison record, glossary, and status.
- **Next assumption:** Chapter 3 receives stable token IDs and replaces the fake
  candidate table with genuine embedding/projection/logit computation.

## Review questions

### Unicode/text reviewer

- Does every use of character distinguish grapheme, scalar value, code point,
  and byte where the distinction matters?
- Are normalization and malformed UTF-8 policies explicit rather than implied?
- Does streaming emit only well-formed UTF-8 and bound pending bytes?

### Tokenizer engineer

- Are training and encoding separated?
- Are BPE rank/tie rules deterministic and independently tested?
- Are SentencePiece BPE and Unigram kept distinct?
- Can ordinary user text ever inject a control identity?

### Inference engineer

- Are model, tokenizer, template, and special-token semantics bound together?
- Does token count appear as a context/work/accounting variable without an
  unsupported performance conclusion?
- Does fake model selection remain visibly fake?

### Beginner reader

- Can the reader hand-tokenize `lower` before reading production comparisons?
- Do examples show why a token is not a word or Unicode character?
- Is the difference between raw completion and chat serialization concrete?

## Open questions and later work

1. The exact real model artifact for Parts III and XI remains unchosen; its
   tokenizer and chat template must be pinned with the model, not borrowed from
   this toy fixture.
2. Chapter 3 must replace `DemoModel` rather than wrapping its integer scores.
3. Chapter 4 owns genuine logits, greedy/stochastic sampling, autoregressive
   feedback, and complete stop policy.
4. Later production chapters must choose protocol behavior for malformed
   model-output bytes and audit Hermon's lossy terminal flush end to end.
5. Efficient production BPE data structures, tokenizer concurrency, and
   allocation benchmarks are intentionally deferred until a real workload and
   artifact are selected.

## Completion review record — 2026-09-02

The Unicode/text, tokenizer-engineering, inference-engineering, beginner, and
cross-artifact passes were performed separately after research, prose, code,
fixtures, experiments, labs, and diagrams existed.

### Unicode and text pass

- Checked each important count as grapheme/scalar/byte/token rather than using
  *character* ambiguously.
- Reconciled the strict implementation policy with the prose: incomplete
  suffixes can retain at most three bytes; a definitely invalid append fails
  atomically without emitting a new prefix or using lossy replacement.
- Covered composed/decomposed text, Chinese, emoji/ZWJ, whitespace, arbitrary
  bytes, definite malformed input, and incomplete terminal input.

### Tokenizer-engineering pass

- Rechecked BPE rule construction, rank order, leftmost occurrence tie,
  unmergeable input, byte fallback, deterministic round trip, invalid IDs, and
  merge/special ID collisions against independent fixtures.
- Kept BPE training separate from encoding and SentencePiece separate from its
  BPE/Unigram model types.
- Verified ordinary marker-like bytes cannot synthesize control identities;
  only typed template segments insert specials.

### Inference-engineering pass

- Re-read the Hermon chat-template, tokenize, token-to-piece, buffer, finalizer,
  shim, pinned llama.cpp API, standalone tokenizer crate, and relevant tests.
- Preserved CURRENT for the pinned llama.cpp real-model path, LIBRARY for
  `hermon-tokenizer`, and the lossy terminal-flush caveat without modifying the
  external repository.
- Confirmed ENGINE-0 still uses fake hand-ranked candidates and contains no
  embeddings, learned weights, projection, logits, or Chapter 3 computation.

### Beginner and editorial pass

- The chapter moves from visible text to units, pipeline, hand BPE, special
  authority, chat serialization, streaming bytes, executable design, proof,
  real comparison, systems consequences, and Hermon evidence.
- Hand fixtures precede production source, and every real comparison states its
  artifact/corpus limitation. No token-count result is presented as quality or
  speed evidence.
- CHECK/BUILD/BREAK/EXTEND work is split across Labs 2–4 with explicit cleanup.

### Cross-link and executable pass

- Final manuscript length: 7,306 words.
- Seven canonical Chapter 2 diagrams render at no more than 100 columns.
- The Rust workspace has no external dependencies and passes format, check, 37
  tests, and Clippy with warnings denied.
- The Python comparison probe reproduced all five recorded count pairs with
  `tiktoken==0.14.0`, `sentencepiece==0.2.2`, and the pinned model hash.
- Structure, Markdown links, credential guard, large-file gate, diagram width,
  and `git diff --check` pass; glossary, terminology, part index, book table of
  contents, labs, README, Hermon inventory, status, and Chapter 3 handoff agree.
