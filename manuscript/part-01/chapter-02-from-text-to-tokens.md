# Chapter 2 — From Text to Tokens

## The model does not receive your sentence

Suppose two people send what appears to be the same request:

```text
Explain why token boundaries matter.
```

One client sends a raw completion string. Another sends a chat message with the
role `user`. One runtime normalizes the text; another preserves every byte. One
uses the vocabulary packaged with the model; another accidentally loads a
tokenizer with the same family name but a different revision. All four paths
produce valid arrays of integers. Only some of those arrays are the input the
model was trained to interpret.

The model does not receive a sentence, a chat bubble, or a row of visible
characters. It receives **token IDs**. Each ID refers to one entry in one
finite, configured **vocabulary**. The conversion from text to those IDs is
**tokenization**, and it can include Unicode normalization, whitespace rules,
pre-tokenization, a segmentation algorithm, byte fallback, and explicit
insertion of control identities. On the output side, generated IDs become byte
pieces. Those pieces may cut through a UTF-8 character, so a runtime sometimes
has to wait for several tokens before it can emit valid text.

This boundary is part of model semantics. If it is wrong, the numerical model
can execute every matrix multiplication correctly and still receive the wrong
sequence.

Chapter 1 made request ownership real while treating the prompt as opaque.
This chapter opens that first black box. We will distinguish text, Unicode
scalar values, bytes, and token IDs; derive byte-pair encoding (BPE) by hand;
separate BPE from SentencePiece's Unigram algorithm; make special-token
insertion explicit; serialize a tiny chat correctly; and stream decoded bytes
without producing malformed UTF-8. Then we will evolve ENGINE-0 so tokenization
is real while its model remains deliberately fake.

Chapter 3 will replace that fake model with numerical inference. We do not
cross that boundary here.

## Four sequences can describe one visible input

Begin with a small piece of text:

```text
世🚀
```

It is tempting to count two “characters” and move on. An inference engine needs
more precise units. The
[text/Unicode/bytes/tokens diagram](../../diagrams/tokenizer/text-unicode-bytes-tokens.txt)
shows four:

```text
visible text       Unicode scalars       UTF-8 bytes           token IDs
"世🚀"         ->  U+4E16 U+1F680   ->  E4 B8 96 F0 9F 9A 80 -> [x, y, z]
                     2 scalars                7 bytes            model-specific
```

The visible text is what a renderer shows. A **Unicode code point** is an
integer position in the Unicode codespace. A **Unicode scalar value** is a code
point excluding the surrogate range reserved for UTF-16. UTF-8 encodes each
scalar value as one to four bytes. A tokenizer then maps a configured transform
of those bytes or text units to vocabulary identifiers.

None of these boundaries has to match another.

The letter `é` can be one scalar value, U+00E9. It can also be represented by
the canonically equivalent sequence U+0065 LATIN SMALL LETTER E followed by
U+0301 COMBINING ACUTE ACCENT. A user may perceive one grapheme in both cases,
but the second representation has two scalar values and three UTF-8 bytes.
An emoji such as `👩🏽‍💻` contains multiple scalar values: a person, a skin-tone
modifier, a zero-width joiner, and a laptop. A renderer may present the sequence
as one grapheme. A tokenizer may split it into many IDs, sometimes in the
middle of one scalar's UTF-8 encoding.

Use *character* only when its meaning is genuinely unambiguous. At system
boundaries, name the unit: grapheme, scalar value, UTF-8 byte, or token ID.

> **FIRST PRINCIPLE**
> A token is an identity in a particular model vocabulary. It is not
> necessarily a word, character, Unicode scalar value, byte, or output string.

This distinction immediately explains why token-count rules cannot be derived
from word count. Punctuation may have its own ID or join a neighboring piece.
A leading space may be part of a token. A common word may be one ID while a
rare spelling becomes many. Chinese text has no obligation to follow
space-delimited English boundaries. An emoji can require several byte-fragment
tokens.

## UTF-8 is a byte encoding, not tokenization

Unicode assigns scalar values. UTF-8 specifies how to encode those values as
bytes. Its leading byte determines the expected sequence length:

```text
0xxxxxxx                            one byte
110xxxxx 10xxxxxx                  two bytes
1110xxxx 10xxxxxx 10xxxxxx         three bytes
11110xxx 10xxxxxx 10xxxxxx 10xxxxxx four bytes
```

Those bit shapes are not the whole validity rule. Overlong encodings are
forbidden. UTF-16 surrogate code points are not scalar values and cannot be
encoded as valid UTF-8. Values above U+10FFFF are invalid. Continuation bytes
must appear in the permitted positions. The exact well-formed ranges are
specified by the Unicode Standard and RFC 3629.

Rust's `str` guarantees valid UTF-8. That is useful for application text, but a
tokenizer/runtime boundary still benefits from byte-oriented interfaces for
three reasons.

First, byte-level vocabularies define pieces in terms of bytes. Second,
generated token pieces may not be valid UTF-8 independently. Third, a low-level
engine needs an explicit policy for arbitrary or malformed bytes rather than
letting a string conversion choose silently.

Our Chapter 2 tokenizer therefore accepts `&[u8]` and decodes token IDs to
`&[u8]`. A separate stream framer interprets accumulated output as UTF-8. This
does not claim that every tokenizer API should expose raw bytes to every user.
It keeps representation and interpretation separate at the teaching boundary.

### Malformed bytes require policy

When bytes purport to be UTF-8, Unicode conformance treats ill-formed sequences
as an error condition; they cannot be interpreted as characters. A product may
signal an error, substitute U+FFFD REPLACEMENT CHARACTER according to a
documented maximal-subpart policy, or expose a binary channel instead of a text
channel. What it must not do is accidentally mix policies.

ENGINE-0 uses a strict policy:

- when the buffer is valid or merely incomplete, emit its longest complete
  valid prefix and retain a suffix that later bytes could complete;
- on a definitively invalid append, preserve the buffer and fail without
  emitting a new prefix from that append;
- fail a would-be successful terminal if an incomplete suffix remains;
- never perform lossy replacement.

This policy is especially useful for teaching because failure is observable
and raw pending bytes remain available for diagnostics. Hermon's current
production path chooses a different terminal behavior, which we will inspect
later.

## A tokenizer is a configured pipeline

“The tokenizer uses BPE” is not a complete tokenizer specification. A useful
pipeline model is:

```text
input
  -> normalization
  -> pre-tokenization / boundary rules
  -> vocabulary model (BPE, Unigram, WordPiece, ...)
  -> special-token post-processing
  -> token IDs
```

Decoding has its own configured inverse or cleanup steps. Hugging Face
Tokenizers exposes these responsibilities as normalizer, pre-tokenizer, model,
post-processor, and decoder components. Only the model component is mandatory.
That does not make the others unimportant; it means the artifact must say
which ones exist.

### Normalization

Unicode normalization can replace one valid scalar sequence with an equivalent
or compatibility-related sequence. NFC tends to compose; NFD decomposes. NFKC
and NFKD additionally apply compatibility mappings. A tokenizer can instead
choose identity normalization and preserve the original sequence.

Other configured normalization may lowercase, strip accents, collapse
whitespace, add a dummy leading space, or replace spaces with a marker. These
are semantic transforms before vocabulary lookup. They can change token IDs
and destroy exact byte round trip.

Consider the composed and decomposed forms of `é`:

```text
NFC-like input:  C3 A9
NFD-like input:  65 CC 81
```

An identity tokenizer sees different bytes. An NFC-normalizing tokenizer may
make them equal before segmentation. Neither behavior is universally correct.
The model and tokenizer configuration determine the contract.

### Pre-tokenization

A pre-tokenizer introduces boundaries or transforms a stream into regions the
vocabulary model will process. It might separate punctuation, isolate
whitespace, use a regular expression, or map bytes to a reversible visible
alphabet. A BPE model usually does not merge across a boundary it never sees.

This is one reason two byte-level BPE tokenizers can segment the same text
differently even if both cover all bytes. Their regular expressions, byte
mappings, vocabularies, and merge ranks can differ.

### Vocabulary model

The model component maps a region into vocabulary pieces and IDs. BPE applies
ranked merge rules. A Unigram model searches segmentations using learned piece
scores. WordPiece uses another greedy vocabulary procedure. A word-level model
may require a pre-tokenizer and map unknown words to UNK.

The algorithm name still does not give an ID meaning. The exact vocabulary
artifact does.

### Post-processing and decoding

A post-processor can add BOS, EOS, separator, or classification tokens after
ordinary encoding. A decoder may reverse a byte mapping, turn a metaspace
symbol back into whitespace, remove continuation markers, or perform cleanup.
Exact round trip depends on the whole chain.

> **PROVE IT**
> For every round-trip claim, state the domain and configuration. “All byte
> sequences round-trip through the toy byte-fallback BPE with no normalization
> or special removal” is testable. “Tokenization is reversible” is too broad.

## Vocabulary identity is model identity

A vocabulary is a finite table. One direction associates IDs with token
definitions; the encoder uses enough additional structure to choose a sequence
of those IDs for input. Let $V_{\mathrm{vocab}}$ be vocabulary size and $N$ be
the number of token IDs in one encoded input. Then every ordinary ID $x_i$
must satisfy

$$
x_i\in\{0,1,\ldots,V_{\mathrm{vocab}}-1\},
\qquad 0\le i<N.
$$

This is the half-open code interval `0..V_vocab` under the vocabulary's
indexing convention.
But `id = 42` is not portable. Vocabulary A may define 42 as bytes for a common
word fragment. Vocabulary B may assign 42 to punctuation, a byte fallback
piece, or a control identity.

The vocabulary also does not guarantee that every ID is ordinary text. Some
entries are controls. Some are unused. Some represent unknown input. Some byte
pieces do not form valid text alone.

This is why model loading eventually has to validate more than tensor shapes.
The artifact's weights learned relationships among positions carrying these
exact identities. Swapping a tokenizer changes the meaning of the embedding
row selected for each input position. We will build embeddings in Chapter 3;
for now, retain the contract: stable IDs must reach that future lookup.

ENGINE-0 introduces a small `TokenId(u32)` newtype instead of passing bare
integers everywhere. The type cannot prove which vocabulary an ID belongs to,
but it prevents accidental arithmetic and makes interfaces say when they carry
identity rather than text.

## Build BPE from first principles

Byte-pair encoding appears in several tokenizer families, but the essential
encoding operation can be understood with five letters. Our toy vocabulary
starts with all 256 byte values, then adds learned merge results.

The fixed merge table for `lower` is:

```text
rank 0: l   + o   -> lo
rank 1: lo  + w   -> low
rank 2: e   + r   -> er
rank 3: low + er  -> lower
```

The [BPE merge diagram](../../diagrams/tokenizer/bpe-merge-process.txt) applies
it:

```text
bytes:        l | o | w | e | r
rank 0:      [l + o] -> lo       => lo | w | e | r
rank 1:     [lo + w] -> low      => low | e | r
rank 2:                    [e+r] => low | er
rank 3:             [low + er]  => lower
```

The encoder starts from base symbols. It inspects adjacent pairs that have
merge rules, chooses the available rule with the lowest rank, replaces that
pair with the result symbol, and repeats until no rule applies.

Two details matter.

First, these rules are fixed before the prompt arrives. BPE *training* learns a
merge vocabulary from a corpus. BPE *encoding* applies the learned artifact.
Counting pairs in a user's prompt and inventing a merge would change the
tokenizer during inference.

Second, order is semantic. The result symbol of a later merge may require an
earlier result. Greedily choosing a visually large piece without respecting
ranks can produce a different segmentation. If the same rank could apply at
multiple positions, the toy contract chooses the leftmost occurrence, then
scans again. Production formats commonly assign unique ranks to merge pairs,
but occurrence order still needs deterministic behavior.

### Hand cases

For `lolo`, rank 0 applies twice:

```text
l | o | l | o
[l+o] | l | o -> lo | l | o
lo | [l+o]     -> lo | lo
```

The result IDs are `[256, 256]` in our fixture.

For `xyz`, no merge rule applies. The result remains three base byte IDs:
`[120, 121, 122]`.

For empty input, the result is an empty ID vector. That is a valid tokenizer
result. The generation request layer may independently reject an empty prompt.
Keeping those decisions separate prevents a tokenizer from owning API policy.

### A clarity-first implementation

`TinyBpeTokenizer` stores the fixed rules and decoded bytes for every merge
result. Construction rejects duplicate pairs, duplicate ranks, reserved result
IDs, duplicate result IDs, and references to symbols not defined by an earlier
rule.

Encoding begins with one `TokenId` per byte. On every iteration it scans all
adjacent pairs, remembers the lowest-rank rule, merges one occurrence, and
repeats. This is deliberately easy to audit and can be quadratic in input
length. The limitation is part of the lesson. A production tokenizer may use
priority queues, linked symbol structures, caching, vectorized search, or
parallel pre-tokenization, but it must preserve the configured segmentation.

> **BUILD IT**
> Complete [Lab 2 — Tokenize by
> Hand](../../labs/lab-02-tokenize-by-hand.md) before changing the merge table.
> Predict the states for `lower`, `lolo`, and `xyz`, then compare with the CLI.

## Byte fallback is coverage, not one-token-per-byte

Our toy starts with every byte, so any input byte sequence can be represented.
Known pairs can still merge into larger pieces. `blue`, for example, follows
`b+l -> bl`, `bl+u -> blu`, and `blu+e -> blue`, becoming one ID in the toy
vocabulary. Unknown byte patterns remain base IDs.

This gives complete byte coverage without an ordinary unknown path. It does
not mean every input produces one token per byte. It also does not mean the
output is valid UTF-8. The byte sequence `C3 28` can round-trip exactly through
the tokenizer while remaining malformed as UTF-8.

SentencePiece can optionally include 256 byte fallback pieces. GPT-2-style
byte-level BPE uses a reversible byte-to-Unicode alphabet before applying BPE.
These are related coverage strategies with different representations. Neither
phrase is sufficient to identify a tokenizer.

OpenAI's GPT-2 reference encoder makes the layers visible. It encodes input as
UTF-8 bytes, maps all bytes to a reversible Unicode alphabet, splits with a
regular expression, applies ranked BPE, and maps pieces to IDs. Decode performs
the reverse vocabulary and byte mapping before UTF-8 decoding. A token can
therefore represent several original bytes, and its byte fragment can be
invalid UTF-8 alone.

## SentencePiece names a toolkit, not one algorithm

SentencePiece trains directly from raw sentences and packages vocabulary,
trainer settings, normalization, and other behavior in its model artifact. It
supports several model types. The two important ones here are BPE and Unigram.

SentencePiece BPE applies a merge-based model. Its surrounding behavior may
use a dummy prefix and a metaspace marker such as `▁` to represent word
boundaries without assuming that spaces should disappear.

The SentencePiece Unigram model is different. It has a vocabulary of candidate
pieces with learned scores. Encoding searches for a high-probability
segmentation, commonly by dynamic programming. Training starts with a larger
candidate set and prunes pieces according to the objective. There is no fixed
ranked merge trace equivalent to our `lower` example.

Calling both “SentencePiece tokenization” hides the algorithm decision. A real
engine must inspect the model type stored in the artifact. It cannot dispatch
to BPE merely because a file was produced by SentencePiece.

### Unknown tokens and byte fallback

SentencePiece model configuration includes special pieces such as UNK, BOS,
EOS, and optionally PAD. Character coverage can leave rare input outside the
ordinary alphabet. Without byte fallback, such input can map to UNK. That
mapping loses information: decoding UNK cannot reconstruct the original bytes.

With byte fallback enabled, out-of-vocabulary scalars can decompose into UTF-8
byte pieces. This avoids ordinary unknown loss at the segmentation stage. It
does not reverse earlier normalization. If the normalizer collapsed two spaces
to one before fallback, the second space is already gone.

## When does decode(encode(x)) equal x?

Exact round trip is conditional. It can hold when:

- the input domain is covered without UNK loss;
- normalization is identity or otherwise preserves the exact bytes in scope;
- pre-tokenization uses a reversible representation;
- decoding reverses every representation transform;
- no special tokens are inserted, removed, or rendered differently;
- no cleanup step changes spaces or punctuation.

It can fail when any of those conditions fails.

Our toy BPE has identity normalization, a complete byte alphabet, reversible
merges, and no implicit special insertion. Therefore, over the domain
$\mathcal{B}^{*}$ of finite byte strings:

$$
\operatorname{decode}(\operatorname{encode}(\mathbf{x}))=\mathbf{x},
\qquad \mathbf{x}\in\mathcal{B}^{*}.
$$

The equality is over bytes. If `x` is malformed UTF-8, the result is still the
same malformed byte vector and cannot enter the text stream successfully.

The real SentencePiece comparison later in this chapter demonstrates a
different configured contract. Its byte fallback covers the normalized input,
but its normalizer collapses or removes some whitespace. The decoded surface
is not the original source string for that fixture.

## Special tokens are control identities

The canonical [special-token trust boundary](../../diagrams/tokenizer/special-token-trust-boundary.txt)
separates untrusted surface text from template-authorized control identities.

Model vocabularies commonly reserve identities with roles such as:

- **BOS**: beginning of sequence;
- **EOS**: end of sequence or one configured end marker;
- **PAD**: padding used to fill a shape or batch convention;
- **UNK**: unknown input that could not be represented ordinarily;
- role markers: system, user, assistant, tool, or model-specific roles;
- end-of-turn or separator markers.

These names do not define universal IDs. BOS might be automatically added for
one model and forbidden for another. EOS may differ from an end-of-turn marker.
PAD must not automatically be treated as EOS. Some models have several
end-of-generation identities. UNK can be mandatory even when a modern model
rarely reaches it.

Special tokens also cross a trust boundary. Imagine a vocabulary with a
diagnostic spelling `<|assistant|>`. If the ordinary encoder automatically
recognizes that byte string anywhere, user content can inject the assistant
control identity. The model then sees a role transition that the structured
request did not authorize.

ENGINE-0 prevents that structurally. `Tokenizer::encode(&[u8])` encodes
ordinary data only. It has no flag that turns marker-looking text into control
IDs. Trusted template output is a sequence of typed segments:

```rust
pub enum TemplateSegment {
    Text(Vec<u8>),
    Special(SpecialToken),
}
```

`Text` goes through ordinary encoding. `Special` goes through an explicit
special-ID lookup. The string `<|assistant|>` inside `Text` remains its literal
bytes. This separation makes the authority visible in types and tests.

The toy assigns ordinary byte IDs `0..=255`, merge IDs beginning at 256, and
special IDs beginning at 1000. BOS, EOS, PAD, UNK, SYSTEM, USER, ASSISTANT, and
END_TURN occupy distinct IDs. An ordinary decode request for one of those
control IDs returns a typed error instead of pretending the diagnostic name is
user text.

> **FIRST PRINCIPLE**
> Ordinary text encoding and special-token insertion are different operations
> with different authority.

## A chat template is part of model input

A chat API presents messages as records:

```text
role=system, content="Be concise."
role=user,   content="Why does this split?"
```

A causal language model still receives one ID sequence. A **chat template**
serializes roles, content, separators, end-of-turn markers, optional tool data,
and the beginning of the generation turn according to the convention used
during training.

The [chat-template diagram](../../diagrams/tokenizer/chat-template-pipeline.txt)
shows the boundary:

```text
[messages] --render--> [Text | Special segments] --encode/insert--> [token IDs]
```

Our `TinyChatTemplate` is intentionally smaller than production template
languages:

```text
BOS
SYSTEM encode(system content) END_TURN
USER   encode(user content)   END_TURN
ASSISTANT                       # when generation should begin
```

The implementation accepts typed roles, not arbitrary strings. It returns
typed segments, not a marker-filled string that must later be reparsed. That is
enough to demonstrate the semantic owner without embedding Jinja in the
teaching engine.

### The wrong template can be valid text

A common fallback is:

```text
system: Be concise.
user: Why does this split?
```

Those bytes are valid UTF-8. The tokenizer can encode them. The request can
fit the context. The runtime can execute without error. Yet the IDs contain no
SYSTEM, USER, END_TURN, or ASSISTANT controls from the tiny model contract.
Validity at the string or array layer does not establish semantic correctness.

ENGINE-0 cannot measure the effect on answer quality because its model still
returns a fixed candidate table. [Lab 4](../../labs/lab-04-use-the-wrong-chat-template.md)
therefore compares exact bytes and IDs, not generated quality. A real-model
quality claim would require a pinned model, correct baseline, workload, and
evaluation design.

### Raw completion is not chat completion

A raw completion surface takes ordinary input intended to continue directly.
A chat completion surface first serializes structured messages. Even when both
contain the same visible user sentence, their model inputs may differ by BOS,
role headers, turn terminators, assistant prefixes, and whitespace.

Do not automatically apply a chat template to raw input. Do not flatten chat
by intuition. The API surface selects a preparation contract.

### Avoid duplicate special insertion

Hugging Face's official chat-template guidance notes an important failure
mode. A template may already include BOS, EOS, or other controls. If a caller
renders the template to text and then tokenizes with automatic special
addition enabled, it can duplicate them. Tokenizing through the template-aware
surface is often safer because the owner knows which controls are already
present.

`add_generation_prompt` is also model-specific. For templates that use an
assistant-turn opener, it appends the sequence that tells the model an
assistant reply comes next. Some templates do not need such a prefix. The flag
does not define a universal string.

## Bind model, tokenizer, template, and special semantics

A directory containing `model.gguf`, `tokenizer.json`, and a template file is
not correct merely because every file parses. The components must belong to the
same model contract.

The [contract diagram](../../diagrams/tokenizer/model-tokenizer-template-contract.txt)
makes the dependency visible:

```text
        (tokenizer + revision) ◀──▶ (special IDs) ◀──▶ (chat template)
                    ╲                    │                    ╱
                     └──────────── exact identities ────────┘
                                          │
                                          ▼
                                  (running model weights)
```

ENGINE-0 introduces a small `ModelContract` that records:

- model identity;
- tokenizer name and revision;
- chat-template identity;
- the tokenizer-owned special-ID mapping.

Before preparing chat input, it compares the concrete tokenizer and template
identities with the contract. This is not a full artifact loader. It is a
conceptual object that makes one failure explicit: swapping only one semantic
component is not a supported configuration.

Real systems need stronger provenance. A model repository revision, hashes,
embedded metadata, tokenizer assets, template revision, and runtime support
matrix may all participate. Part III will inspect model containers and
metadata. The invariant begins here.

## Generated tokens become bytes before they become text

On output, a model selects a token ID. The tokenizer vocabulary maps that ID to
a byte piece. Only the concatenation of pieces is expected to represent output
text.

The [token-to-stream diagram](../../diagrams/tokenizer/token-to-byte-stream.txt)
separates events and state:

```text
model selects ID ──▶ vocabulary lookup ──▶ byte piece ──▶ [UTF-8 buffer]
       │                      │                  │               │
       ▼                      ▼                  ▼               ├──▶ valid text event
 token event          typed ID error       may be partial       └──▶ wait / fail
```

Consider `世`, whose UTF-8 bytes are `E4 B8 96`. Suppose one generated ID maps
to `E4 B8` and the next maps to `96`.

The first selection is a real token event. It advances generation history and
token accounting. It cannot produce a valid text event yet. The buffer owns two
pending bytes. The next token completes the scalar, after which the runtime can
emit `世` and clear the buffer.

The [partial-boundary diagram](../../diagrams/tokenizer/utf8-partial-token-boundary.txt)
shows the timeline:

```text
token x bytes: E4 B8        [pending E4 B8]        text: (nothing)
token y bytes:       96  -> [pending E4 B8 96] -> validate -> text: "世"
```

### The buffer has a small semantic bound

After every complete valid prefix is emitted, a valid incomplete UTF-8 suffix
can contain at most three bytes: the beginning of a four-byte scalar. That does
not bound a network output queue or the number of model tokens. It bounds this
specific framing state.

`Utf8StreamDecoder::push` appends one piece and asks Rust's UTF-8 validator for
the status:

- complete valid buffer: emit it and clear;
- error with no definite error length: emit the valid prefix, retain the
  incomplete suffix;
- error with a definite error length: return `InvalidSequence` with the bytes.

`finish` succeeds only when no suffix remains. It never calls
`String::from_utf8_lossy`.

### Token latency and text latency can differ

Chapter 1 defined time to first token at an observable output boundary. Now we
can name two lower-level events:

```text
first selected token ready
first valid text piece emitted
```

They often occur close together for ASCII-heavy output. They can differ when
early token pieces are incomplete UTF-8 or non-rendered controls. ENGINE-0
records both `time_to_first_token` and `time_to_first_text`. These trace values
remain instructional, not benchmarks.

> **BUILD IT**
> Complete [Lab 3 — Stream UTF-8 Across Token
> Boundaries](../../labs/lab-03-stream-utf8-across-tokens.md). Inject a partial
> scalar, a definite malformed sequence, and an incomplete terminal suffix.

## Evolve ENGINE-0 without building a second runtime

The Chapter 1 lifecycle remains:

```text
validate -> admit -> execute -> select -> stream -> one terminal outcome
```

The input and output boundaries become real:

```text
input bytes -> tokenizer.encode -> Request.input_tokens
                                  |
                                  v
                         fake candidate model
                                  |
generated ID -> tokenizer.decode_token -> UTF-8 framer -> text stream
```

The complete ownership map is stored in
[`engine0-token-ownership.txt`](../../diagrams/tokenizer/engine0-token-ownership.txt).

`Request` no longer owns an opaque prompt `String`. It owns:

```rust
pub struct Request {
    pub id: RequestId,
    pub input_tokens: Vec<TokenId>,
    pub input_bytes: usize,
    pub max_new_tokens: usize,
}
```

`input_bytes` exists for trace/accounting context. The model consumes IDs.
Whitespace-only input is valid data now; an empty encoded sequence is rejected
by request validation. The tokenizer itself can still encode empty bytes to an
empty vector because that is the mathematically correct transform.

`Token` now contains identity and kind, not a cached `String`. This removes the
Chapter 1 defect in which the fake model effectively owned detokenized text.
The tokenizer owns ID-to-byte meaning.

`Runtime<M, S, T>` owns a model, selector, and output tokenizer. After an
ordinary token is selected, it:

1. appends the identity to runtime-owned generation state;
2. emits the token identity event;
3. looks up the byte piece;
4. pushes the bytes into the request-local UTF-8 framer;
5. emits a text event only when a valid non-empty prefix exists;
6. continues or reaches one terminal outcome.

EOS remains a control token. It is selected and traced but not decoded as
ordinary output. Before successful EOS or maximum-token completion, the
runtime verifies that the UTF-8 buffer is empty. A pending incomplete suffix
turns the attempted success into `Failed(Utf8Stream(...))`. Cancellation and
an earlier model failure retain their own terminal causes; pending bytes are
not emitted after a non-success terminal.

### The fake model is still fake

`DemoModel` returns the same hand-computable candidate ordering as Chapter 1:

```text
step 0: blue=9, green=4, EOS=1
step 1: EOS=10, blue=1
```

The names now correspond to genuine vocabulary IDs whose byte pieces are
`blue` and `green`. The integer candidate scores are still not logits. No
embedding lookup, hidden vector, learned parameter, output projection, or
vocabulary-scale numerical result exists.

This distinction protects Chapter 3. We have made model input and output
representation real without pretending the source of candidate scores is real.

### CLI probes

The executable keeps generation and adds two narrow probes:

```sh
cd code/mini-engine
cargo run -p engine0 -- tokenize lower
cargo run -p engine0 -- decode 259
cargo run -p engine0 -- --trace 'What color is the sky?'
```

`tokenize` prints tokenizer identity, byte/scalar/token counts, IDs, and piece
bytes. `decode` accepts numeric ordinary IDs, prints each byte piece, and runs
the strict UTF-8 framer. The generation trace adds `InputEncoded`,
`TokenDecoded`, `Utf8Buffered`, and `TextEmitted` stages to the Chapter 1
lifecycle events.

The CLI is a teaching surface, not a standard tokenizer file format or
production protocol.

## Prove tokenization independently

The implementation is not its own oracle. Chapter 2 has three small committed
fixture groups under `code/mini-engine/fixtures/tokenizer/` and a prose oracle
under `code/reference/`:

- the BPE rank table and hand encodings;
- byte fragments for valid, malformed, and incomplete UTF-8;
- the tiny chat-template control order.

The byte tokenizer is also an independent executable oracle for raw byte
round trips. It maps each byte directly to the numerically equal ID and does
not share the BPE merge algorithm.

The test suite covers four layers.

### Tokenizer semantics

- empty, ASCII, Chinese, emoji, combining, whitespace, repeated, and arbitrary
  malformed byte inputs;
- deterministic encoding and exact byte round trip for the toy BPE;
- merge prerequisites, rank order, leftmost repeat behavior, and unmergeable
  inputs;
- invalid IDs and invalid merge tables;
- ordinary marker-looking text versus explicit special insertion.

### Template and binding semantics

- exact BOS/role/end-turn/assistant control placement;
- optional generation prompt;
- marker-looking user content remains ordinary;
- naive flattening produces different IDs;
- mismatched tokenizer identity fails before preparation.

### UTF-8 framing semantics

- partial scalars across token boundaries;
- several complete scalars in one piece;
- valid prefix plus incomplete suffix;
- definite malformed input;
- incomplete terminal suffix;
- empty pieces never create meaningless empty text events.

### Lifecycle integration

- the `blue` then EOS oracle remains stable;
- token and text events are separate and ordered;
- cancellation, model failure, tokenizer failure, malformed output, and
  incomplete output each have one terminal;
- no token, text, or trace event follows terminal;
- repeated semantic streams agree with timing excluded.

Run the gate:

```sh
cd code/mini-engine
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Compare two real tokenizers without ranking them

Toy fixtures establish correctness, but real tokenizers show how much the
configured artifact matters. The durable experiment record is
[`research/part-01/tokenizer-comparison.md`](../../research/part-01/tokenizer-comparison.md).

It compares:

1. OpenAI `tiktoken` 0.14.0, named encoding `cl100k_base`;
2. SentencePiece 0.2.2 with Google's pinned small Japanese BPE byte-fallback
   test model.

The SentencePiece artifact is intentionally labeled as a test fixture, not a
production LLM tokenizer. The comparison observes counts and pieces. It makes
no speed or quality claim.

| Input | UTF-8 bytes | Scalars | `cl100k_base` | SentencePiece fixture |
| --- | ---: | ---: | ---: | ---: |
| `Tokenizers count spaces.` | 24 | 24 | 5 | 25 |
| two-line C-like loop | 39 | 39 | 21 | 38 |
| `模型把文本变成编号。` | 30 | 10 | 10 | 19 |
| `👩🏽‍💻🚀` | 19 | 5 | 13 | 20 |
| whitespace/newline fixture | 34 | 34 | 9 | 30 |

The numbers are not a scoreboard. The small SentencePiece vocabulary was
trained on a different corpus and has different normalization. Its behavior is
useful precisely because the setup is unlike `cl100k_base`.

### Pieces can cross scalar boundaries

For the Chinese input, `cl100k_base` splits the UTF-8 bytes of `把` across two
tokens: one piece contains `E6 8A`; the next contains `8A`. Neither is valid
UTF-8 alone. The SentencePiece model uses byte fallback for several Chinese
scalars. Both reconstruct the input after complete byte concatenation.

For `👩🏽‍💻🚀`, `cl100k_base` produces 13 tokens. Many pieces are fragments such
as `F0 9F`, `91`, and `A9`. The SentencePiece fixture produces a dummy
metaspace boundary plus one fallback token per emoji byte, for 20 tokens. This
is the production-shaped reason for the UTF-8 buffer, not an artificial corner
case.

### Normalization can defeat exact surface round trip

The whitespace fixture contains leading spaces, a tab, two internal spaces, a
newline, and trailing spaces. `cl100k_base` preserves the exact surface in this
experiment. The SentencePiece fixture decodes:

```text
leading and internal trailing
```

Its configured normalization removed and collapsed whitespace. Byte fallback
did not restore the original because fallback operated after normalization.
The two-line code input similarly decodes as one normalized line under this
fixture.

This is why a tokenizer contract includes normalization and decoding, not only
the subword algorithm.

## Token count is a systems variable

Tokenization is model semantics, but token count also enters system economics.
If an encoded prompt has length $N$, later stages see $N$ positions, not the
number of words or bytes the caller expected.

For a model context capacity $T_{\max}$ and requested generation budget
$N_{\mathrm{new}}$, admission must at least respect the dimensional constraint

$$
N+N_{\mathrm{new}}\le T_{\max}\quad[\mathrm{token\ positions}].
$$

Token count affects:

- whether input fits the model's context limit;
- how much prefill work Chapter 20 will analyze;
- how much per-position KV state later chapters allocate;
- prefix/cache matching in token space;
- maximum-generation and stop accounting;
- batching shapes and admission budgets;
- provider billing when tokens are the accounting unit;
- rate limits and observability labels.

Do not turn this list into a premature performance formula. Different model
architectures, batching, hardware, and token distributions change costs. The
durable statement is dimensional: the engine schedules token positions, so the
correct tokenizer's $N$ is an input to later work and memory models. The
canonical [system-impact map](../../diagrams/tokenizer/token-count-system-impact.txt)
shows those downstream dependencies without inventing a cost coefficient.

A tokenizer with fewer tokens for one sentence is not automatically better.
Vocabulary size `V` also affects future embedding and output dimensions. A
larger vocabulary can reduce sequence length for some inputs while increasing
parameter rows and changing training behavior. Language coverage and
segmentation quality matter. We have not measured any end-to-end tradeoff here.

### Cache boundaries use IDs, not reconstructed text

Two text surfaces that render identically after normalization may have
different original bytes. Two tokenizers can produce different ID sequences
for the same bytes. Prefix reuse in a model runtime must compare the identities
used for model state, not a casually re-rendered string.

Hermon's current batched runtime tokenizes before choosing its best prefix slot
because the reusable state corresponds to token positions. Its separate
`hermon-tokenizer` crate sketches a tokenizer prefix cache, but that crate does
not own the current real-model route.

## Inside Hermon: pinned llama.cpp owns the current path

The following account is **CURRENT** for Hermon commit
[`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25),
inspected on 2026-09-02. The detailed evidence is in the
[Chapter 2 research note](../../research/part-01/chapter-02-from-text-to-tokens.md).

The Hermon repository pins llama.cpp submodule commit
[`389ff61`](https://github.com/ggml-org/llama.cpp/commit/389ff61d77b5c71cec0cf92fe4e5d01ace80b797).
The upstream llama.cpp head observed during the inspection was newer. Pinned
does not mean stale by itself, and CURRENT does not mean “equal to upstream
HEAD.” It means that this pinned source owns the reachable production path we
traced.

> **INSIDE HERMON — CURRENT**
> The default `BatchedRuntime` constructs a message view, asks the llama.cpp
> model wrapper to apply the model's embedded chat template with an assistant
> generation prompt, and uses a naive role flattener only when no embedded
> template exists. It then calls context tokenization with special addition and
> special parsing enabled.

The Rust safe wrapper in `hermon-llamacpp/src/linked.rs` crosses a narrow C shim.
The shim gets the model's llama.cpp vocabulary and calls `llama_tokenize`. The
two Boolean controls matter: llama.cpp documents `add_special` as allowing
configured BOS/EOS addition and `parse_special` as allowing control-token
spellings to be recognized instead of treated as plaintext.

Hermon's blocking raw stream calls the same tokenizer with `parse_special=false`.
Its chat path first applies a trusted template, then uses true. This is the same
authority distinction our typed teaching segments make, expressed through the
pinned upstream API.

> **INSIDE HERMON — LIBRARY**
> `crates/hermon-tokenizer` currently defines a tokenizer-kind enum, a
> `Tokenize` trait, and a prefix-cache skeleton. Its own source labels BPE,
> SentencePiece, Tiktoken, and Hugging Face ingestion as future work. Repository
> search found no current real-model request route using it.

Calling that crate “Hermon's tokenizer” without the status would be misleading.
The real-model owner today is the pinned llama.cpp vocabulary path. The crate is
a LIBRARY/research surface.

### Hermon's UTF-8 buffer

Each active batched sequence owns `utf8_buf: Vec<u8>`. After sampling, the
worker calls `token_to_piece(token, false)`, appends the resulting bytes, finds
the longest prefix accepted by Rust's UTF-8 validator, emits that prefix as a
`StreamItem::Piece(String)`, and retains the rest. This is direct production
evidence that piece and token boundaries differ.

The current implementation has a bounded caveat.

> **INSIDE HERMON — CURRENT CAVEAT**
> The buffer path uses `valid_up_to()` without separately testing whether
> `error_len()` identifies a definite malformed sequence. At successful
> finalization it converts leftover bytes with `String::from_utf8_lossy`.

Therefore Hermon's current policy does not match ENGINE-0's strict error
policy. A malformed suffix can remain buffered, and an incomplete or invalid
terminal suffix is rendered with replacement rather than becoming an explicit
runtime error. The blocking convenience engine has a similar lossy terminal
flush.

This chapter records the behavior; it does not modify Hermon. Later production
streaming and protocol chapters must decide the desired policy, test it at the
runtime boundary, and trace terminal cause to the wire.

### What tests establish

Hermon's ignored real-model tests exercise tokenization, chat-template
application, token-to-piece concatenation, and next-token differential behavior
when an external model fixture and linked backend are supplied. They establish
useful model-dependent evidence when run. Ordinary CI without the fixture does
not prove every tokenizer/model combination.

The inspected default path has no focused unit test separating valid-incomplete
from definitively malformed UTF-8 fragments. ENGINE-0's tests are evidence for
the teaching contract, not retroactive proof of Hermon.

## Common mistakes

### “A token is a word”

Tokens can contain a whole word, a prefix, punctuation, a leading space, many
bytes, or an incomplete scalar fragment. Use vocabulary identity as the
definition.

### “UTF-8 characters are tokens”

UTF-8 encodes scalar values into bytes. Tokenization is a separate configured
mapping. A tokenizer can split within the byte encoding of one scalar.

### “SentencePiece means Unigram”

SentencePiece supports Unigram, BPE, word, and character model types. Inspect
the serialized model configuration.

### “Byte fallback guarantees exact input round trip”

It guarantees coverage of bytes presented to the vocabulary model. Earlier
normalization can already have changed whitespace or scalar sequences.

### “Decode each token as a string”

An individual piece may be invalid UTF-8. Decode to bytes, buffer per request,
and emit only complete valid prefixes under a named malformed-data policy.

### “The marker string is the special token”

The control identity is the token. A printed marker is a diagnostic surface.
Ordinary user bytes that resemble it must not gain control authority
implicitly.

### “Any chat formatting is close enough”

The template defines exact model input. A plausible `role: content` string can
be syntactically valid and semantically wrong.

### “Add BOS and EOS everywhere for safety”

Templates and tokenizer post-processors may already insert required controls.
Duplication changes the sequence. Follow the bound model contract.

### “Fewer tokens is always better”

Token count affects work, but vocabulary size, language coverage, learned
semantics, model compatibility, and workload also matter. Counts from unrelated
tokenizers are not a quality leaderboard.

> **ENGINEERING FAILURE**
> A service loads weights from revision A and a tokenizer/template from
> revision B. Every file parses and all token IDs are in range. The model emits
> poor or strangely terminated output. The defect is not numerical: the same
> integers select different learned vocabulary rows and the role controls no
> longer match training. Artifact identity must bind the components before
> execution.

## Exercises

### CHECK

1. For `é` written as U+0065 U+0301, count visible graphemes, scalar values,
   UTF-8 bytes, and tokens under the toy BPE. Repeat after NFC composition.
2. Explain why byte vectors can round-trip through `TinyBpeTokenizer` even when
   they cannot enter `Utf8StreamDecoder` successfully.
3. Classify BOS, EOS, PAD, UNK, USER, and END_TURN as ordinary content,
   controls, or information-loss markers. Which names can share behavior only
   if model metadata says so?
4. A prompt has 1,000 Unicode scalar values and 1,600 tokens. Which count enters
   a 1,200-token context limit? What additional model/runtime facts are needed
   to estimate memory or latency?

### BUILD

Complete [Lab 2](../../labs/lab-02-tokenize-by-hand.md),
[Lab 3](../../labs/lab-03-stream-utf8-across-tokens.md), and
[Lab 4](../../labs/lab-04-use-the-wrong-chat-template.md). Keep the hand oracle,
implementation, and failure injection separate.

Add a CLI input containing a tab, newline, Chinese text, and an emoji. Record
bytes, scalar values, IDs, and decoded bytes. Do not call the ID count a word
count.

### BREAK

In a temporary branch:

1. make ordinary encoding recognize `<|assistant|>` as control ID 1006;
2. feed the marker inside a user message;
3. observe the special-insertion test fail;
4. restore the typed boundary.

Then replace the strict terminal check with lossy UTF-8 conversion. Run the
malformed and incomplete fixtures and explain which evidence disappeared.

### EXTEND

Implement an alternative tiny tokenizer configuration that applies one named
normalization rule before the same BPE. Add paired composed/decomposed and
whitespace fixtures. State exactly which byte round trips no longer hold.

Do not add a third-party dependency merely to hide the transform. The purpose
is to make the semantic difference inspectable.

## What this chapter has not built

We still do not have a language model.

`DemoModel` has no parameters. A token ID does not yet select an embedding row.
There is no hidden vector, output projection, vocabulary-shaped score array, or
logit. The candidate list contains small hand-ranked integers so the Chapter 1
lifecycle remains executable while we replace one boundary at a time.

We also have not built a production tokenizer loader, trained a vocabulary,
implemented efficient BPE, benchmarked allocations, or selected the real model
artifact for later parts. The real-tokenizer experiment uses external packages
in a temporary environment and commits only small textual observations.

We have not implemented stochastic sampling, temperature, softmax, top-k,
top-p, random-number state, stop strings, or the complete autoregressive
feedback loop. Chapter 4 owns those mechanisms after genuine logits exist.

## Summary

The visible input to an application passes through several representations
before a model can use it. Graphemes, Unicode scalar values, UTF-8 bytes, and
token IDs are different sequences with different boundaries. A token is one
identity in one configured vocabulary.

A tokenizer is more than a segmentation algorithm. Normalization,
pre-tokenization, vocabulary/model rules, post-processing, special tokens, and
decoding jointly determine behavior. BPE encoding applies a fixed ranked merge
artifact; it does not learn from the prompt. SentencePiece can store BPE or
Unigram models, and the algorithms are not interchangeable. Byte fallback
provides coverage after configured preprocessing but cannot recover information
already removed by normalization.

Special tokens are controls with explicit authority. Ordinary text that looks
like a marker must remain ordinary. Chat templates serialize structured
messages into the exact sequence expected by the model, including role and
turn boundaries. Raw completion and chat completion therefore need different
preparation paths. Model, tokenizer revision, template, and special semantics
form one contract.

On output, vocabulary lookup produces bytes. Token pieces can split UTF-8
scalar values, so a per-request decoder buffers a valid incomplete suffix and
emits only complete text. Token identity events and text-piece events are not
one-to-one. Malformed and incomplete terminal behavior must be a named policy.

ENGINE-0 now implements these boundaries with no external Rust dependencies.
Its request owns token IDs, its tokenizer owns ID-to-byte meaning, its template
inserts typed controls, its contract rejects mismatched identities, and its
UTF-8 framer rejects malformed output without lossy replacement. Its request
lifecycle and exactly-once terminal behavior remain intact.

Hermon's current real-model path reaches the pinned llama.cpp tokenizer and
embedded chat-template APIs. The separate `hermon-tokenizer` crate remains a
LIBRARY skeleton. Hermon's per-sequence UTF-8 buffer demonstrates the real
token/piece mismatch, while its lossy terminal flush documents a policy
difference for later audit.

## Next: the smallest possible language model

Chapter 3 receives stable token IDs. It will replace `DemoModel` itself—not
wrap it—with the smallest genuine numerical language model. We will distinguish
immutable learned parameters from runtime activations, look up an embedding
row, carry a hidden vector through an output projection, produce one logit per
vocabulary item, account for every shape, and verify the result with an
independent hand-computable oracle.

## Primary references

- The Unicode Consortium, [The Unicode Standard, Version
  17.0.0, Chapter 3](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/)
  and [Unicode Standard Annex #15: Unicode Normalization
  Forms](https://www.unicode.org/reports/tr15/).
- IETF, [RFC 3629 — UTF-8, a transformation format of ISO
  10646](https://www.rfc-editor.org/rfc/rfc3629).
- Rico Sennrich, Barry Haddow, and Alexandra Birch, “Neural Machine Translation
  of Rare Words with Subword Units,” and the official
  [`subword-nmt` implementation at `92d6139`](https://github.com/rsennrich/subword-nmt/tree/92d6139d07d30e12735a0af9e7f7f925ebe62c54).
- OpenAI, [GPT-2 `encoder.py` at
  `9b63575`](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py)
  and [`tiktoken`](https://github.com/openai/tiktoken).
- Google, [SentencePiece source and documentation at
  `ac0f71d`](https://github.com/google/sentencepiece/tree/ac0f71dcbc85f94292266e55bf6cee1b4d6c9dc1),
  including the serialized model type and normalization/special-symbol docs.
- Hugging Face, [Tokenizer
  components](https://huggingface.co/docs/tokenizers/components) and
  [chat-template guidance](https://huggingface.co/docs/transformers/chat_templating),
  source snapshots `d582781` and `e15d467` recorded in the research note.
- llama.cpp, [tokenization and chat-template API at Hermon's pinned
  `389ff61`](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/include/llama.h).
- Hermon, [`batched.rs`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/batched.rs),
  [`hermon-llamacpp`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-llamacpp/src/linked.rs),
  and [`hermon-tokenizer`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-tokenizer/src/lib.rs)
  at commit `472a44c`.
