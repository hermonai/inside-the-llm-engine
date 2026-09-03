# Runtime Diagrams

Request lifecycle, prefill/decode iteration, stream, cancellation, and failure
flows.

Chapter 1 canonical diagrams:

The full purpose/status metadata is in [`../INDEX.md`](../INDEX.md).

- [`request-to-token.txt`](request-to-token.txt) — lifecycle and terminal paths;
- [`model-vs-engine.txt`](model-vs-engine.txt) — artifact, running model, and runtime state;
- [`inference-stack.txt`](inference-stack.txt) — library, engine, server, service, provider, and backend;
- [`token-byte-owner.txt`](token-byte-owner.txt) — the book's three recurring journeys.
- [`latency-decomposition.txt`](latency-decomposition.txt) — named latency intervals;
- [`control-plane-data-plane.txt`](control-plane-data-plane.txt) — decision and byte-movement planes;
- [`hermon-current-request-path.txt`](hermon-current-request-path.txt) — verified Hermon status path.
