#![forbid(unsafe_code)]

//! The Chapter 42 capstone: a small inference engine, end to end.
//!
//! The pieces follow the book's order. A [`config::ModelConfig`] fixes the
//! shapes; [`weights::Weights`] fills a Llama-style decoder with seeded random
//! numbers, so the engine needs no download and every run is reproducible.
//! [`kernels`] holds the arithmetic, each with one fixed summation order.
//! [`kv::KvPool`] is the paged cache: fixed-size blocks, a free list, and a
//! block table per sequence. [`model::Model::forward`] runs one scheduler step
//! for any mix of prefill chunks and decode tokens. [`engine::Engine`] is the
//! loop around it: admission, a per-step token budget, chunked prefill,
//! sampling, streaming UTF-8 text, cancellation, and exactly one terminal
//! event per request. [`oracle`] recomputes every position from scratch, with
//! no cache, no paging and no batching; the tests require the engine to match
//! it bit for bit.
//!
//! Bit-for-bit agreement is possible because every kernel computes each output
//! element with the same loop whatever else is in the batch. That is the
//! batch invariance of Chapter 38, bought here by never splitting a reduction
//! differently for different batch sizes. Production kernels give it up for
//! speed; this engine keeps it so its tests can be exact.

pub mod config;
pub mod engine;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod oracle;
pub mod weights;

pub use config::ModelConfig;
pub use engine::{Engine, EngineConfig, Event, GenerationRequest, Outcome};
pub use model::Model;
