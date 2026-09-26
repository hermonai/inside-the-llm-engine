//! The engine loop: admission, a per-step token budget, chunked prefill,
//! sampling, streaming text, cancellation, and exactly one terminal event per
//! request.
//!
//! Each call to [`Engine::step`] is one scheduler iteration. It spends at most
//! `token_budget` tokens: first one token for every sequence that is decoding,
//! so running requests keep their pace, then prompt chunks for sequences still
//! in prefill, oldest first. A request is admitted only when the pool can hold
//! its prompt plus its longest possible answer, so no admitted request is ever
//! preempted; the price is that memory is reserved for tokens that may never be
//! generated, which is the trade Chapter 16 prices.

use std::collections::{HashSet, VecDeque};
use std::fmt;

use engine0::sampling::SplitMix64;
use engine0::utf8::Utf8StreamDecoder;
use engine0::StopReason;

use crate::kernels::{argmax, softmax};
use crate::kv::{BlockId, KvError, KvPool};
use crate::model::{Model, SeqChunk};

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    Greedy,
    /// Softmax at `temperature`, drawn with a per-request seeded generator, so
    /// a request's output does not depend on what else is running.
    Temperature {
        temperature: f32,
        seed: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub stop: Option<u32>,
    pub sampling: Sampling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub n_blocks: usize,
    pub block_size: usize,
    pub max_running: usize,
    pub token_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Completed(StopReason),
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The `index`-th generated token of a request.
    Token {
        id: RequestId,
        index: usize,
        token: u32,
    },
    /// Text that became complete UTF-8 with the latest token (possibly empty
    /// while a multi-byte character is still arriving).
    Text { id: RequestId, text: String },
    /// Exactly once per admitted request, and nothing for it afterwards.
    Finished {
        id: RequestId,
        outcome: Outcome,
        generated: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    EmptyPrompt,
    ZeroMaxTokens,
    TokenOutOfVocabulary(u32),
    /// The request could never fit, even in an empty pool.
    TooLong {
        blocks_needed: usize,
        pool_blocks: usize,
    },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => f.write_str("empty prompt"),
            Self::ZeroMaxTokens => f.write_str("max_new_tokens must be positive"),
            Self::TokenOutOfVocabulary(t) => write!(f, "token {t} is outside the vocabulary"),
            Self::TooLong {
                blocks_needed,
                pool_blocks,
            } => {
                write!(
                    f,
                    "needs {blocks_needed} cache blocks; the pool has {pool_blocks}"
                )
            }
        }
    }
}

impl std::error::Error for SubmitError {}

struct Sequence {
    id: RequestId,
    prompt: Vec<u32>,
    generated: Vec<u32>,
    table: Vec<BlockId>,
    /// Positions whose keys and values are in the cache.
    computed: usize,
    max_new_tokens: usize,
    stop: Option<u32>,
    sampling: Sampling,
    rng: SplitMix64,
    utf8: Utf8StreamDecoder,
}

impl Sequence {
    fn in_prefill(&self) -> bool {
        self.computed < self.prompt.len()
    }
}

/// Counts an engine keeps about its own steps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub steps: u64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
}

pub struct Engine {
    model: Model,
    pool: KvPool,
    cfg: EngineConfig,
    waiting: VecDeque<(RequestId, GenerationRequest)>,
    running: Vec<Sequence>,
    cancelled: HashSet<RequestId>,
    next_id: RequestId,
    pub stats: Stats,
}

impl Engine {
    /// Panics if the budget cannot give every running sequence its decode
    /// token each step: sequences beyond the budget would never be scheduled.
    pub fn new(model: Model, cfg: EngineConfig) -> Self {
        assert!(
            cfg.token_budget >= cfg.max_running && cfg.max_running > 0,
            "token_budget ({}) must be at least max_running ({}), which must be positive",
            cfg.token_budget,
            cfg.max_running
        );
        let pool = KvPool::new(
            cfg.n_blocks,
            cfg.block_size,
            model.cfg.n_layers,
            model.cfg.kv_width(),
        );
        Self {
            model,
            pool,
            cfg,
            waiting: VecDeque::new(),
            running: Vec::new(),
            cancelled: HashSet::new(),
            next_id: 1,
            stats: Stats::default(),
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn free_blocks(&self) -> usize {
        self.pool.free_blocks()
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn running(&self) -> usize {
        self.running.len()
    }

    pub fn submit(&mut self, req: GenerationRequest) -> Result<RequestId, SubmitError> {
        if req.prompt.is_empty() {
            return Err(SubmitError::EmptyPrompt);
        }
        if req.max_new_tokens == 0 {
            return Err(SubmitError::ZeroMaxTokens);
        }
        if let Some(&t) = req
            .prompt
            .iter()
            .find(|&&t| t as usize >= self.model.cfg.vocab)
        {
            return Err(SubmitError::TokenOutOfVocabulary(t));
        }
        let needed = self.pool.blocks_for(req.prompt.len() + req.max_new_tokens);
        if needed > self.pool.total_blocks() {
            return Err(SubmitError::TooLong {
                blocks_needed: needed,
                pool_blocks: self.pool.total_blocks(),
            });
        }
        let id = self.next_id;
        self.next_id += 1;
        self.waiting.push_back((id, req));
        Ok(id)
    }

    /// Takes effect at the start of the next step.
    pub fn cancel(&mut self, id: RequestId) {
        self.cancelled.insert(id);
    }

    /// One scheduler iteration. Returns the events it produced, in order.
    pub fn step(&mut self) -> Result<Vec<Event>, KvError> {
        let mut events = Vec::new();
        self.apply_cancellations(&mut events)?;
        self.admit()?;
        if self.running.is_empty() {
            return Ok(events);
        }

        // Spend the budget: decode tokens first, then prompt chunks, oldest first.
        let mut budget = self.cfg.token_budget;
        let mut plan: Vec<(usize, usize, usize)> = Vec::new(); // (sequence, start, len)
        for (i, s) in self.running.iter().enumerate() {
            if budget > 0 && !s.in_prefill() {
                plan.push((i, s.computed, 1));
                budget -= 1;
            }
        }
        for (i, s) in self.running.iter().enumerate() {
            if budget > 0 && s.in_prefill() {
                let len = (s.prompt.len() - s.computed).min(budget);
                plan.push((i, s.computed, len));
                budget -= len;
            }
        }
        if plan.is_empty() {
            return Ok(events);
        }

        let token_lists: Vec<Vec<u32>> = plan
            .iter()
            .map(|&(i, start, len)| {
                let s = &self.running[i];
                (start..start + len)
                    .map(|p| {
                        if p < s.prompt.len() {
                            s.prompt[p]
                        } else {
                            s.generated[p - s.prompt.len()]
                        }
                    })
                    .collect()
            })
            .collect();
        let chunks: Vec<SeqChunk<'_>> = plan
            .iter()
            .zip(&token_lists)
            .map(|(&(i, start, len), tokens)| {
                let s = &self.running[i];
                SeqChunk {
                    tokens,
                    start,
                    table: &s.table,
                    want_logits: start + len >= s.prompt.len(),
                }
            })
            .collect();
        let logits = self.model.forward(&chunks, &mut self.pool)?;
        drop(chunks);

        self.stats.steps += 1;
        let mut finished = Vec::new();
        for (&(i, _, len), out) in plan.iter().zip(logits) {
            let s = &mut self.running[i];
            if s.in_prefill() {
                self.stats.prefill_tokens += len as u64;
            } else {
                self.stats.decode_tokens += len as u64;
            }
            s.computed += len;
            let Some(logits) = out else { continue };
            let token = sample(&logits, s.sampling, &mut s.rng);
            s.generated.push(token);
            events.push(Event::Token {
                id: s.id,
                index: s.generated.len() - 1,
                token,
            });
            let text = match s.utf8.push(&[token as u8]) {
                Ok(piece) => piece.unwrap_or_default(),
                Err(_) => {
                    // Not valid UTF-8: show a replacement character and resynchronize.
                    s.utf8 = Utf8StreamDecoder::new();
                    "\u{FFFD}".to_string()
                }
            };
            events.push(Event::Text { id: s.id, text });
            let reason = if Some(token) == s.stop {
                Some(StopReason::EndOfSequence)
            } else if s.generated.len() == s.max_new_tokens {
                Some(StopReason::MaxTokens)
            } else {
                None
            };
            if let Some(reason) = reason {
                finished.push((i, Outcome::Completed(reason)));
            }
        }
        // Remove finished sequences from the back so indices stay valid.
        finished.sort_by(|a, b| b.0.cmp(&a.0));
        for (i, outcome) in finished {
            let s = self.running.remove(i);
            self.pool.release(&s.table)?;
            events.push(Event::Finished {
                id: s.id,
                outcome,
                generated: s.generated.len(),
            });
        }
        Ok(events)
    }

    /// Steps until every submitted request has finished.
    pub fn run_to_completion(&mut self) -> Result<Vec<Event>, KvError> {
        let mut all = Vec::new();
        while !self.is_idle() {
            all.extend(self.step()?);
        }
        Ok(all)
    }

    fn apply_cancellations(&mut self, events: &mut Vec<Event>) -> Result<(), KvError> {
        if self.cancelled.is_empty() {
            return Ok(());
        }
        let cancelled = std::mem::take(&mut self.cancelled);
        let mut kept = VecDeque::new();
        for (id, req) in self.waiting.drain(..) {
            if cancelled.contains(&id) {
                events.push(Event::Finished {
                    id,
                    outcome: Outcome::Cancelled,
                    generated: 0,
                });
            } else {
                kept.push_back((id, req));
            }
        }
        self.waiting = kept;
        let mut i = 0;
        while i < self.running.len() {
            if cancelled.contains(&self.running[i].id) {
                let s = self.running.remove(i);
                self.pool.release(&s.table)?;
                events.push(Event::Finished {
                    id: s.id,
                    outcome: Outcome::Cancelled,
                    generated: s.generated.len(),
                });
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// First come, first served: the head of the queue waits for memory rather
    /// than being overtaken, so a long request cannot starve.
    fn admit(&mut self) -> Result<(), KvError> {
        while self.running.len() < self.cfg.max_running {
            let Some((_, req)) = self.waiting.front() else {
                break;
            };
            let needed = self.pool.blocks_for(req.prompt.len() + req.max_new_tokens);
            if needed > self.pool.free_blocks() {
                break;
            }
            let (id, req) = self.waiting.pop_front().expect("peeked above");
            let table = self.pool.alloc(needed)?;
            let seed = match req.sampling {
                Sampling::Temperature { seed, .. } => seed,
                Sampling::Greedy => 0,
            };
            self.running.push(Sequence {
                id,
                prompt: req.prompt,
                generated: Vec::new(),
                table,
                computed: 0,
                max_new_tokens: req.max_new_tokens,
                stop: req.stop,
                sampling: req.sampling,
                rng: SplitMix64::new(seed),
                utf8: Utf8StreamDecoder::new(),
            });
        }
        Ok(())
    }
}

fn sample(logits: &[f32], sampling: Sampling, rng: &mut SplitMix64) -> u32 {
    match sampling {
        Sampling::Greedy => argmax(logits) as u32,
        Sampling::Temperature { temperature, .. } => {
            let mut p: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
            softmax(&mut p);
            let draw = rng.next_unit_f64() as f32;
            let mut cumulative = 0.0f32;
            for (i, &pi) in p.iter().enumerate() {
                cumulative += pi;
                if draw < cumulative {
                    return i as u32;
                }
            }
            (p.len() - 1) as u32
        }
    }
}
