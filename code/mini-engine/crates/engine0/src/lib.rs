#![forbid(unsafe_code)]

//! ENGINE-1 after Chapter 3: a tiny numerical language-model runtime.
//!
//! A token ID now selects an embedding row and a scalar output projection
//! produces genuine vocabulary logits. Deterministic argmax remains separate
//! from model execution until Chapter 4 introduces sampling.

pub mod chat;
pub mod model;
pub mod tokenizer;
pub mod utf8;

use std::fmt;
use std::time::{Duration, Instant};

use chat::{ContractError, ModelContract};
use model::{ForwardPass, Logits, Model};
use tokenizer::{SpecialToken, TokenId, Tokenizer};
use utf8::Utf8StreamDecoder;

pub type RequestId = u64;

/// A generation request after text has crossed the tokenizer boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub id: RequestId,
    pub input_tokens: Vec<TokenId>,
    pub input_bytes: usize,
    pub max_new_tokens: usize,
}

impl Request {
    pub fn from_text(
        id: RequestId,
        input: &[u8],
        max_new_tokens: usize,
        tokenizer: &impl Tokenizer,
    ) -> Result<Self, tokenizer::TokenizerError> {
        Ok(Self {
            id,
            input_tokens: tokenizer.encode(input)?,
            input_bytes: input.len(),
            max_new_tokens,
        })
    }

    pub fn from_token_ids(
        id: RequestId,
        input_tokens: Vec<TokenId>,
        input_bytes: usize,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            id,
            input_tokens,
            input_bytes,
            max_new_tokens,
        }
    }

    fn validate(&self) -> Result<(), GenerationError> {
        if self.input_tokens.is_empty() {
            return Err(GenerationError::InvalidRequest(
                "input must encode to at least one token".to_string(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(GenerationError::InvalidRequest(
                "max_new_tokens must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Text,
    EndOfSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: TokenId,
    pub kind: TokenKind,
}

impl Token {
    pub const fn text(id: TokenId) -> Self {
        Self {
            id,
            kind: TokenKind::Text,
        }
    }

    pub const fn end(id: TokenId) -> Self {
        Self {
            id,
            kind: TokenKind::EndOfSequence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationState {
    generated: Vec<Token>,
}

impl GenerationState {
    fn new() -> Self {
        Self {
            generated: Vec::new(),
        }
    }

    pub fn step(&self) -> usize {
        self.generated.len()
    }

    pub fn generated(&self) -> &[Token] {
        &self.generated
    }
}

pub trait Selector {
    fn select(&self, logits: &Logits) -> Result<TokenId, GenerationError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySelector;

impl Selector for GreedySelector {
    fn select(&self, logits: &Logits) -> Result<TokenId, GenerationError> {
        let mut best_index = 0usize;
        let mut best = *logits.as_slice().first().ok_or(GenerationError::NoLogits)?;
        for (index, value) in logits.as_slice()[1..].iter().copied().enumerate() {
            if value > best {
                best = value;
                best_index = index + 1;
            }
        }
        let id = u32::try_from(best_index).map_err(|_| {
            GenerationError::InvalidRequest("selected token ID exceeds u32".to_string())
        })?;
        Ok(TokenId(id))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    InvalidRequest(String),
    Model(String),
    Tokenizer(String),
    Utf8Stream(String),
    NoLogits,
    AlreadyTerminal,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            Self::Model(message) => write!(f, "model error: {message}"),
            Self::Tokenizer(message) => write!(f, "tokenizer error: {message}"),
            Self::Utf8Stream(message) => write!(f, "UTF-8 stream error: {message}"),
            Self::NoLogits => f.write_str("model returned no logits"),
            Self::AlreadyTerminal => f.write_str("request already has a terminal outcome"),
        }
    }
}

impl std::error::Error for GenerationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfSequence,
    MaxTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalOutcome {
    Completed(StopReason),
    Cancelled,
    Failed(GenerationError),
}

impl fmt::Display for TerminalOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed(StopReason::EndOfSequence) => f.write_str("completed:end_of_sequence"),
            Self::Completed(StopReason::MaxTokens) => f.write_str("completed:max_tokens"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::Failed(error) => write!(f, "failed:{error}"),
        }
    }
}

/// Token identities and valid text pieces are separate ordered events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Token {
        request_id: RequestId,
        index: usize,
        token: Token,
    },
    Text {
        request_id: RequestId,
        through_token_index: usize,
        text: String,
    },
    Terminal {
        request_id: RequestId,
        outcome: TerminalOutcome,
    },
}

pub trait TokenSink {
    fn send(&mut self, event: StreamEvent);
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RecordingSink {
    pub events: Vec<StreamEvent>,
}

impl TokenSink for RecordingSink {
    fn send(&mut self, event: StreamEvent) {
        self.events.push(event);
    }
}

pub trait Cancellation {
    fn is_cancelled(&self, state: &GenerationState) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self, _state: &GenerationState) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CancelAtStep(pub usize);

impl Cancellation for CancelAtStep {
    fn is_cancelled(&self, state: &GenerationState) -> bool {
        state.step() >= self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraceKind {
    InputEncoded {
        bytes: usize,
        token_count: usize,
    },
    Admitted,
    ExecutionStarted,
    ModelInvoked {
        step: usize,
    },
    EmbeddingReady {
        input_token: TokenId,
        shape: [usize; 1],
        values: Vec<f32>,
    },
    LogitsReady {
        shape: [usize; 1],
        values: Vec<f32>,
    },
    TokenSelected {
        step: usize,
        token_id: TokenId,
    },
    TokenEmitted {
        index: usize,
        token_id: TokenId,
    },
    TokenDecoded {
        index: usize,
        token_id: TokenId,
        bytes: usize,
    },
    Utf8Buffered {
        pending_bytes: usize,
    },
    TextEmitted {
        through_token_index: usize,
        bytes: usize,
    },
    Terminal(TerminalOutcome),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceEvent {
    pub at: Duration,
    pub request_id: RequestId,
    pub kind: TraceKind,
}

pub trait TraceSink {
    fn record(&mut self, event: TraceEvent);

    /// Full activations are useful for ENGINE-1 but unsafe as a default for
    /// future large tensors. Sinks must opt in explicitly.
    fn record_tensor_values(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct NoopTrace;

impl TraceSink for NoopTrace {
    fn record(&mut self, _event: TraceEvent) {}
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct RecordingTrace {
    pub events: Vec<TraceEvent>,
}

impl TraceSink for RecordingTrace {
    fn record(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    fn record_tensor_values(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleTimings {
    pub admitted_at: Option<Duration>,
    pub execution_started_at: Option<Duration>,
    pub first_token_ready_at: Option<Duration>,
    pub first_text_emitted_at: Option<Duration>,
    pub terminal_at: Duration,
}

impl LifecycleTimings {
    pub fn queue_delay(&self) -> Option<Duration> {
        Some(self.execution_started_at?.saturating_sub(self.admitted_at?))
    }

    pub fn time_to_first_token(&self) -> Option<Duration> {
        Some(self.first_token_ready_at?.saturating_sub(self.admitted_at?))
    }

    pub fn time_to_first_text(&self) -> Option<Duration> {
        Some(
            self.first_text_emitted_at?
                .saturating_sub(self.admitted_at?),
        )
    }

    pub fn token_ready_to_text_delay(&self) -> Option<Duration> {
        Some(
            self.first_text_emitted_at?
                .saturating_sub(self.first_token_ready_at?),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationResult {
    pub request_id: RequestId,
    pub tokens: Vec<Token>,
    pub outcome: TerminalOutcome,
    pub timings: LifecycleTimings,
}

#[derive(Debug, Clone)]
pub struct Runtime<M, S, T> {
    model: M,
    selector: S,
    tokenizer: T,
    eos_token: TokenId,
}

impl<M, S, T> Runtime<M, S, T>
where
    M: Model,
    S: Selector,
    T: Tokenizer,
{
    pub fn try_new(
        model: M,
        selector: S,
        tokenizer: T,
        contract: &ModelContract,
    ) -> Result<Self, ContractError> {
        contract.validate_vocabulary(&tokenizer, model.vocabulary_size())?;
        let eos_token = tokenizer
            .special_id(SpecialToken::Eos)
            .ok_or(ContractError::MissingEndOfSequence)?;
        Ok(Self {
            model,
            selector,
            tokenizer,
            eos_token,
        })
    }

    pub fn generate(
        &self,
        request: Request,
        cancellation: &dyn Cancellation,
        sink: &mut dyn TokenSink,
        trace: &mut dyn TraceSink,
    ) -> GenerationResult {
        let started = Instant::now();
        let mut lifecycle = Lifecycle::new(request.id, started, sink, trace);
        let mut state = GenerationState::new();
        let mut decoder = Utf8StreamDecoder::new();

        if let Err(error) = request.validate() {
            return lifecycle.result(state, TerminalOutcome::Failed(error));
        }

        lifecycle.trace(TraceKind::InputEncoded {
            bytes: request.input_bytes,
            token_count: request.input_tokens.len(),
        });
        lifecycle.trace(TraceKind::Admitted);
        lifecycle.admitted_at = Some(started.elapsed());
        lifecycle.trace(TraceKind::ExecutionStarted);
        lifecycle.execution_started_at = Some(started.elapsed());

        let outcome = loop {
            if cancellation.is_cancelled(&state) {
                break TerminalOutcome::Cancelled;
            }
            if state.step() >= request.max_new_tokens {
                break completed_or_utf8_error(StopReason::MaxTokens, &decoder);
            }

            lifecycle.trace(TraceKind::ModelInvoked { step: state.step() });
            let mut history = Vec::with_capacity(request.input_tokens.len() + state.step());
            history.extend_from_slice(&request.input_tokens);
            history.extend(state.generated.iter().map(|token| token.id));
            let forward = match self.model.forward(&history) {
                Ok(forward) => forward,
                Err(error) => {
                    break TerminalOutcome::Failed(GenerationError::Model(error.to_string()))
                }
            };
            if forward.logits.len() != self.model.vocabulary_size() {
                break TerminalOutcome::Failed(GenerationError::Model(format!(
                    "model returned {} logits for vocabulary size {}",
                    forward.logits.len(),
                    self.model.vocabulary_size()
                )));
            }
            trace_forward(&mut lifecycle, &forward);
            let selected_id = match self.selector.select(&forward.logits) {
                Ok(token_id) => token_id,
                Err(error) => break TerminalOutcome::Failed(error),
            };
            let selected = if selected_id == self.eos_token {
                Token::end(selected_id)
            } else {
                Token::text(selected_id)
            };
            lifecycle.trace(TraceKind::TokenSelected {
                step: state.step(),
                token_id: selected.id,
            });

            if selected.kind == TokenKind::EndOfSequence {
                break completed_or_utf8_error(StopReason::EndOfSequence, &decoder);
            }

            if lifecycle.first_token_ready_at.is_none() {
                lifecycle.first_token_ready_at = Some(started.elapsed());
            }
            let index = state.generated.len();
            state.generated.push(selected);
            lifecycle.emit_token(index, selected);

            let bytes = match self.tokenizer.decode_token(selected.id) {
                Ok(bytes) => bytes,
                Err(error) => {
                    break TerminalOutcome::Failed(GenerationError::Tokenizer(error.to_string()))
                }
            };
            lifecycle.trace(TraceKind::TokenDecoded {
                index,
                token_id: selected.id,
                bytes: bytes.len(),
            });
            let text = match decoder.push(bytes) {
                Ok(text) => text,
                Err(error) => {
                    break TerminalOutcome::Failed(GenerationError::Utf8Stream(error.to_string()))
                }
            };
            lifecycle.trace(TraceKind::Utf8Buffered {
                pending_bytes: decoder.pending_bytes().len(),
            });
            if let Some(text) = text {
                lifecycle.emit_text(index, text);
            }
        };

        lifecycle.result(state, outcome)
    }
}

fn trace_forward(lifecycle: &mut Lifecycle<'_>, forward: &ForwardPass) {
    if !lifecycle.trace_sink.record_tensor_values() {
        return;
    }
    lifecycle.trace(TraceKind::EmbeddingReady {
        input_token: forward.input_token,
        shape: [forward.hidden.len()],
        values: forward.hidden.clone(),
    });
    lifecycle.trace(TraceKind::LogitsReady {
        shape: [forward.logits.len()],
        values: forward.logits.as_slice().to_vec(),
    });
}

fn completed_or_utf8_error(reason: StopReason, decoder: &Utf8StreamDecoder) -> TerminalOutcome {
    match decoder.finish() {
        Ok(()) => TerminalOutcome::Completed(reason),
        Err(error) => TerminalOutcome::Failed(GenerationError::Utf8Stream(error.to_string())),
    }
}

struct Lifecycle<'a> {
    request_id: RequestId,
    started: Instant,
    sink: &'a mut dyn TokenSink,
    trace_sink: &'a mut dyn TraceSink,
    terminal: bool,
    admitted_at: Option<Duration>,
    execution_started_at: Option<Duration>,
    first_token_ready_at: Option<Duration>,
    first_text_emitted_at: Option<Duration>,
}

impl<'a> Lifecycle<'a> {
    fn new(
        request_id: RequestId,
        started: Instant,
        sink: &'a mut dyn TokenSink,
        trace_sink: &'a mut dyn TraceSink,
    ) -> Self {
        Self {
            request_id,
            started,
            sink,
            trace_sink,
            terminal: false,
            admitted_at: None,
            execution_started_at: None,
            first_token_ready_at: None,
            first_text_emitted_at: None,
        }
    }

    fn trace(&mut self, kind: TraceKind) {
        if self.terminal {
            return;
        }
        self.trace_sink.record(TraceEvent {
            at: self.started.elapsed(),
            request_id: self.request_id,
            kind,
        });
    }

    fn emit_token(&mut self, index: usize, token: Token) {
        if self.terminal {
            return;
        }
        self.sink.send(StreamEvent::Token {
            request_id: self.request_id,
            index,
            token,
        });
        self.trace(TraceKind::TokenEmitted {
            index,
            token_id: token.id,
        });
    }

    fn emit_text(&mut self, through_token_index: usize, text: String) {
        if self.terminal || text.is_empty() {
            return;
        }
        let bytes = text.len();
        self.sink.send(StreamEvent::Text {
            request_id: self.request_id,
            through_token_index,
            text,
        });
        if self.first_text_emitted_at.is_none() {
            self.first_text_emitted_at = Some(self.started.elapsed());
        }
        self.trace(TraceKind::TextEmitted {
            through_token_index,
            bytes,
        });
    }

    fn finish(&mut self, outcome: TerminalOutcome) -> Result<Duration, GenerationError> {
        if self.terminal {
            return Err(GenerationError::AlreadyTerminal);
        }
        self.sink.send(StreamEvent::Terminal {
            request_id: self.request_id,
            outcome: outcome.clone(),
        });
        self.trace_sink.record(TraceEvent {
            at: self.started.elapsed(),
            request_id: self.request_id,
            kind: TraceKind::Terminal(outcome),
        });
        self.terminal = true;
        Ok(self.started.elapsed())
    }

    fn result(mut self, state: GenerationState, outcome: TerminalOutcome) -> GenerationResult {
        let terminal_at = self
            .finish(outcome.clone())
            .expect("runtime owns the only terminal transition");
        GenerationResult {
            request_id: self.request_id,
            tokens: state.generated,
            outcome,
            timings: LifecycleTimings {
                admitted_at: self.admitted_at,
                execution_started_at: self.execution_started_at,
                first_token_ready_at: self.first_token_ready_at,
                first_text_emitted_at: self.first_text_emitted_at,
                terminal_at,
            },
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::tokenizer::TINY_LM_RUST;

    #[test]
    fn terminal_transition_can_happen_only_once() {
        let started = Instant::now();
        let mut sink = RecordingSink::default();
        let mut trace = RecordingTrace::default();
        let mut lifecycle = Lifecycle::new(7, started, &mut sink, &mut trace);

        assert!(lifecycle
            .finish(TerminalOutcome::Completed(StopReason::MaxTokens))
            .is_ok());
        assert_eq!(
            lifecycle.finish(TerminalOutcome::Cancelled),
            Err(GenerationError::AlreadyTerminal)
        );
    }

    #[test]
    fn token_text_and_trace_emission_are_blocked_after_terminal() {
        let started = Instant::now();
        let mut sink = RecordingSink::default();
        let mut trace = RecordingTrace::default();
        let mut lifecycle = Lifecycle::new(8, started, &mut sink, &mut trace);
        lifecycle
            .finish(TerminalOutcome::Cancelled)
            .expect("first terminal");
        lifecycle.emit_token(0, Token::text(TINY_LM_RUST));
        lifecycle.emit_text(0, "forbidden".to_string());
        lifecycle.trace(TraceKind::ExecutionStarted);

        assert_eq!(sink.events.len(), 1);
        assert_eq!(trace.events.len(), 1);
    }
}
