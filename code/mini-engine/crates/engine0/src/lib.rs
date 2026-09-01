#![forbid(unsafe_code)]

//! ENGINE-0: a deterministic request-to-token lifecycle.
//!
//! This crate deliberately contains no tokenizer, neural network, model-file
//! parser, network server, or hardware backend. Its fake model returns small,
//! hand-computable candidate sets so Chapter 1 can isolate runtime ownership,
//! streaming, stopping, cancellation, failure, and timing boundaries.

use std::fmt;
use std::time::{Duration, Instant};

/// Stable identity for one submitted generation request.
pub type RequestId = u64;

/// The smallest generation request ENGINE-0 accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub id: RequestId,
    pub prompt: String,
    pub max_new_tokens: usize,
}

impl Request {
    pub fn new(id: RequestId, prompt: impl Into<String>, max_new_tokens: usize) -> Self {
        Self {
            id,
            prompt: prompt.into(),
            max_new_tokens,
        }
    }

    fn validate(&self) -> Result<(), GenerationError> {
        if self.prompt.trim().is_empty() {
            return Err(GenerationError::InvalidRequest(
                "prompt must contain non-whitespace text".to_string(),
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

/// Whether a token is ordinary output or the model's end marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Text,
    EndOfSequence,
}

/// One vocabulary identity plus its teaching text representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub kind: TokenKind,
}

impl Token {
    pub fn text(id: u32, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
            kind: TokenKind::Text,
        }
    }

    pub fn end(id: u32) -> Self {
        Self {
            id,
            text: "<eos>".to_string(),
            kind: TokenKind::EndOfSequence,
        }
    }
}

/// A fake, scored model candidate. The score is not a real neural logit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub token: Token,
    pub score: i32,
}

impl Candidate {
    pub fn new(token: Token, score: i32) -> Self {
        Self { token, score }
    }
}

/// Mutable state owned by the runtime for one active request.
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

/// Replace this fake source with a tokenizer/model in later milestones.
pub trait Model {
    fn candidates(
        &self,
        request: &Request,
        state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError>;
}

/// Chooses one token from a model's candidate set.
pub trait Selector {
    fn select(&self, candidates: &[Candidate]) -> Result<Token, GenerationError>;
}

/// Deterministic highest-score selection with stable first-candidate tie-break.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySelector;

impl Selector for GreedySelector {
    fn select(&self, candidates: &[Candidate]) -> Result<Token, GenerationError> {
        let mut best = candidates.first().ok_or(GenerationError::NoCandidates)?;
        for candidate in &candidates[1..] {
            if candidate.score > best.score {
                best = candidate;
            }
        }
        Ok(best.token.clone())
    }
}

/// The small deterministic source used by the executable and Lab 1.
#[derive(Debug, Default, Clone, Copy)]
pub struct DemoModel {
    fail_at_step: Option<usize>,
}

impl DemoModel {
    pub fn failing_at(step: usize) -> Self {
        Self {
            fail_at_step: Some(step),
        }
    }
}

impl Model for DemoModel {
    fn candidates(
        &self,
        _request: &Request,
        state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError> {
        if self.fail_at_step == Some(state.step()) {
            return Err(ModelError::new(format!(
                "injected model failure at step {}",
                state.step()
            )));
        }

        if state.step() == 0 {
            Ok(vec![
                Candidate::new(Token::text(1, "blue"), 9),
                Candidate::new(Token::text(2, "green"), 4),
                Candidate::new(Token::end(0), 1),
            ])
        } else {
            Ok(vec![
                Candidate::new(Token::end(0), 10),
                Candidate::new(Token::text(1, "blue"), 1),
            ])
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError {
    message: String,
}

impl ModelError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    InvalidRequest(String),
    Model(String),
    NoCandidates,
    AlreadyTerminal,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            Self::Model(message) => write!(f, "model error: {message}"),
            Self::NoCandidates => f.write_str("model returned no candidates"),
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

/// Ordered events visible to a consumer of generated output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Token {
        request_id: RequestId,
        index: usize,
        token: Token,
    },
    Terminal {
        request_id: RequestId,
        outcome: TerminalOutcome,
    },
}

/// An infallible sink keeps ENGINE-0 focused on lifecycle semantics.
/// Sink failures and network backpressure arrive in later milestones.
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

/// Determines whether an admitted request should stop as cancelled.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceKind {
    Admitted,
    ExecutionStarted,
    ModelInvoked { step: usize },
    TokenSelected { step: usize, token_id: u32 },
    TokenEmitted { index: usize, token_id: u32 },
    Terminal(TerminalOutcome),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub at: Duration,
    pub request_id: RequestId,
    pub kind: TraceKind,
}

pub trait TraceSink {
    fn record(&mut self, event: TraceEvent);
}

#[derive(Debug, Default)]
pub struct NoopTrace;

impl TraceSink for NoopTrace {
    fn record(&mut self, _event: TraceEvent) {}
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RecordingTrace {
    pub events: Vec<TraceEvent>,
}

impl TraceSink for RecordingTrace {
    fn record(&mut self, event: TraceEvent) {
        self.events.push(event);
    }
}

/// Named lifecycle points measured from entry to `Runtime::generate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleTimings {
    pub admitted_at: Option<Duration>,
    pub execution_started_at: Option<Duration>,
    pub first_token_ready_at: Option<Duration>,
    pub first_token_emitted_at: Option<Duration>,
    pub terminal_at: Duration,
}

impl LifecycleTimings {
    pub fn queue_delay(&self) -> Option<Duration> {
        Some(self.execution_started_at?.saturating_sub(self.admitted_at?))
    }

    pub fn time_to_first_token(&self) -> Option<Duration> {
        Some(
            self.first_token_emitted_at?
                .saturating_sub(self.admitted_at?),
        )
    }

    pub fn ready_to_emit_delay(&self) -> Option<Duration> {
        Some(
            self.first_token_emitted_at?
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

/// Owns the model/selector policy and advances one request synchronously.
#[derive(Debug, Clone)]
pub struct Runtime<M, S> {
    model: M,
    selector: S,
}

impl<M, S> Runtime<M, S>
where
    M: Model,
    S: Selector,
{
    pub fn new(model: M, selector: S) -> Self {
        Self { model, selector }
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

        if let Err(error) = request.validate() {
            return lifecycle.result(state, TerminalOutcome::Failed(error));
        }

        lifecycle.trace(TraceKind::Admitted);
        lifecycle.admitted_at = Some(started.elapsed());
        lifecycle.trace(TraceKind::ExecutionStarted);
        lifecycle.execution_started_at = Some(started.elapsed());

        let outcome = loop {
            if cancellation.is_cancelled(&state) {
                break TerminalOutcome::Cancelled;
            }
            if state.step() >= request.max_new_tokens {
                break TerminalOutcome::Completed(StopReason::MaxTokens);
            }

            lifecycle.trace(TraceKind::ModelInvoked { step: state.step() });
            let candidates = match self.model.candidates(&request, &state) {
                Ok(candidates) => candidates,
                Err(error) => break TerminalOutcome::Failed(GenerationError::Model(error.message)),
            };
            let selected = match self.selector.select(&candidates) {
                Ok(token) => token,
                Err(error) => break TerminalOutcome::Failed(error),
            };
            lifecycle.trace(TraceKind::TokenSelected {
                step: state.step(),
                token_id: selected.id,
            });

            if selected.kind == TokenKind::EndOfSequence {
                break TerminalOutcome::Completed(StopReason::EndOfSequence);
            }

            if lifecycle.first_token_ready_at.is_none() {
                lifecycle.first_token_ready_at = Some(started.elapsed());
            }
            let index = state.generated.len();
            state.generated.push(selected.clone());
            lifecycle.emit_token(index, selected);
        };

        lifecycle.result(state, outcome)
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
    first_token_emitted_at: Option<Duration>,
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
            first_token_emitted_at: None,
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
        let token_id = token.id;
        self.sink.send(StreamEvent::Token {
            request_id: self.request_id,
            index,
            token,
        });
        if self.first_token_emitted_at.is_none() {
            self.first_token_emitted_at = Some(self.started.elapsed());
        }
        self.trace(TraceKind::TokenEmitted { index, token_id });
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
                first_token_emitted_at: self.first_token_emitted_at,
                terminal_at,
            },
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

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
    fn token_and_trace_emission_are_blocked_after_terminal() {
        let started = Instant::now();
        let mut sink = RecordingSink::default();
        let mut trace = RecordingTrace::default();
        let mut lifecycle = Lifecycle::new(8, started, &mut sink, &mut trace);
        lifecycle
            .finish(TerminalOutcome::Cancelled)
            .expect("first terminal");
        lifecycle.emit_token(0, Token::text(1, "forbidden"));
        lifecycle.trace(TraceKind::ExecutionStarted);

        assert_eq!(sink.events.len(), 1);
        assert_eq!(trace.events.len(), 1);
    }
}
