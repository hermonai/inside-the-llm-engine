use engine0::chat::ModelContract;
use engine0::model::{ForwardPass, Logits, Model, ModelError, TinyLanguageModel};
use engine0::sampling::SamplingConfig;
use engine0::tokenizer::{
    SpecialToken, TinyLmTokenizer, TokenId, Tokenizer, TokenizerError, TokenizerIdentity,
    TINY_LM_EOS, TINY_LM_LIKE, TINY_LM_RUST,
};
use engine0::{
    CancelAtStep, Cancellation, GenerationError, GenerationState, NeverCancel, RecordingSink,
    RecordingTrace, Request, Runtime, StopReason, StreamEvent, TerminalOutcome, Token, TraceKind,
};
use std::cell::Cell;

fn request(id: u64, text: &[u8], max: usize) -> Request {
    Request::from_text(id, text, max, &TinyLmTokenizer).expect("ENGINE-1 fixture encode")
}

fn runtime<M: Model>(model: M) -> Runtime<M, TinyLmTokenizer> {
    Runtime::try_new(model, TinyLmTokenizer, &ModelContract::engine1())
        .expect("matching ENGINE-1 contract")
}

fn run_fixture(request: Request) -> (engine0::GenerationResult, RecordingSink, RecordingTrace) {
    let runtime = runtime(TinyLanguageModel::chapter3_fixture());
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(request, &NeverCancel, &mut sink, &mut trace);
    (result, sink, trace)
}

fn terminal_count(events: &[StreamEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, StreamEvent::Terminal { .. }))
        .count()
}

fn close_vector(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= 1e-6 + 1e-6 * e.abs())
}

#[test]
fn numerical_model_selects_rust_then_eos() {
    let (result, sink, _) = run_fixture(request(1, b"I like", 8));
    assert_eq!(result.tokens, vec![Token::text(TINY_LM_RUST)]);
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
    assert!(sink.events.iter().any(|event| matches!(
        event,
        StreamEvent::Text { text, .. } if text == " Rust"
    )));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn trace_exposes_embedding_logits_selection_and_text_stages() {
    let (_, _, trace) = run_fixture(request(2, b"I like", 8));
    let kinds: Vec<&TraceKind> = trace.events.iter().map(|event| &event.kind).collect();

    assert!(matches!(
        kinds[0],
        TraceKind::InputEncoded {
            bytes: 6,
            token_count: 2
        }
    ));
    assert!(matches!(kinds[1], TraceKind::Admitted));
    assert!(matches!(kinds[2], TraceKind::ExecutionStarted));
    assert!(matches!(
        kinds[3],
        TraceKind::SamplerConfigured(SamplingConfig::Greedy)
    ));
    assert!(matches!(kinds[4], TraceKind::ModelInvoked { step: 0 }));
    assert!(matches!(
        kinds[5],
        TraceKind::EmbeddingReady { input_token, shape: [3], values }
            if *input_token == TINY_LM_LIKE && values == &[1.0, -0.5, 2.0]
    ));
    assert!(matches!(
        kinds[6],
        TraceKind::LogitsReady { shape: [4], values }
            if close_vector(values, &[-0.7, 0.1, 0.4, 2.2])
    ));
    assert!(matches!(
        kinds[7],
        TraceKind::TokenSelected { token_id, .. } if *token_id == TINY_LM_RUST
    ));
    assert!(matches!(
        kinds[8],
        TraceKind::TokenCommitted { index: 0, token_id } if *token_id == TINY_LM_RUST
    ));
    assert!(matches!(kinds[9], TraceKind::TokenEmitted { .. }));
    assert!(matches!(
        kinds[10],
        TraceKind::TokenDecoded { bytes: 5, .. }
    ));
    assert!(matches!(kinds[12], TraceKind::TextEmitted { bytes: 5, .. }));
    assert!(matches!(kinds[13], TraceKind::ModelInvoked { step: 1 }));
    assert!(matches!(
        kinds.last(),
        Some(TraceKind::Terminal(TerminalOutcome::Completed(
            StopReason::EndOfSequence
        )))
    ));
}

struct ConstantModel {
    logits: [f32; 4],
}

impl Model for ConstantModel {
    fn vocabulary_size(&self) -> usize {
        4
    }

    fn forward(&self, input: &[TokenId]) -> Result<ForwardPass, ModelError> {
        let input_token = *input.last().ok_or(ModelError::EmptyInput)?;
        let model =
            TinyLanguageModel::try_new(4, 1, vec![1.0; 4], self.logits.to_vec(), vec![0.0; 4])?;
        let mut pass = model.forward(&[TokenId(0)])?;
        pass.input_token = input_token;
        Ok(pass)
    }
}

#[test]
fn max_token_stop_is_explicit_and_terminal_is_exactly_once() {
    let runtime = runtime(ConstantModel {
        logits: [-1.0, -1.0, -1.0, 2.0],
    });
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(3, b"I like", 2),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert_eq!(result.tokens.len(), 2);
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::MaxTokens)
    );
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn cancellation_emits_no_later_token_and_one_terminal() {
    let runtime = runtime(ConstantModel {
        logits: [-1.0, -1.0, -1.0, 2.0],
    });
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(4, b"I like", 8),
        &CancelAtStep(1),
        &mut sink,
        &mut trace,
    );
    assert_eq!(result.tokens.len(), 1);
    assert_eq!(result.outcome, TerminalOutcome::Cancelled);
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn empty_encoded_input_fails_without_admission_or_token() {
    let (result, sink, trace) = run_fixture(request(5, b"", 8));
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::InvalidRequest(_))
    ));
    assert!(result.tokens.is_empty());
    assert_eq!(terminal_count(&sink.events), 1);
    assert_eq!(trace.events.len(), 1);
}

struct FailingModel;

impl Model for FailingModel {
    fn vocabulary_size(&self) -> usize {
        4
    }

    fn forward(&self, _input: &[TokenId]) -> Result<ForwardPass, ModelError> {
        Err(ModelError::InjectedFailure("test failure".to_string()))
    }
}

#[test]
fn model_failure_cannot_also_complete() {
    let runtime = runtime(FailingModel);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(6, b"I like", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Model(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
}

struct WrongLogitCountModel;

impl Model for WrongLogitCountModel {
    fn vocabulary_size(&self) -> usize {
        4
    }

    fn forward(&self, input: &[TokenId]) -> Result<ForwardPass, ModelError> {
        Ok(ForwardPass {
            input_token: *input.last().ok_or(ModelError::EmptyInput)?,
            hidden: vec![0.0],
            logits: Logits::try_from_values(vec![0.0; 3])?,
        })
    }
}

#[test]
fn runtime_rejects_wrong_logit_vector_length() {
    let runtime = runtime(WrongLogitCountModel);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(61, b"I like", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Model(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn repeated_runs_have_identical_semantic_streams() {
    let (_, first, _) = run_fixture(request(7, b"I like", 8));
    let (_, second, _) = run_fixture(request(7, b"I like", 8));
    assert_eq!(first.events, second.events);
}

#[test]
fn terminal_event_is_always_last() {
    let (_, sink, _) = run_fixture(request(8, b"I like", 8));
    let terminal_index = sink
        .events
        .iter()
        .position(|event| matches!(event, StreamEvent::Terminal { .. }))
        .expect("terminal event");
    assert_eq!(terminal_index + 1, sink.events.len());
}

#[derive(Clone)]
struct PieceTokenizer {
    pieces: Vec<(TokenId, Vec<u8>)>,
    eos: TokenId,
}

impl Tokenizer for PieceTokenizer {
    fn identity(&self) -> TokenizerIdentity {
        TokenizerIdentity {
            name: "test-pieces",
            revision: "v1",
        }
    }

    fn vocabulary_size(&self) -> usize {
        4
    }

    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError> {
        Ok(input.iter().map(|byte| TokenId(u32::from(*byte))).collect())
    }

    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError> {
        self.pieces
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or(TokenizerError::InvalidTokenId(id))
    }

    fn special_id(&self, token: SpecialToken) -> Option<TokenId> {
        (token == SpecialToken::Eos).then_some(self.eos)
    }
}

fn piece_contract() -> ModelContract {
    ModelContract {
        model: "piece-test-model",
        tokenizer: "test-pieces",
        tokenizer_revision: "v1",
        chat_template: "none",
        vocabulary_size: 4,
    }
}

#[test]
fn incomplete_utf8_at_successful_stop_becomes_failed_terminal() {
    let partial = TokenId(1);
    let runtime = Runtime::try_new(
        ConstantModel {
            logits: [-1.0, 2.0, -1.0, -1.0],
        },
        PieceTokenizer {
            pieces: vec![(partial, vec![0xf0, 0x9f])],
            eos: TINY_LM_EOS,
        },
        &piece_contract(),
    )
    .unwrap();
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::from_token_ids(9, vec![TINY_LM_LIKE], 1, 1),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Utf8Stream(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn tokenizer_model_vocabulary_mismatch_is_rejected_before_execution() {
    let model = TinyLanguageModel::try_new(3, 1, vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]).unwrap();
    let result = Runtime::try_new(model, TinyLmTokenizer, &ModelContract::engine1());
    assert!(matches!(
        result,
        Err(engine0::chat::ContractError::VocabularyMismatch { .. })
    ));
}

#[test]
fn eos_is_terminal_and_never_decoded_as_text() {
    let runtime = runtime(ConstantModel {
        logits: [2.0, -1.0, -1.0, -1.0],
    });
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(10, b"I like", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert!(result.tokens.is_empty());
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
    assert!(!sink
        .events
        .iter()
        .any(|event| matches!(event, StreamEvent::Text { .. })));
}

#[test]
fn invalid_sampler_config_fails_once_without_admission_or_model_work() {
    let runtime = runtime(TinyLanguageModel::chapter3_fixture());
    let bad =
        request(11, b"I like", 8).with_sampling(SamplingConfig::stochastic(-1.0, None, None, 7));
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(bad, &NeverCancel, &mut sink, &mut trace);
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Sampling(_))
    ));
    assert!(result.tokens.is_empty());
    assert_eq!(terminal_count(&sink.events), 1);
    assert_eq!(trace.events.len(), 1);
}

#[test]
fn stochastic_trace_exposes_probabilities_draw_and_selection() {
    let runtime = runtime(TinyLanguageModel::chapter3_fixture());
    let stochastic = request(12, b"I like", 1).with_sampling(SamplingConfig::stochastic(
        1.0,
        Some(3),
        Some(0.9),
        42,
    ));
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(stochastic, &NeverCancel, &mut sink, &mut trace);
    assert_eq!(result.tokens.len(), 1);
    assert!(trace.events.iter().any(|event| matches!(
        &event.kind,
        TraceKind::ProbabilitiesReady { values }
            if values.len() == 4 && (values.iter().sum::<f64>() - 1.0).abs() < 1e-12
    )));
    assert!(trace.events.iter().any(|event| matches!(
        event.kind,
        TraceKind::RandomDraw { sample_index: 0, value } if (0.0..1.0).contains(&value)
    )));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn seeded_sampling_is_request_local_and_repeats_independently() {
    let runtime = runtime(ConstantModel {
        logits: [-100.0, 0.0, 0.1, 0.2],
    });
    let run = |id, seed| {
        let mut sink = RecordingSink::default();
        let mut trace = RecordingTrace::default();
        runtime.generate(
            request(id, b"I like", 6)
                .with_sampling(SamplingConfig::stochastic(1.0, None, None, seed)),
            &NeverCancel,
            &mut sink,
            &mut trace,
        )
    };
    let first = run(13, 99);
    let _interleaved_other_seed = run(14, 1234);
    let repeated = run(15, 99);
    assert_eq!(first.tokens, repeated.tokens);
    assert_eq!(first.outcome, repeated.outcome);
}

struct CancelOnSecondCheck {
    checks: Cell<usize>,
}

impl Cancellation for CancelOnSecondCheck {
    fn is_cancelled(&self, _state: &GenerationState) -> bool {
        let check = self.checks.get();
        self.checks.set(check + 1);
        check >= 1
    }
}

#[test]
fn cancellation_after_forward_prevents_sampling_and_commit() {
    let runtime = runtime(TinyLanguageModel::chapter3_fixture());
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let cancellation = CancelOnSecondCheck {
        checks: Cell::new(0),
    };
    let result = runtime.generate(
        request(16, b"I like", 8),
        &cancellation,
        &mut sink,
        &mut trace,
    );
    assert_eq!(result.outcome, TerminalOutcome::Cancelled);
    assert!(result.tokens.is_empty());
    assert!(!trace
        .events
        .iter()
        .any(|event| matches!(event.kind, TraceKind::TokenSelected { .. })));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn tokenizer_failure_after_commit_still_has_one_failed_terminal() {
    let missing_piece = TokenId(2);
    let runtime = Runtime::try_new(
        ConstantModel {
            logits: [-1.0, -1.0, 2.0, -1.0],
        },
        PieceTokenizer {
            pieces: vec![(TokenId(1), b"x".to_vec())],
            eos: TINY_LM_EOS,
        },
        &piece_contract(),
    )
    .unwrap();
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::from_token_ids(17, vec![TINY_LM_LIKE], 1, 2),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert_eq!(result.tokens, vec![Token::text(missing_piece)]);
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Tokenizer(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
    assert!(matches!(
        sink.events.last(),
        Some(StreamEvent::Terminal { .. })
    ));
}
