use engine0::tokenizer::{
    TinyBpeTokenizer, TokenId, Tokenizer, TokenizerError, TokenizerIdentity, TOKEN_BLUE, TOKEN_EOS,
};
use engine0::{
    CancelAtStep, Candidate, DemoModel, GenerationError, GenerationState, GreedySelector, Model,
    ModelError, NeverCancel, RecordingSink, RecordingTrace, Request, Runtime, StopReason,
    StreamEvent, TerminalOutcome, Token, TraceKind,
};

fn request(id: u64, text: &[u8], max: usize) -> Request {
    Request::from_text(id, text, max, &TinyBpeTokenizer::teaching()).expect("teaching encode")
}

fn run_demo(request: Request) -> (engine0::GenerationResult, RecordingSink, RecordingTrace) {
    let runtime = Runtime::new(
        DemoModel::default(),
        GreedySelector,
        TinyBpeTokenizer::teaching(),
    );
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

#[test]
fn hand_computable_oracle_selects_blue_then_eos() {
    let (result, sink, _) = run_demo(request(1, b"What color?", 8));

    assert_eq!(result.tokens, vec![Token::text(TOKEN_BLUE)]);
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
    assert!(sink.events.iter().any(|event| matches!(
        event,
        StreamEvent::Text { text, .. } if text == "blue"
    )));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn trace_exposes_encode_token_decode_buffer_and_text_stages() {
    let (_, _, trace) = run_demo(request(2, b"order", 8));
    let kinds: Vec<&TraceKind> = trace.events.iter().map(|event| &event.kind).collect();

    assert!(matches!(
        kinds[0],
        TraceKind::InputEncoded {
            bytes: 5,
            token_count: 4
        }
    ));
    assert!(matches!(kinds[1], TraceKind::Admitted));
    assert!(matches!(kinds[2], TraceKind::ExecutionStarted));
    assert!(matches!(kinds[3], TraceKind::ModelInvoked { step: 0 }));
    assert!(matches!(
        kinds[4],
        TraceKind::TokenSelected { token_id, .. } if *token_id == TOKEN_BLUE
    ));
    assert!(matches!(kinds[5], TraceKind::TokenEmitted { .. }));
    assert!(matches!(kinds[6], TraceKind::TokenDecoded { bytes: 4, .. }));
    assert!(matches!(
        kinds[7],
        TraceKind::Utf8Buffered { pending_bytes: 0 }
    ));
    assert!(matches!(kinds[8], TraceKind::TextEmitted { bytes: 4, .. }));
    assert!(matches!(kinds[9], TraceKind::ModelInvoked { step: 1 }));
    assert!(matches!(
        kinds.last(),
        Some(TraceKind::Terminal(TerminalOutcome::Completed(
            StopReason::EndOfSequence
        )))
    ));
}

struct NeverEndingModel;

impl Model for NeverEndingModel {
    fn candidates(
        &self,
        _request: &Request,
        _state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError> {
        Ok(vec![Candidate::new(Token::text(TOKEN_BLUE), 1)])
    }
}

#[test]
fn max_token_stop_is_explicit_and_terminal_is_exactly_once() {
    let runtime = Runtime::new(
        NeverEndingModel,
        GreedySelector,
        TinyBpeTokenizer::teaching(),
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(3, b"bounded", 2),
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
    let runtime = Runtime::new(
        NeverEndingModel,
        GreedySelector,
        TinyBpeTokenizer::teaching(),
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(4, b"cancel me", 8),
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
    let (result, sink, trace) = run_demo(request(5, b"", 8));

    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::InvalidRequest(_))
    ));
    assert!(result.tokens.is_empty());
    assert_eq!(terminal_count(&sink.events), 1);
    assert_eq!(trace.events.len(), 1);
}

#[test]
fn whitespace_is_real_input_not_a_blank_string_error() {
    let (result, _, _) = run_demo(request(6, b" \t\n", 8));
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
}

#[test]
fn model_failure_after_a_token_cannot_also_complete() {
    let runtime = Runtime::new(
        DemoModel::failing_at(1),
        GreedySelector,
        TinyBpeTokenizer::teaching(),
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(request(7, b"fail", 8), &NeverCancel, &mut sink, &mut trace);

    assert_eq!(result.tokens, vec![Token::text(TOKEN_BLUE)]);
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Model(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn empty_candidate_set_fails_explicitly() {
    struct EmptyModel;
    impl Model for EmptyModel {
        fn candidates(
            &self,
            _request: &Request,
            _state: &GenerationState,
        ) -> Result<Vec<Candidate>, ModelError> {
            Ok(Vec::new())
        }
    }

    let runtime = Runtime::new(EmptyModel, GreedySelector, TinyBpeTokenizer::teaching());
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(request(8, b"empty", 8), &NeverCancel, &mut sink, &mut trace);

    assert_eq!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::NoCandidates)
    );
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn repeated_runs_have_identical_semantic_streams() {
    let (_, first, _) = run_demo(request(9, b"repeat", 8));
    let (_, second, _) = run_demo(request(9, b"repeat", 8));
    assert_eq!(first.events, second.events);
}

#[test]
fn terminal_event_is_always_last() {
    let (_, sink, _) = run_demo(request(10, b"terminal", 8));
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
}

impl Tokenizer for PieceTokenizer {
    fn identity(&self) -> TokenizerIdentity {
        TokenizerIdentity {
            name: "test-pieces",
            revision: "v1",
        }
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

    fn special_id(&self, _token: engine0::tokenizer::SpecialToken) -> Option<TokenId> {
        None
    }
}

struct SequenceModel {
    tokens: Vec<Token>,
}

impl Model for SequenceModel {
    fn candidates(
        &self,
        _request: &Request,
        state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError> {
        Ok(vec![Candidate::new(self.tokens[state.step()], 1)])
    }
}

#[test]
fn two_token_byte_fragments_emit_one_valid_scalar() {
    let first = TokenId(2000);
    let second = TokenId(2001);
    let tokenizer = PieceTokenizer {
        pieces: vec![(first, vec![0xe4, 0xb8]), (second, vec![0x96])],
    };
    let model = SequenceModel {
        tokens: vec![
            Token::text(first),
            Token::text(second),
            Token::end(TOKEN_EOS),
        ],
    };
    let runtime = Runtime::new(model, GreedySelector, tokenizer);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::from_token_ids(11, vec![TokenId(1)], 1, 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );

    let texts: Vec<&str> = sink
        .events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["世"]);
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
}

#[test]
fn incomplete_utf8_at_successful_stop_becomes_failed_terminal() {
    let partial = TokenId(2000);
    let runtime = Runtime::new(
        SequenceModel {
            tokens: vec![Token::text(partial), Token::end(TOKEN_EOS)],
        },
        GreedySelector,
        PieceTokenizer {
            pieces: vec![(partial, vec![0xf0, 0x9f])],
        },
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::from_token_ids(12, vec![TokenId(1)], 1, 8),
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
fn malformed_output_bytes_fail_without_lossy_replacement() {
    let invalid = TokenId(2000);
    let runtime = Runtime::new(
        SequenceModel {
            tokens: vec![Token::text(invalid)],
        },
        GreedySelector,
        PieceTokenizer {
            pieces: vec![(invalid, vec![0xc3, 0x28])],
        },
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::from_token_ids(13, vec![TokenId(1)], 1, 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );

    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Utf8Stream(_))
    ));
    assert!(!sink.events.iter().any(|event| matches!(
        event,
        StreamEvent::Text { text, .. } if text.contains('\u{fffd}')
    )));
}

#[test]
fn unknown_generated_id_is_a_typed_tokenizer_failure() {
    let unknown = TokenId(9999);
    let runtime = Runtime::new(
        SequenceModel {
            tokens: vec![Token::text(unknown)],
        },
        GreedySelector,
        TinyBpeTokenizer::teaching(),
    );
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        request(14, b"unknown", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Tokenizer(_))
    ));
}
