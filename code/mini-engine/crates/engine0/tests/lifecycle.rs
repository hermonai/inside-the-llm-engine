use engine0::{
    CancelAtStep, Candidate, DemoModel, GenerationError, GenerationState, GreedySelector, Model,
    ModelError, NeverCancel, RecordingSink, RecordingTrace, Request, Runtime, StopReason,
    StreamEvent, TerminalOutcome, Token, TraceKind,
};

fn run_demo(request: Request) -> (engine0::GenerationResult, RecordingSink, RecordingTrace) {
    let runtime = Runtime::new(DemoModel::default(), GreedySelector);
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
    let (result, sink, _) = run_demo(Request::new(1, "What color?", 8));

    assert_eq!(result.tokens, vec![Token::text(1, "blue")]);
    assert_eq!(
        result.outcome,
        TerminalOutcome::Completed(StopReason::EndOfSequence)
    );
    assert_eq!(sink.events.len(), 2);
}

#[test]
fn event_order_is_admit_start_model_select_emit_model_select_terminal() {
    let (_, _, trace) = run_demo(Request::new(2, "order", 8));
    let kinds: Vec<&TraceKind> = trace.events.iter().map(|event| &event.kind).collect();

    assert!(matches!(kinds[0], TraceKind::Admitted));
    assert!(matches!(kinds[1], TraceKind::ExecutionStarted));
    assert!(matches!(kinds[2], TraceKind::ModelInvoked { step: 0 }));
    assert!(matches!(
        kinds[3],
        TraceKind::TokenSelected {
            step: 0,
            token_id: 1
        }
    ));
    assert!(matches!(
        kinds[4],
        TraceKind::TokenEmitted {
            index: 0,
            token_id: 1
        }
    ));
    assert!(matches!(kinds[5], TraceKind::ModelInvoked { step: 1 }));
    assert!(matches!(
        kinds[6],
        TraceKind::TokenSelected {
            step: 1,
            token_id: 0
        }
    ));
    assert!(matches!(
        kinds[7],
        TraceKind::Terminal(TerminalOutcome::Completed(StopReason::EndOfSequence))
    ));
}

struct NeverEndingModel;

impl Model for NeverEndingModel {
    fn candidates(
        &self,
        _request: &Request,
        state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError> {
        Ok(vec![Candidate::new(
            Token::text(10 + state.step() as u32, "tick"),
            1,
        )])
    }
}

#[test]
fn max_token_stop_is_explicit_and_terminal_is_exactly_once() {
    let runtime = Runtime::new(NeverEndingModel, GreedySelector);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::new(3, "bounded", 2),
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
    assert!(matches!(
        sink.events.last(),
        Some(StreamEvent::Terminal { .. })
    ));
}

#[test]
fn cancellation_emits_no_later_token_and_one_terminal() {
    let runtime = Runtime::new(NeverEndingModel, GreedySelector);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::new(4, "cancel me", 8),
        &CancelAtStep(1),
        &mut sink,
        &mut trace,
    );

    assert_eq!(result.tokens.len(), 1);
    assert_eq!(result.outcome, TerminalOutcome::Cancelled);
    assert_eq!(terminal_count(&sink.events), 1);
    assert!(matches!(
        sink.events.last(),
        Some(StreamEvent::Terminal { .. })
    ));
}

#[test]
fn invalid_request_fails_without_admission_or_token() {
    let (result, sink, trace) = run_demo(Request::new(5, "   ", 8));

    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::InvalidRequest(_))
    ));
    assert!(result.tokens.is_empty());
    assert_eq!(terminal_count(&sink.events), 1);
    assert_eq!(trace.events.len(), 1);
    assert!(matches!(
        trace.events[0].kind,
        TraceKind::Terminal(TerminalOutcome::Failed(_))
    ));
}

#[test]
fn model_failure_after_a_token_cannot_also_complete() {
    let runtime = Runtime::new(DemoModel::failing_at(1), GreedySelector);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::new(6, "fail", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );

    assert_eq!(result.tokens, vec![Token::text(1, "blue")]);
    assert!(matches!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::Model(_))
    ));
    assert_eq!(terminal_count(&sink.events), 1);
    assert!(matches!(
        sink.events.last(),
        Some(StreamEvent::Terminal { .. })
    ));
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

    let runtime = Runtime::new(EmptyModel, GreedySelector);
    let mut sink = RecordingSink::default();
    let mut trace = RecordingTrace::default();
    let result = runtime.generate(
        Request::new(7, "empty", 8),
        &NeverCancel,
        &mut sink,
        &mut trace,
    );

    assert_eq!(
        result.outcome,
        TerminalOutcome::Failed(GenerationError::NoCandidates)
    );
    assert_eq!(terminal_count(&sink.events), 1);
}

#[test]
fn repeated_runs_have_identical_semantic_streams() {
    let (_, first, _) = run_demo(Request::new(8, "repeat", 8));
    let (_, second, _) = run_demo(Request::new(8, "repeat", 8));
    assert_eq!(first.events, second.events);
}

#[test]
fn nothing_is_emitted_after_any_terminal() {
    let (_, sink, _) = run_demo(Request::new(9, "terminal", 8));
    let terminal_index = sink
        .events
        .iter()
        .position(|event| matches!(event, StreamEvent::Terminal { .. }))
        .expect("terminal event");
    assert_eq!(terminal_index + 1, sink.events.len());
}
