use std::process::ExitCode;

use engine0::{
    CancelAtStep, DemoModel, GreedySelector, NeverCancel, Request, Runtime, StreamEvent,
    TerminalOutcome, TokenSink, TraceEvent, TraceSink,
};

struct ConsoleSink;

impl TokenSink for ConsoleSink {
    fn send(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Token { index, token, .. } => {
                println!(
                    "stream token[{index}] id={} text={:?}",
                    token.id, token.text
                );
            }
            StreamEvent::Terminal { outcome, .. } => println!("stream terminal {outcome}"),
        }
    }
}

struct ConsoleTrace;

impl TraceSink for ConsoleTrace {
    fn record(&mut self, event: TraceEvent) {
        eprintln!(
            "trace +{:>6}us request={} {:?}",
            event.at.as_micros(),
            event.request_id,
            event.kind
        );
    }
}

struct SilentTrace;

impl TraceSink for SilentTrace {
    fn record(&mut self, _event: TraceEvent) {}
}

#[derive(Debug)]
struct Options {
    prompt: String,
    max_tokens: usize,
    cancel_at: Option<usize>,
    fail_at: Option<usize>,
    trace: bool,
}

fn usage() -> &'static str {
    "usage: engine0 [--trace] [--max-tokens N] [--cancel-at STEP] [--fail-at STEP] PROMPT"
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut prompt_parts = Vec::new();
    let mut max_tokens = 8usize;
    let mut cancel_at = None;
    let mut fail_at = None;
    let mut trace = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trace" => trace = true,
            "--max-tokens" => {
                let value = args.next().ok_or("--max-tokens requires a value")?;
                max_tokens = value
                    .parse()
                    .map_err(|_| format!("invalid --max-tokens value: {value}"))?;
            }
            "--cancel-at" => {
                let value = args.next().ok_or("--cancel-at requires a value")?;
                cancel_at = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --cancel-at value: {value}"))?,
                );
            }
            "--fail-at" => {
                let value = args.next().ok_or("--fail-at requires a value")?;
                fail_at = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --fail-at value: {value}"))?,
                );
            }
            "-h" | "--help" => return Err(usage().to_string()),
            value if value.starts_with('-') => return Err(format!("unknown option: {value}")),
            value => prompt_parts.push(value.to_string()),
        }
    }

    if prompt_parts.is_empty() {
        return Err(usage().to_string());
    }

    Ok(Options {
        prompt: prompt_parts.join(" "),
        max_tokens,
        cancel_at,
        fail_at,
        trace,
    })
}

fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let model = options
        .fail_at
        .map(DemoModel::failing_at)
        .unwrap_or_default();
    let runtime = Runtime::new(model, GreedySelector);
    let request = Request::new(1, options.prompt, options.max_tokens);
    let mut sink = ConsoleSink;
    let mut console_trace = ConsoleTrace;
    let mut silent_trace = SilentTrace;
    let trace: &mut dyn TraceSink = if options.trace {
        &mut console_trace
    } else {
        &mut silent_trace
    };

    let result = if let Some(step) = options.cancel_at {
        runtime.generate(request, &CancelAtStep(step), &mut sink, trace)
    } else {
        runtime.generate(request, &NeverCancel, &mut sink, trace)
    };

    println!(
        "timing queue={:?} ttft={:?} ready_to_emit={:?} total={:?}",
        result.timings.queue_delay(),
        result.timings.time_to_first_token(),
        result.timings.ready_to_emit_delay(),
        result.timings.terminal_at,
    );

    match result.outcome {
        TerminalOutcome::Failed(_) => ExitCode::from(1),
        TerminalOutcome::Completed(_) | TerminalOutcome::Cancelled => ExitCode::SUCCESS,
    }
}
