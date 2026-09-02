use std::process::ExitCode;

use engine0::tokenizer::{TinyBpeTokenizer, TokenId, Tokenizer};
use engine0::utf8::Utf8StreamDecoder;
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
                    "stream token[{index}] id={} kind={:?}",
                    token.id, token.kind
                );
            }
            StreamEvent::Text {
                through_token_index,
                text,
                ..
            } => println!("stream text[through={through_token_index}] {text:?}"),
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
struct GenerateOptions {
    prompt: String,
    max_tokens: usize,
    cancel_at: Option<usize>,
    fail_at: Option<usize>,
    trace: bool,
}

fn usage() -> &'static str {
    "usage:\n  engine0 [--trace] [--max-tokens N] [--cancel-at STEP] [--fail-at STEP] PROMPT\n  engine0 tokenize TEXT\n  engine0 decode TOKEN_ID..."
}

fn parse_generate(args: Vec<String>) -> Result<GenerateOptions, String> {
    let mut args = args.into_iter();
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

    Ok(GenerateOptions {
        prompt: prompt_parts.join(" "),
        max_tokens,
        cancel_at,
        fail_at,
        trace,
    })
}

fn tokenize(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("tokenize requires text".to_string());
    }
    let text = args.join(" ");
    let tokenizer = TinyBpeTokenizer::teaching();
    let ids = tokenizer
        .encode(text.as_bytes())
        .map_err(|error| error.to_string())?;
    println!(
        "tokenizer={}@{} bytes={} scalars={} tokens={}",
        tokenizer.identity().name,
        tokenizer.identity().revision,
        text.len(),
        text.chars().count(),
        ids.len()
    );
    for (index, id) in ids.iter().copied().enumerate() {
        let bytes = tokenizer
            .decode_token(id)
            .map_err(|error| error.to_string())?;
        println!("token[{index}] id={id} bytes={bytes:02x?}");
    }
    Ok(())
}

fn decode(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("decode requires one or more numeric token IDs".to_string());
    }
    let tokenizer = TinyBpeTokenizer::teaching();
    let mut decoder = Utf8StreamDecoder::new();
    for (index, value) in args.iter().enumerate() {
        let id = TokenId(
            value
                .parse()
                .map_err(|_| format!("invalid token id: {value}"))?,
        );
        let bytes = tokenizer
            .decode_token(id)
            .map_err(|error| error.to_string())?;
        println!("token[{index}] id={id} bytes={bytes:02x?}");
        if let Some(text) = decoder.push(bytes).map_err(|error| error.to_string())? {
            println!("text[through={index}] {text:?}");
        } else {
            println!("buffered={} bytes", decoder.pending_bytes().len());
        }
    }
    decoder.finish().map_err(|error| error.to_string())
}

fn generate(options: GenerateOptions) -> Result<TerminalOutcome, String> {
    let tokenizer = TinyBpeTokenizer::teaching();
    let request = Request::from_text(1, options.prompt.as_bytes(), options.max_tokens, &tokenizer)
        .map_err(|error| error.to_string())?;
    let model = options
        .fail_at
        .map(DemoModel::failing_at)
        .unwrap_or_default();
    let runtime = Runtime::new(model, GreedySelector, tokenizer);
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
        "timing queue={:?} ttft={:?} first_text={:?} token_to_text={:?} total={:?}",
        result.timings.queue_delay(),
        result.timings.time_to_first_token(),
        result.timings.time_to_first_text(),
        result.timings.token_ready_to_text_delay(),
        result.timings.terminal_at,
    );
    Ok(result.outcome)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("tokenize") => tokenize(&args[1..]).map(|_| None),
        Some("decode") => decode(&args[1..]).map(|_| None),
        _ => parse_generate(args).and_then(generate).map(Some),
    };

    match result {
        Ok(Some(TerminalOutcome::Failed(_))) => ExitCode::from(1),
        Ok(_) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}
