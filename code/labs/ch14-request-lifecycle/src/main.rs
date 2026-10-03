//! Chapter 14 lab: the terminal contract, and cancellation that does or does
//! not reach the model.
//!
//! Part 1 is a small server in one process. An engine thread steps every
//! running request once per tick --- a decode step --- and delivers results to
//! one queue that wakes every waiting handler, as llama.cpp's server does
//! (`server_response::send` and `recv_with_timeout` in `tools/server`, commit
//! d006858). Handler threads stand in for HTTP threads. A streamed request gets
//! a result per token, and its handler writes each one to the client; a
//! non-streamed request gets one final result, and its handler waits for it.
//! Two ways for that handler to notice that its client has gone:
//!
//! - `poll`: wait for a result in slices, and check the client only when a
//!   slice expires with no result (llama.cpp's `HTTP_POLLING_SECONDS`). A
//!   result for any request wakes every waiter, and a woken waiter that finds
//!   nothing for itself starts a new slice;
//! - `event`: the disconnect itself wakes the handler, which cancels at once
//!   (vLLM races each request against the server's disconnect message).
//!
//! The program replays the chapter's opening at one-tenth of its time scale
//! and counts the tokens made for nobody.
//!
//! Part 2 checks the terminal contract without threads: thousands of random
//! scenarios run through the engine, and every request must end exactly once,
//! with nothing after its end, its streamed tokens in order and complete, the
//! outcome an independent simulation predicts, and no token after a
//! cancellation.
//!
//!     cargo run --release -p ch14-request-lifecycle -- [--tick-ms 4] [--slice-ms 100]
//!         [--close-ms 350] [--scenarios 10000] [--seed 1]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod ownership;

pub type Id = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    EndOfSequence,
    MaxTokens,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// One streamed token and its index in the request's output.
    Token { id: Id, index: usize },
    /// The terminal event: how the request ended, after how many tokens.
    Final {
        id: Id,
        tokens: usize,
        outcome: Outcome,
    },
    /// A reply from the status endpoint to the monitor that asked.
    Status { id: Id },
}

impl Event {
    pub fn id(&self) -> Id {
        match *self {
            Event::Token { id, .. } | Event::Final { id, .. } | Event::Status { id } => id,
        }
    }
}

/// What a request asks for, and what its model will do: select EOS before
/// token `eos_at`, or fail before token `fail_at`.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub budget: usize,
    pub eos_at: Option<usize>,
    pub fail_at: Option<usize>,
    pub stream: bool,
}

struct Running {
    id: Id,
    spec: Spec,
    made: usize,
}

/// The model loop without threads or clocks. Requests are admitted and
/// cancelled between steps; each step advances every running request by one
/// token or ends it, checking in the chapter's order: cancellation, then the
/// model (which may fail or select EOS), then the budget.
#[derive(Default)]
pub struct Engine {
    running: Vec<Running>,
    cancels: HashSet<Id>,
}

impl Engine {
    pub fn admit(&mut self, id: Id, spec: Spec) {
        assert!(spec.budget > 0, "a request must allow at least one token");
        assert!(
            self.made(id).is_none(),
            "an active request identity cannot be admitted twice"
        );
        self.running.push(Running { id, spec, made: 0 });
    }

    /// Takes effect at the next step, before the request's next token.
    pub fn cancel(&mut self, id: Id) {
        self.cancels.insert(id);
    }

    pub fn made(&self, id: Id) -> Option<usize> {
        self.running.iter().find(|r| r.id == id).map(|r| r.made)
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_empty()
    }

    pub fn step(&mut self, out: &mut Vec<Event>) {
        let cancels = &mut self.cancels;
        self.running.retain_mut(|r| {
            let end = |tokens, outcome| Event::Final {
                id: r.id,
                tokens,
                outcome,
            };
            if cancels.remove(&r.id) {
                out.push(end(r.made, Outcome::Cancelled));
                return false;
            }
            if r.spec.fail_at == Some(r.made) {
                out.push(end(r.made, Outcome::Failed));
                return false;
            }
            if r.spec.eos_at == Some(r.made) {
                out.push(end(r.made, Outcome::EndOfSequence));
                return false;
            }
            r.made += 1;
            if r.spec.stream {
                out.push(Event::Token {
                    id: r.id,
                    index: r.made - 1,
                });
            }
            if r.made == r.spec.budget {
                out.push(end(r.made, Outcome::MaxTokens));
                return false;
            }
            true
        });
        // A cancellation for a request that already ended has nothing to stop.
        cancels.clear();
    }
}

/// The outcome a request must reach if it is not cancelled, simulated token by
/// token without the engine: the oracle for Part 2.
pub fn natural_end(spec: &Spec) -> (usize, Outcome) {
    let mut k = 0;
    loop {
        if spec.fail_at == Some(k) {
            return (k, Outcome::Failed);
        }
        if spec.eos_at == Some(k) {
            return (k, Outcome::EndOfSequence);
        }
        k += 1;
        if k == spec.budget {
            return (k, Outcome::MaxTokens);
        }
    }
}

/// Checks the terminal contract for one request's events, in order: exactly
/// one Final, and nothing after it; streamed tokens numbered 0, 1, 2, ... and
/// as many as the Final reports; no tokens for a non-streamed request; and the
/// outcome the oracle predicts, or a cancellation that stopped the request with
/// no token after it was requested (`cancelled_after` tokens).
pub fn check(
    events: &[Event],
    spec: &Spec,
    cancelled_after: Option<usize>,
) -> Result<Outcome, String> {
    let Some(first) = events.first() else {
        return Err("no events".into());
    };
    if events.iter().any(|e| e.id() != first.id()) {
        return Err("mixed request identities".into());
    }
    if events.iter().any(|e| matches!(e, Event::Status { .. })) {
        return Err("a status reply is not a generation event".into());
    }
    let finals: Vec<usize> = (0..events.len())
        .filter(|&i| matches!(events[i], Event::Final { .. }))
        .collect();
    if finals.len() != 1 {
        return Err(format!("{} terminal events", finals.len()));
    }
    if finals[0] != events.len() - 1 {
        return Err("an event after the terminal event".into());
    }
    let tokens: Vec<usize> = events
        .iter()
        .filter_map(|e| match *e {
            Event::Token { index, .. } => Some(index),
            _ => None,
        })
        .collect();
    let Event::Final {
        tokens: n, outcome, ..
    } = events[finals[0]]
    else {
        unreachable!()
    };
    if spec.stream && tokens != (0..n).collect::<Vec<_>>() {
        return Err(format!(
            "streamed tokens {tokens:?} for a final count of {n}"
        ));
    }
    if !spec.stream && !tokens.is_empty() {
        return Err("tokens streamed for a non-streamed request".into());
    }
    let (end, natural) = natural_end(spec);
    // The harness records cancellation only for a still-running request.
    // A subsequent natural end is therefore wrong, even if its counts match.
    if cancelled_after.is_some() && outcome != Outcome::Cancelled {
        return Err("observed cancellation did not win the next step".into());
    }
    match (outcome, cancelled_after) {
        (Outcome::Cancelled, Some(at)) if n == at => Ok(outcome),
        (Outcome::Cancelled, Some(at)) => {
            Err(format!("cancelled after {n} tokens, {at} made when asked"))
        }
        (Outcome::Cancelled, None) => Err("cancelled without a cancellation".into()),
        _ if (n, outcome) == (end, natural) => Ok(outcome),
        _ => Err(format!(
            "ended {outcome:?} after {n} tokens, expected {natural:?} after {end}"
        )),
    }
}

/// A small xorshift generator, so that every scenario is reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// Counts of outcomes over many scenarios.
#[derive(Debug, Default)]
pub struct Tally {
    pub requests: usize,
    pub outcomes: HashMap<&'static str, usize>,
}

/// One random scenario: up to eight requests arriving at random steps, with
/// random budgets, EOS positions, injected failures and cancellations.
pub fn random_scenario(rng: &mut Rng, tally: &mut Tally) -> Result<(), String> {
    let n = 1 + rng.below(8);
    let mut specs = Vec::new();
    for id in 0..n {
        let spec = Spec {
            budget: 1 + rng.below(40),
            eos_at: rng.chance(50).then(|| rng.below(40)),
            fail_at: rng.chance(15).then(|| rng.below(40)),
            stream: rng.chance(50),
        };
        let arrive = rng.below(20);
        let cancel = rng.chance(30).then(|| arrive + rng.below(40));
        specs.push((id, spec, arrive, cancel));
    }
    let mut engine = Engine::default();
    let mut log = Vec::new();
    let mut cancelled_after: HashMap<Id, usize> = HashMap::new();
    for step in 0.. {
        for &(id, spec, arrive, cancel) in &specs {
            if arrive == step {
                engine.admit(id, spec);
            }
            if cancel == Some(step) {
                // Only a request that is still running can be cancelled.
                if let Some(made) = engine.made(id) {
                    cancelled_after.insert(id, made);
                    engine.cancel(id);
                }
            }
        }
        if engine.is_idle() && step >= 20 {
            break;
        }
        engine.step(&mut log);
    }
    for &(id, spec, _, _) in &specs {
        let events: Vec<Event> = log.iter().copied().filter(|e| e.id() == id).collect();
        let outcome = check(&events, &spec, cancelled_after.get(&id).copied())
            .map_err(|e| format!("request {id} ({spec:?}): {e}"))?;
        let name = match outcome {
            Outcome::EndOfSequence => "end of sequence",
            Outcome::MaxTokens => "token budget",
            Outcome::Cancelled => "cancelled",
            Outcome::Failed => "failed",
        };
        *tally.outcomes.entry(name).or_default() += 1;
        tally.requests += 1;
    }
    Ok(())
}

// Part 1: the same engine behind a server with handler threads.

/// Where results wait for their handlers: one lock, one condition variable and
/// one queue for every request, as in llama.cpp's `server_response`. Results
/// for a request that nobody is waiting for are dropped.
#[derive(Default)]
pub struct Results {
    inner: Mutex<(VecDeque<Event>, HashSet<Id>)>,
    ready: Condvar,
}

impl Results {
    pub fn wait_for(&self, id: Id) {
        self.inner.lock().unwrap().1.insert(id);
    }

    pub fn stop_waiting(&self, id: Id) {
        let mut inner = self.inner.lock().unwrap();
        inner.1.remove(&id);
        inner.0.retain(|e| e.id() != id);
    }

    pub fn send(&self, event: Event) {
        let mut inner = self.inner.lock().unwrap();
        if inner.1.contains(&event.id()) {
            inner.0.push_back(event);
            self.ready.notify_all();
        }
    }

    /// Wakes every waiter without a result: how a disconnect reaches a handler
    /// that waits for events.
    pub fn wake_all(&self) {
        let _inner = self.inner.lock().unwrap();
        self.ready.notify_all();
    }

    /// The poll design: wait up to `slice` for a result for `id`. A result for
    /// any request wakes the wait; if none is ours, wait again, a full slice.
    /// `None` means a whole slice passed with no result for anyone.
    pub fn recv_with_timeout(&self, id: Id, slice: Duration) -> Option<Event> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(i) = inner.0.iter().position(|e| e.id() == id) {
                return inner.0.remove(i);
            }
            let (guard, wait) = self.ready.wait_timeout(inner, slice).unwrap();
            inner = guard;
            if wait.timed_out() {
                return None;
            }
        }
    }

    /// The event design: wait for a result for `id` or for the client to go,
    /// whichever comes first; `None` means the client has gone.
    pub fn recv_or_gone(&self, id: Id, gone: &AtomicBool) -> Option<Event> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(i) = inner.0.iter().position(|e| e.id() == id) {
                return inner.0.remove(i);
            }
            if gone.load(Ordering::SeqCst) {
                return None;
            }
            inner = self.ready.wait(inner).unwrap();
        }
    }
}

pub enum Command {
    Admit(Id, Spec),
    Cancel(Id),
    Status(Id),
}

/// The engine on a thread, one step per tick, with a log of released requests
/// and each running request's token count for the harness to read.
pub struct Server {
    pub tx: mpsc::Sender<Command>,
    pub results: Arc<Results>,
    released: Arc<Mutex<Vec<(Event, Instant)>>>,
    made: Arc<Mutex<HashMap<Id, usize>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Server {
    pub fn start(tick: Duration) -> Server {
        let (tx, rx) = mpsc::channel::<Command>();
        let results = Arc::new(Results::default());
        let released = Arc::new(Mutex::new(Vec::new()));
        let made = Arc::new(Mutex::new(HashMap::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let (results, released, made, stop) = (
                results.clone(),
                released.clone(),
                made.clone(),
                stop.clone(),
            );
            thread::spawn(move || {
                let mut engine = Engine::default();
                let mut ids = Vec::new();
                let mut out = Vec::new();
                // Steps start on a fixed schedule, so a step is `tick` long on
                // average however long each sleep overshoots.
                let mut next = Instant::now();
                while !stop.load(Ordering::SeqCst) {
                    let mut statuses = Vec::new();
                    while let Ok(command) = rx.try_recv() {
                        match command {
                            Command::Admit(id, spec) => {
                                engine.admit(id, spec);
                                ids.push(id);
                            }
                            Command::Cancel(id) => engine.cancel(id),
                            Command::Status(id) => statuses.push(id),
                        }
                    }
                    out.clear();
                    engine.step(&mut out);
                    out.extend(statuses.into_iter().map(|id| Event::Status { id }));
                    let now = Instant::now();
                    for &event in &out {
                        if matches!(event, Event::Final { .. }) {
                            released.lock().unwrap().push((event, now));
                        }
                        results.send(event);
                    }
                    let mut made = made.lock().unwrap();
                    for &id in &ids {
                        if let Some(n) = engine.made(id) {
                            made.insert(id, n);
                        }
                    }
                    drop(made);
                    next += tick;
                    let now = Instant::now();
                    if next > now {
                        thread::sleep(next - now);
                    } else {
                        next = now;
                    }
                }
            })
        };
        Server {
            tx,
            results,
            released,
            made,
            stop,
            worker: Some(worker),
        }
    }

    pub fn made(&self, id: Id) -> usize {
        self.made.lock().unwrap().get(&id).copied().unwrap_or(0)
    }

    /// The request's terminal event and when the engine released it.
    pub fn released(&self, id: Id) -> Option<(Event, Instant)> {
        let released = self.released.lock().unwrap();
        released.iter().copied().find(|(e, _)| e.id() == id)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Design {
    Poll,
    Event,
}

#[derive(Clone, Copy, Debug)]
pub enum Neighbour {
    Idle,
    /// Another client streams this many tokens, from the start.
    Stream(usize),
    /// A monitor asks the status endpoint every `every`, for `for_how_long`.
    Monitor {
        every: Duration,
        for_how_long: Duration,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub tick: Duration,
    pub slice: Duration,
    pub close_after: Duration,
    pub budget: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Replay {
    pub after_close: usize,
    pub delay: Duration,
}

/// A client leaves a non-streamed request: how many tokens does the engine
/// make after the client has gone, and how long until it releases the request?
pub fn replay(design: Design, neighbour: Neighbour, t: Timing) -> Replay {
    let server = Server::start(t.tick);
    let (orphan, gone) = (1, Arc::new(AtomicBool::new(false)));
    let spec = Spec {
        budget: t.budget,
        eos_at: None,
        fail_at: None,
        stream: false,
    };
    server.results.wait_for(orphan);
    server.tx.send(Command::Admit(orphan, spec)).unwrap();
    let handler = {
        let (results, tx, gone) = (server.results.clone(), server.tx.clone(), gone.clone());
        thread::spawn(move || loop {
            let got = match design {
                Design::Poll => results.recv_with_timeout(orphan, t.slice),
                Design::Event => results.recv_or_gone(orphan, &gone),
            };
            match got {
                Some(Event::Final { .. }) => return,
                Some(_) => {}
                None if gone.load(Ordering::SeqCst) => {
                    tx.send(Command::Cancel(orphan)).unwrap();
                    results.stop_waiting(orphan);
                    return;
                }
                None => {}
            }
        })
    };
    let neighbour = match neighbour {
        Neighbour::Idle => None,
        Neighbour::Stream(tokens) => {
            let (results, tx) = (server.results.clone(), server.tx.clone());
            results.wait_for(2);
            let spec = Spec {
                budget: tokens,
                eos_at: None,
                fail_at: None,
                stream: true,
            };
            tx.send(Command::Admit(2, spec)).unwrap();
            Some(thread::spawn(move || {
                while !matches!(
                    results.recv_with_timeout(2, t.slice),
                    Some(Event::Final { .. })
                ) {}
                results.stop_waiting(2);
            }))
        }
        Neighbour::Monitor {
            every,
            for_how_long,
        } => {
            let (results, tx) = (server.results.clone(), server.tx.clone());
            Some(thread::spawn(move || {
                let start = Instant::now();
                let mut id = 1000;
                while start.elapsed() < for_how_long {
                    results.wait_for(id);
                    tx.send(Command::Status(id)).unwrap();
                    results.recv_with_timeout(id, t.slice);
                    results.stop_waiting(id);
                    id += 1;
                    thread::sleep(every);
                }
            }))
        }
    };
    thread::sleep(t.close_after);
    let at_close = server.made(orphan);
    let closed = Instant::now();
    gone.store(true, Ordering::SeqCst);
    if design == Design::Event {
        server.results.wake_all();
    }
    let replay = loop {
        if let Some((Event::Final { tokens, .. }, when)) = server.released(orphan) {
            break Replay {
                after_close: tokens - at_close,
                delay: when.saturating_duration_since(closed),
            };
        }
        assert!(closed.elapsed() < Duration::from_secs(60), "never released");
        thread::sleep(Duration::from_millis(1));
    };
    handler.join().unwrap();
    if let Some(neighbour) = neighbour {
        neighbour.join().unwrap();
    }
    replay
}

/// A streaming client reads `read` tokens and leaves; its handler learns it
/// when it next writes a token, and cancels.
pub fn replay_stream(t: Timing, read: usize) -> Replay {
    let server = Server::start(t.tick);
    let spec = Spec {
        budget: t.budget,
        eos_at: None,
        fail_at: None,
        stream: true,
    };
    server.results.wait_for(1);
    server.tx.send(Command::Admit(1, spec)).unwrap();
    let (mut delivered, mut closed) = (0, None);
    loop {
        match server.results.recv_with_timeout(1, t.slice) {
            Some(Event::Token { .. }) if closed.is_none() => {
                delivered += 1;
                if delivered == read {
                    closed = Some((server.made(1).max(delivered), Instant::now()));
                }
            }
            Some(Event::Token { .. }) => {
                // The write of this token fails: the client has gone.
                server.tx.send(Command::Cancel(1)).unwrap();
                server.results.stop_waiting(1);
                break;
            }
            _ => {}
        }
    }
    let (at_close, closed) = closed.unwrap();
    loop {
        if let Some((Event::Final { tokens, .. }, when)) = server.released(1) {
            return Replay {
                after_close: tokens - at_close,
                delay: when.saturating_duration_since(closed),
            };
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn num(name: &str, default: u64) -> u64 {
    arg(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let t = Timing {
        tick: Duration::from_millis(num("--tick-ms", 4)),
        slice: Duration::from_millis(num("--slice-ms", 100)),
        close_after: Duration::from_millis(num("--close-ms", 350)),
        budget: 1500,
    };
    let scenarios = num("--scenarios", 10_000) as usize;
    let seed = num("--seed", 1);
    assert!(
        !t.tick.is_zero() && !t.slice.is_zero(),
        "positive timing intervals required"
    );
    ownership::demonstrate();
    if std::env::args().any(|a| a == "--contracts-only") {
        let mut rng = Rng::new(seed);
        let mut tally = Tally::default();
        for _ in 0..scenarios {
            random_scenario(&mut rng, &mut tally).expect("terminal contract");
        }
        println!("checked {scenarios} scenarios, {} requests", tally.requests);
        return;
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    println!("Part 1: a client leaves; how many tokens does the engine make for nobody?");
    println!(
        "a decode step every {:.0} ms; handlers wait in slices of {:.0} ms; the orphan asks for {} tokens,",
        ms(t.tick),
        ms(t.slice),
        t.budget
    );
    println!(
        "non-streamed, and its client leaves after {:.0} ms\n",
        ms(t.close_after)
    );
    println!(
        "  {:<18} {:<34} {:>13} {:>13}",
        "handler", "rest of the server", "tokens after", "close to"
    );
    println!(
        "  {:<18} {:<34} {:>13} {:>13}",
        "", "", "the close", "release (ms)"
    );
    let monitor = Neighbour::Monitor {
        every: t.tick / 2,
        for_how_long: t.slice * 20,
    };
    let cases = [
        (Neighbour::Idle, "idle".to_string()),
        (
            Neighbour::Stream(500),
            "another client streams 500 tokens".to_string(),
        ),
        (
            monitor,
            format!("a monitor asks for status every {:.0} ms", ms(t.tick / 2)),
        ),
    ];
    for design in [Design::Poll, Design::Event] {
        for (neighbour, what) in &cases {
            let r = replay(design, *neighbour, t);
            let name = match design {
                Design::Poll => "poll model",
                Design::Event => "event model",
            };
            println!(
                "  {:<18} {:<34} {:>13} {:>13.0}",
                name,
                what,
                r.after_close,
                ms(r.delay)
            );
        }
    }
    let s = replay_stream(t, 20);
    println!(
        "\n  a streaming client that leaves after 20 tokens: {} more token(s), released {:.0} ms after it left",
        s.after_close,
        ms(s.delay)
    );

    println!("\nPart 2: the terminal contract on {scenarios} random scenarios (seed {seed})");
    println!(
        "1-8 requests each, random arrivals, budgets, EOS, injected failures and cancellations\n"
    );
    let mut rng = Rng::new(seed);
    let mut tally = Tally::default();
    for i in 0..scenarios {
        if let Err(e) = random_scenario(&mut rng, &mut tally) {
            eprintln!("contract broken in scenario {i}: {e}");
            std::process::exit(1);
        }
    }
    let mut outcomes: Vec<_> = tally.outcomes.iter().collect();
    outcomes.sort();
    let summary: Vec<String> = outcomes.iter().map(|(k, v)| format!("{k} {v}")).collect();
    println!("  {} requests: {}", tally.requests, summary.join(", "));
    println!("  every request ended exactly once, nothing followed its end, streamed tokens");
    println!("  arrived in order and complete, uncancelled requests ended as the oracle");
    println!("  predicted, and no request made a token after its cancellation");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(budget: usize, eos_at: Option<usize>, fail_at: Option<usize>, stream: bool) -> Spec {
        Spec {
            budget,
            eos_at,
            fail_at,
            stream,
        }
    }

    fn run(spec: Spec, cancel_at_step: Option<usize>) -> Vec<Event> {
        let mut engine = Engine::default();
        engine.admit(7, spec);
        let mut out = Vec::new();
        for step in 0.. {
            if cancel_at_step == Some(step) {
                engine.cancel(7);
            }
            if engine.is_idle() {
                break;
            }
            engine.step(&mut out);
        }
        out
    }

    #[test]
    fn each_way_to_end_is_one_final() {
        let cases = [
            (spec(3, None, None, true), Outcome::MaxTokens, 3),
            (spec(9, Some(2), None, true), Outcome::EndOfSequence, 2),
            (spec(9, Some(2), Some(1), false), Outcome::Failed, 1),
            (spec(9, Some(0), None, true), Outcome::EndOfSequence, 0),
        ];
        for (s, outcome, tokens) in cases {
            let events = run(s, None);
            assert_eq!(check(&events, &s, None), Ok(outcome));
            assert_eq!(
                *events.last().unwrap(),
                Event::Final {
                    id: 7,
                    tokens,
                    outcome
                }
            );
        }
    }

    #[test]
    fn cancellation_stops_before_the_next_token() {
        let s = spec(100, None, None, true);
        let events = run(s, Some(5));
        assert_eq!(check(&events, &s, Some(5)), Ok(Outcome::Cancelled));
        assert_eq!(events.len(), 6);
    }

    #[test]
    fn the_checker_catches_a_second_final() {
        let s = spec(2, None, None, true);
        let mut events = run(s, None);
        events.push(Event::Final {
            id: 7,
            tokens: 2,
            outcome: Outcome::Cancelled,
        });
        assert!(check(&events, &s, None).is_err());
    }

    #[test]
    fn checker_rejects_natural_end_after_observed_cancellation() {
        let s = spec(3, None, None, true);
        assert!(check(&run(s, None), &s, Some(1)).is_err());
    }

    #[test]
    fn checker_rejects_mixed_id_status_missing_and_late_output() {
        let s = spec(2, None, None, true);
        let good = run(s, None);
        let mut mixed = good.clone();
        mixed[0] = Event::Token { id: 8, index: 0 };
        assert!(check(&mixed, &s, None).is_err());
        let mut status = good.clone();
        status.insert(0, Event::Status { id: 7 });
        assert!(check(&status, &s, None).is_err());
        assert!(check(&good[..2], &s, None).is_err());
        let mut late = good;
        late.push(Event::Token { id: 7, index: 2 });
        assert!(check(&late, &s, None).is_err());
    }

    #[test]
    fn failure_beats_eos_and_cancellation_beats_both() {
        let s = spec(3, Some(1), Some(1), true);
        assert_eq!(check(&run(s, None), &s, None), Ok(Outcome::Failed));
        assert_eq!(check(&run(s, Some(1)), &s, Some(1)), Ok(Outcome::Cancelled));
    }

    #[test]
    fn duplicate_active_admission_is_rejected_without_changing_owner() {
        let mut engine = Engine::default();
        let s = spec(2, None, None, true);
        engine.admit(7, s);
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.admit(7, s);
        }));
        assert!(rejected.is_err());
        let mut out = Vec::new();
        engine.step(&mut out);
        engine.step(&mut out);
        assert_eq!(check(&out, &s, None), Ok(Outcome::MaxTokens));
    }

    #[test]
    fn exhaustive_small_specs_match_closed_form_end() {
        for budget in 1..=5 {
            for eos in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)] {
                for fail in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)] {
                    // Independent minimum of three boundaries, not another step loop.
                    let mut boundaries = vec![(budget, 2, Outcome::MaxTokens)];
                    if let Some(k) = eos {
                        if k < budget {
                            boundaries.push((k, 1, Outcome::EndOfSequence));
                        }
                    }
                    if let Some(k) = fail {
                        if k < budget {
                            boundaries.push((k, 0, Outcome::Failed));
                        }
                    }
                    boundaries.sort_by_key(|&(k, priority, _)| (k, priority));
                    let (n, _, outcome) = boundaries[0];
                    for stream in [false, true] {
                        let s = spec(budget, eos, fail, stream);
                        assert_eq!(natural_end(&s), (n, outcome));
                        assert_eq!(
                            *run(s, None).last().unwrap(),
                            Event::Final {
                                id: 7,
                                tokens: n,
                                outcome
                            }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn random_scenarios_keep_the_contract() {
        let mut rng = Rng::new(3);
        let mut tally = Tally::default();
        for _ in 0..2000 {
            random_scenario(&mut rng, &mut tally).unwrap();
        }
        assert_eq!(tally.outcomes.len(), 4, "every outcome occurs");
    }

    #[test]
    fn a_streaming_neighbour_keeps_a_polled_orphan_alive() {
        let t = Timing {
            tick: Duration::from_millis(2),
            slice: Duration::from_millis(50),
            close_after: Duration::from_millis(175),
            budget: 2000,
        };
        let idle = replay(Design::Poll, Neighbour::Idle, t);
        let busy = replay(Design::Poll, Neighbour::Stream(400), t);
        let event = replay(Design::Event, Neighbour::Stream(400), t);
        // Idle: caught at the end of a slice. Busy: not before the neighbour's
        // 400 steps end (800 ms, 312 steps after the close), with a margin for
        // a slow machine. Events: within a couple of steps.
        assert!(idle.after_close <= 40, "{idle:?}");
        assert!(busy.after_close >= 200, "{busy:?}");
        assert!(event.after_close <= 3, "{event:?}");
    }
}
