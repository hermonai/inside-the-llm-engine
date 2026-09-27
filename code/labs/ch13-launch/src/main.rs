//! Chapter 13 lab: what a launch costs, what a graph saves and what fusion saves.
//!
//! A worker thread stands in for the GPU. It owns a buffer and runs the
//! commands the host sends it through a queue, strictly in order, as a CUDA
//! stream or a Metal command queue does; the host never touches the buffer.
//!
//! Part 1 applies a chain of n dependent additions, x += c_k for k = 1..n, in
//! four ways:
//!
//! - `sync`: send one addition, wait for it, send the next --- a launch
//!   followed by a synchronization, every time;
//! - `stream`: send all n, then wait once --- asynchronous launches;
//! - `graph`: record the n additions once and send the recording as one
//!   command --- a CUDA graph, or one Metal command buffer;
//! - `fused`: one command whose kernel applies all n additions to a tile of
//!   elements while it sits in registers.
//!
//! All four must reproduce the oracle's bits, since each performs the same float
//! additions in the same order. Only then are they timed, at several buffer
//! sizes.
//!
//! Part 2 measures fusion on data too large for the caches: RMSNorm, a scale
//! and a shift (y = norm(x) * w + b, three nodes that llama.cpp's Metal backend
//! fuses into one kernel) as three passes or one.
//!
//! Part 3 derives what padding a batch to the nearest captured graph size
//! costs, with vLLM's default capture sizes and with powers of two.
//!
//!     cargo run --release -p ch13-launch -- [--ops 1000] [--sizes 256,4096,65536,1048576]
//!         [--reps 7] [--tokens 4096] [--dim 3072] [--threads 1,4]

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Elements the fused kernel keeps in registers at once: sixteen 128-bit
/// registers of f32, enough independent chains to keep four adders busy.
/// Also the number of accumulators in a row's sum of squares; a power of two.
const TILE: usize = 64;

/// Rows of the fusion test handed to a thread at a time.
const ROWS: usize = 16;

pub fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

/// The chain's constants: small, and summing to zero every seven steps, so the
/// values stay near the inputs' range however often the chain is replayed.
pub fn constants(n: usize) -> Vec<f32> {
    (0..n).map(|k| ((k % 7) as f32 - 3.0) * 1e-3).collect()
}

/// One kernel: x[i] += c for every element.
pub fn add(x: &mut [f32], c: f32) {
    for v in x.iter_mut() {
        *v += c;
    }
}

/// The fused kernel: every addition applied to a tile of elements while it is
/// in registers, in the same order as n calls to `add`, so the bits are the same.
pub fn add_all(x: &mut [f32], cs: &[f32]) {
    let mut tiles = x.chunks_exact_mut(TILE);
    for tile in &mut tiles {
        let mut a = [0.0f32; TILE];
        a.copy_from_slice(tile);
        for &c in cs {
            for v in a.iter_mut() {
                *v += c;
            }
        }
        tile.copy_from_slice(&a);
    }
    for v in tiles.into_remainder() {
        *v = cs.iter().fold(*v, |a, &c| a + c);
    }
}

/// The oracle: each element's additions done one at a time on the host.
pub fn oracle(x: &[f32], cs: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| cs.iter().fold(v, |a, &c| a + c))
        .collect()
}

pub enum Command {
    Add(f32),
    Graph(Arc<[f32]>),
    Fused(Arc<[f32]>),
    Fence,
    Read(mpsc::Sender<Vec<f32>>),
}

/// A worker thread that owns a buffer and runs commands in order: this lab's
/// GPU. `fence` returns once everything sent before it has run.
pub struct Device {
    tx: Option<mpsc::Sender<Command>>,
    done: mpsc::Receiver<()>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Device {
    pub fn new(x: Vec<f32>) -> Device {
        let (tx, rx) = mpsc::channel::<Command>();
        let (done_tx, done) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut x = x;
            for command in rx {
                match command {
                    Command::Add(c) => add(&mut x, c),
                    Command::Graph(cs) => {
                        for &c in cs.iter() {
                            add(&mut x, c);
                        }
                    }
                    Command::Fused(cs) => add_all(&mut x, &cs),
                    Command::Fence => done_tx.send(()).unwrap(),
                    Command::Read(out) => out.send(x.clone()).unwrap(),
                }
            }
        });
        Device {
            tx: Some(tx),
            done,
            worker: Some(worker),
        }
    }

    pub fn send(&self, command: Command) {
        self.tx.as_ref().unwrap().send(command).unwrap();
    }

    pub fn fence(&self) {
        self.send(Command::Fence);
        self.done.recv().unwrap();
    }

    /// Copies the buffer back to the host, after everything sent before.
    pub fn read(&self) -> Vec<f32> {
        let (tx, rx) = mpsc::channel();
        self.send(Command::Read(tx));
        rx.recv().unwrap()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    Sync,
    Stream,
    Graph,
    Fused,
}

pub const METHODS: [Method; 4] = [Method::Sync, Method::Stream, Method::Graph, Method::Fused];

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Sync => "sync",
            Method::Stream => "stream",
            Method::Graph => "graph",
            Method::Fused => "fused",
        }
    }
}

/// Applies the chain `cs` once, one of four ways, and waits for it to finish.
/// The graph is recorded before the call: replaying it is one command.
pub fn run(device: &Device, method: Method, cs: &Arc<[f32]>) {
    match method {
        Method::Sync => {
            for &c in cs.iter() {
                device.send(Command::Add(c));
                device.fence();
            }
        }
        Method::Stream => {
            for &c in cs.iter() {
                device.send(Command::Add(c));
            }
            device.fence();
        }
        Method::Graph => {
            device.send(Command::Graph(Arc::clone(cs)));
            device.fence();
        }
        Method::Fused => {
            device.send(Command::Fused(Arc::clone(cs)));
            device.fence();
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// Part 2: RMSNorm, scale and shift, as three passes or one.

/// Sum of squares with `TILE` accumulators, enough independent chains that
/// one core keeps up with memory, combined pairwise in a fixed order.
fn sum_squares(row: &[f32]) -> f32 {
    let mut acc = [0.0f32; TILE];
    let mut chunks = row.chunks_exact(TILE);
    for c in &mut chunks {
        for (a, v) in acc.iter_mut().zip(c) {
            *a += v * v;
        }
    }
    let mut width = TILE;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] += acc[i + width];
        }
    }
    let mut s = acc[0];
    for v in chunks.remainder() {
        s += v * v;
    }
    s
}

fn inv_rms(row: &[f32], eps: f32) -> f32 {
    1.0 / (sum_squares(row) / row.len() as f32 + eps).sqrt()
}

/// Runs `f(first_row, rows)` over blocks of `ROWS` rows of `y` on `threads`
/// threads, handing blocks out from a shared iterator so faster cores take more.
fn par_rows<F>(y: &mut [f32], d: usize, threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let blocks = Mutex::new(y.chunks_mut(ROWS * d).enumerate());
    thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let next = blocks.lock().unwrap().next();
                match next {
                    Some((i, rows)) => f(i * ROWS, rows),
                    None => break,
                }
            });
        }
    });
}

/// Three kernels, three passes over memory: y = norm(x); y *= w; y += b.
pub fn unfused(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32], eps: f32, threads: usize) {
    let d = w.len();
    par_rows(y, d, threads, |r0, ys| {
        for (k, yr) in ys.chunks_mut(d).enumerate() {
            let xr = &x[(r0 + k) * d..][..d];
            let inv = inv_rms(xr, eps);
            for (yv, xv) in yr.iter_mut().zip(xr) {
                *yv = xv * inv;
            }
        }
    });
    par_rows(y, d, threads, |_, ys| {
        for yr in ys.chunks_mut(d) {
            for (yv, wv) in yr.iter_mut().zip(w) {
                *yv *= wv;
            }
        }
    });
    par_rows(y, d, threads, |_, ys| {
        for yr in ys.chunks_mut(d) {
            for (yv, bv) in yr.iter_mut().zip(b) {
                *yv += bv;
            }
        }
    });
}

/// One kernel, one pass: the same three roundings per element, in the same order.
pub fn fused(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32], eps: f32, threads: usize) {
    let d = w.len();
    par_rows(y, d, threads, |r0, ys| {
        for (k, yr) in ys.chunks_mut(d).enumerate() {
            let xr = &x[(r0 + k) * d..][..d];
            let inv = inv_rms(xr, eps);
            for (((yv, xv), wv), bv) in yr.iter_mut().zip(xr).zip(w).zip(b) {
                *yv = xv * inv * wv + bv;
            }
        }
    });
}

/// Largest error against a float64 evaluation, relative to the largest output.
fn norm_error(x: &[f32], w: &[f32], b: &[f32], y: &[f32], eps: f32) -> f64 {
    let d = w.len();
    let (mut worst, mut peak) = (0.0f64, 0.0f64);
    for (xr, yr) in x.chunks(d).zip(y.chunks(d)) {
        let ms = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for i in 0..d {
            let want = xr[i] as f64 * inv * w[i] as f64 + b[i] as f64;
            worst = worst.max((yr[i] as f64 - want).abs());
            peak = peak.max(want.abs());
        }
    }
    worst / peak
}

// Part 3: padding batches to captured graph sizes (derived).

/// vLLM's default CUDA-graph capture sizes at commit bcdacfc: 1, 2, 4, the
/// multiples of 8 below 256, then multiples of 16 up to the largest (512 on most
/// GPUs, 1,024 on data-centre Blackwell).
pub fn vllm_sizes(max: usize) -> Vec<usize> {
    let mut s = vec![1, 2, 4];
    s.extend((8..256).step_by(8));
    s.extend((256..=max).step_by(16));
    s
}

pub fn powers_of_two(max: usize) -> Vec<usize> {
    std::iter::successors(Some(1usize), |s| Some(s * 2))
        .take_while(|&s| s <= max)
        .collect()
}

/// The size a batch runs at: the smallest captured size that holds it, or
/// `None` when it exceeds them all and must run without a graph.
pub fn padded(batch: usize, sizes: &[usize]) -> Option<usize> {
    sizes.iter().copied().find(|&s| s >= batch)
}

/// Mean fraction of padded (wasted) slots over batches 1..=max, equally likely,
/// and the worst batch.
pub fn padding(sizes: &[usize], max: usize) -> (f64, f64, usize) {
    let (mut sum, mut worst, mut at) = (0.0, 0.0, 1);
    for batch in 1..=max {
        let p = padded(batch, sizes).expect("the sizes must reach max");
        let waste = (p - batch) as f64 / p as f64;
        sum += waste;
        if waste > worst {
            worst = waste;
            at = batch;
        }
    }
    (sum / max as f64, worst, at)
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn list(name: &str, default: &[usize]) -> Vec<usize> {
    arg(name)
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn main() {
    let ops: usize = arg("--ops").and_then(|v| v.parse().ok()).unwrap_or(1000);
    let sizes = list("--sizes", &[256, 4096, 65536, 1048576]);
    let reps: usize = arg("--reps").and_then(|v| v.parse().ok()).unwrap_or(7);
    let tokens: usize = arg("--tokens").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let dim: usize = arg("--dim").and_then(|v| v.parse().ok()).unwrap_or(3072);
    let threads = list("--threads", &[1, 4]);

    let cs: Arc<[f32]> = constants(ops).into();
    println!("Part 1: a chain of {ops} dependent additions x += c on a worker thread");
    println!("every method checked against the host oracle bit for bit, then timed");
    println!("microseconds per addition, median of {reps} runs\n");
    println!(
        "  {:>9} {:>10} {:>10} {:>10} {:>10}",
        "elements", "sync", "stream", "graph", "fused"
    );
    for &n in &sizes {
        let x = fill(n, 1, 2.0);
        let want = oracle(&x, &cs);
        let mut row = Vec::new();
        for method in METHODS {
            let device = Device::new(x.clone());
            run(&device, method, &cs);
            if !same_bits(&device.read(), &want) {
                eprintln!("oracle failed: {} at {n} elements", method.name());
                std::process::exit(1);
            }
            let times: Vec<f64> = (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    run(&device, method, &cs);
                    t.elapsed().as_secs_f64()
                })
                .collect();
            row.push(median(times) / ops as f64 * 1e6);
        }
        println!(
            "  {:>9} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            n, row[0], row[1], row[2], row[3]
        );
    }
    println!("\n  bits: all four methods matched the oracle at every size");

    let eps = 1e-5f32;
    let x = fill(tokens * dim, 2, 4.0);
    let w: Vec<f32> = fill(dim, 3, 1.0).iter().map(|v| 1.0 + v).collect();
    let b = fill(dim, 4, 0.2);
    let mb = (tokens * dim * 4) as f64 / 1e6;
    println!(
        "\nPart 2: y = rms_norm(x) * w + b on {tokens} rows of {dim} (x and y {mb:.1} MB each)"
    );
    println!(
        "bytes counted: each pass reads one full array and writes one (w and b stay in cache)"
    );
    println!("median of {reps} runs\n");
    println!(
        "  {:>7} {:>9} {:>7} {:>9} {:>8} {:>9}",
        "threads", "kernels", "passes", "ms", "GB/s", "error"
    );
    let mut y1 = vec![0.0f32; tokens * dim];
    let mut y2 = vec![0.0f32; tokens * dim];
    for &t in &threads {
        unfused(&x, &w, &b, &mut y1, eps, t);
        fused(&x, &w, &b, &mut y2, eps, t);
        let (e1, e2) = (
            norm_error(&x, &w, &b, &y1, eps),
            norm_error(&x, &w, &b, &y2, eps),
        );
        if e1 > 1e-6 || e2 > 1e-6 || !same_bits(&y1, &y2) {
            eprintln!("oracle failed at {t} threads: errors {e1:.2e}, {e2:.2e}");
            std::process::exit(1);
        }
        let time = |f: &mut dyn FnMut()| {
            median(
                (0..reps)
                    .map(|_| {
                        let s = Instant::now();
                        f();
                        s.elapsed().as_secs_f64()
                    })
                    .collect(),
            )
        };
        let tu = time(&mut || unfused(&x, &w, &b, &mut y1, eps, t));
        let tf = time(&mut || fused(&x, &w, &b, &mut y2, eps, t));
        println!(
            "  {:>7} {:>9} {:>7} {:>9.2} {:>8.1} {:>9.1e}",
            t,
            3,
            3,
            tu * 1e3,
            6.0 * mb / 1e3 / tu,
            e1
        );
        println!(
            "  {:>7} {:>9} {:>7} {:>9.2} {:>8.1} {:>9.1e}   {:.2}x faster",
            t,
            1,
            1,
            tf * 1e3,
            2.0 * mb / 1e3 / tf,
            e2,
            tu / tf
        );
    }
    println!("\n  bits: fused and unfused outputs identical");

    let max = 512;
    println!("\nPart 3: padding each batch up to the next captured graph size (derived)");
    println!("batches 1..{max} equally likely; padding = wasted slots / slots computed\n");
    println!(
        "  {:<16} {:>7} {:>13} {:>16} {:>11}",
        "capture sizes", "graphs", "mean padding", "worst", "17 runs at"
    );
    let every: Vec<usize> = (1..=max).collect();
    for (name, sizes) in [
        ("vLLM default", vllm_sizes(max)),
        ("powers of two", powers_of_two(max)),
        ("every size", every),
    ] {
        let (mean, worst, at) = padding(&sizes, max);
        println!(
            "  {:<16} {:>7} {:>12.1}% {:>7.1}% ({} -> {}) {:>11}",
            name,
            sizes.len(),
            mean * 100.0,
            worst * 100.0,
            at,
            padded(at, &sizes).unwrap(),
            padded(17, &sizes).unwrap()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_methods_match_the_oracle_bit_for_bit() {
        let cs: Arc<[f32]> = constants(100).into();
        for n in [1, 31, 32, 33, 1000] {
            let x = fill(n, 7, 2.0);
            let want = oracle(&x, &cs);
            for method in METHODS {
                let device = Device::new(x.clone());
                run(&device, method, &cs);
                assert!(same_bits(&device.read(), &want), "{method:?} at {n}");
            }
        }
    }

    #[test]
    fn replays_accumulate_in_order() {
        let cs: Arc<[f32]> = constants(10).into();
        let x = fill(64, 8, 1.0);
        let twice: Vec<f32> = cs.iter().chain(cs.iter()).copied().collect();
        let device = Device::new(x.clone());
        run(&device, Method::Graph, &cs);
        run(&device, Method::Stream, &cs);
        assert!(same_bits(&device.read(), &oracle(&x, &twice)));
    }

    #[test]
    fn fused_norm_matches_three_passes_bit_for_bit() {
        let (t, d) = (37, 200);
        let x = fill(t * d, 2, 4.0);
        let w: Vec<f32> = fill(d, 3, 1.0).iter().map(|v| 1.0 + v).collect();
        let b = fill(d, 4, 0.2);
        let (mut y1, mut y2) = (vec![0.0; t * d], vec![0.0; t * d]);
        for threads in [1, 3] {
            unfused(&x, &w, &b, &mut y1, 1e-5, threads);
            fused(&x, &w, &b, &mut y2, 1e-5, threads);
            assert!(same_bits(&y1, &y2));
            assert!(norm_error(&x, &w, &b, &y2, 1e-5) < 1e-6);
        }
    }

    #[test]
    fn vllm_capture_sizes() {
        let s = vllm_sizes(512);
        assert_eq!(s.len(), 51);
        assert_eq!(&s[..5], &[1, 2, 4, 8, 16]);
        assert_eq!(s[33..36], [248, 256, 272]);
        assert_eq!(vllm_sizes(1024).len(), 83);
        assert_eq!(padded(5, &s), Some(8));
        assert_eq!(padded(17, &s), Some(24));
        assert_eq!(padded(257, &s), Some(272));
        assert_eq!(padded(513, &s), None);
        assert_eq!(padded(512, &s), Some(512));
    }

    #[test]
    fn padding_worst_cases() {
        let (_, worst, at) = padding(&powers_of_two(512), 512);
        assert_eq!((at, padded(at, &powers_of_two(512))), (257, Some(512)));
        assert!((worst - 255.0 / 512.0).abs() < 1e-12);
        let (mean, worst, _) = padding(&(1..=512).collect::<Vec<_>>(), 512);
        assert_eq!((mean, worst), (0.0, 0.0));
    }
}
