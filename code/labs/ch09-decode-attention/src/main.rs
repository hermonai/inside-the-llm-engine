//! Chapter 9 lab: decode attention, split along the sequence.
//!
//! One decode step of attention for one layer: a single query per head against
//! a cache of S tokens, in a grouped-query layout (by default 16 query heads
//! sharing 2 key/value heads of width 128, the shape of Qwen2.5-3B). The work is
//! cut into tasks and run on a pool of threads. Each task returns a partial ---
//! a running maximum, denominator and unnormalized output --- and the partials
//! of a head are merged in index order with the log-sum-exp rule.
//!
//! Four plans:
//!
//! - `heads`: one task per query head, walking the whole cache;
//! - `packed`: one task per key/value head, the group's query heads sharing
//!   one pass over its keys and values;
//! - `split`: one task per (query head, chunk of `--chunk` tokens);
//! - `packed+split`: one task per (key/value head, chunk).
//!
//! Every result is checked against float64 attention before it is timed. The
//! program reports the best time, the effective bandwidth (cache bytes counted
//! once), and whether each plan's output is bit-identical at every thread
//! count.
//!
//!     cargo run --release -p ch09-decode-attention -- [--contexts 512,4096,32768]
//!         [--threads 1,2,4,8] [--q-heads 16] [--kv-heads 2] [--chunk 256]

use std::sync::Mutex;
use std::time::Instant;

pub const D: usize = 128;
const TILE: usize = 32;

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

/// A dot product of width D with sixteen independent accumulators, combined in
/// a fixed order, so that it vectorizes and gives the same bits every time.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for ((s, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }
    let mut s = [0.0f32; 4];
    for quad in acc.chunks_exact(4) {
        for (sl, ql) in s.iter_mut().zip(quad) {
            *sl += ql;
        }
    }
    (s[0] + s[2]) + (s[1] + s[3])
}

/// The state of one query head over part of the cache: running maximum,
/// denominator and unnormalized output.
#[derive(Clone)]
pub struct Partial {
    pub m: f32,
    pub l: f32,
    pub o: [f32; D],
}

impl Partial {
    pub fn empty() -> Self {
        Partial {
            m: f32::NEG_INFINITY,
            l: 0.0,
            o: [0.0; D],
        }
    }
}

/// Attention of `qs` (the query heads of one group, or a single head) over
/// tokens `a..b` of one key/value head, a tile of 32 tokens at a time.
pub fn attend(qs: &[&[f32]], k: &[f32], v: &[f32], a: usize, b: usize, out: &mut [Partial]) {
    let scale = 1.0 / (D as f32).sqrt();
    let mut s = vec![[0.0f32; TILE]; qs.len()];
    for p in out.iter_mut() {
        *p = Partial::empty();
    }
    let mut t0 = a;
    while t0 < b {
        let n = TILE.min(b - t0);
        // Scores: each key row is loaded once and used by every query of the group.
        for t in 0..n {
            let key = &k[(t0 + t) * D..(t0 + t + 1) * D];
            for (h, q) in qs.iter().enumerate() {
                s[h][t] = dot(q, key) * scale;
            }
        }
        // Online softmax, one rescale per tile.
        for (h, p) in out.iter_mut().enumerate() {
            let tile_max = s[h][..n].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if tile_max > p.m {
                let f = (p.m - tile_max).exp();
                p.l *= f;
                p.o.iter_mut().for_each(|x| *x *= f);
                p.m = tile_max;
            }
            for x in s[h][..n].iter_mut() {
                *x = (*x - p.m).exp();
                p.l += *x;
            }
        }
        // Weighted values: each value row is loaded once for the whole group.
        for t in 0..n {
            let val = &v[(t0 + t) * D..(t0 + t + 1) * D];
            for (h, p) in out.iter_mut().enumerate() {
                let w = s[h][t];
                for (o, x) in p.o.iter_mut().zip(val) {
                    *o += w * x;
                }
            }
        }
        t0 += n;
    }
}

/// Merges partials over disjoint parts of the cache, in index order:
/// m = max(m1, m2), l = l1 e^(m1-m) + l2 e^(m2-m), o likewise. Returns o / l.
pub fn merge(parts: &[Partial]) -> [f32; D] {
    let mut acc = Partial::empty();
    for p in parts {
        if p.m == f32::NEG_INFINITY {
            continue;
        }
        let m = acc.m.max(p.m);
        let (fa, fp) = ((acc.m - m).exp(), (p.m - m).exp());
        acc.l = acc.l * fa + p.l * fp;
        for (o, x) in acc.o.iter_mut().zip(&p.o) {
            *o = *o * fa + *x * fp;
        }
        acc.m = m;
    }
    let mut out = acc.o;
    out.iter_mut().for_each(|x| *x /= acc.l);
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Plan {
    Heads,
    Packed,
    Split,
    PackedSplit,
}

impl Plan {
    pub fn name(self) -> &'static str {
        match self {
            Plan::Heads => "heads",
            Plan::Packed => "packed",
            Plan::Split => "split",
            Plan::PackedSplit => "packed+split",
        }
    }
}

pub struct Shape {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub s: usize,
    pub chunk: usize,
}

/// One task: query heads `q0..q0+nq` (all of kv head `kv`) over tokens `a..b`.
struct Task {
    kv: usize,
    q0: usize,
    nq: usize,
    a: usize,
    b: usize,
}

/// Runs one decode step of attention with the given plan on `threads` threads.
/// Returns the output ([q_heads][D]) and the number of tasks.
pub fn decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: &Shape,
    plan: Plan,
    threads: usize,
) -> (Vec<f32>, usize) {
    let g = shape.q_heads / shape.kv_heads;
    let chunk = match plan {
        Plan::Heads | Plan::Packed => shape.s.max(1),
        Plan::Split | Plan::PackedSplit => shape.chunk,
    };
    let n_chunks = shape.s.div_ceil(chunk).max(1);
    // Partials laid out [q_head][chunk], so that each head's are contiguous and
    // merged in chunk order whatever order the tasks finish in.
    let mut parts = vec![Partial::empty(); shape.q_heads * n_chunks];
    let n_tasks;
    {
        let mut slots: Vec<Option<&mut Partial>> = parts.iter_mut().map(Some).collect();
        let mut tasks: Vec<(Task, Vec<&mut Partial>)> = Vec::new();
        for kv in 0..shape.kv_heads {
            for c in 0..n_chunks {
                let (a, b) = (c * chunk, ((c + 1) * chunk).min(shape.s));
                let groups: Vec<(usize, usize)> = match plan {
                    Plan::Heads | Plan::Split => (0..g).map(|i| (kv * g + i, 1)).collect(),
                    Plan::Packed | Plan::PackedSplit => vec![(kv * g, g)],
                };
                for (q0, nq) in groups {
                    let out = (q0..q0 + nq)
                        .map(|h| slots[h * n_chunks + c].take().expect("slot used once"))
                        .collect();
                    tasks.push((Task { kv, q0, nq, a, b }, out));
                }
            }
        }
        n_tasks = tasks.len();
        let queue = Mutex::new(tasks.into_iter());
        let kv_len = shape.s * D;
        let work = || loop {
            let next = queue.lock().unwrap().next();
            let Some((t, mut out)) = next else { break };
            let qs: Vec<&[f32]> = (t.q0..t.q0 + t.nq)
                .map(|h| &q[h * D..(h + 1) * D])
                .collect();
            let (kh, vh) = (
                &k[t.kv * kv_len..(t.kv + 1) * kv_len],
                &v[t.kv * kv_len..(t.kv + 1) * kv_len],
            );
            let mut local = vec![Partial::empty(); t.nq];
            attend(&qs, kh, vh, t.a, t.b, &mut local);
            for (slot, p) in out.iter_mut().zip(local) {
                **slot = p;
            }
        };
        if threads <= 1 {
            work();
        } else {
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(work);
                }
            });
        }
    }
    let mut out = vec![0.0f32; shape.q_heads * D];
    for (h, o) in out.chunks_mut(D).enumerate() {
        o.copy_from_slice(&merge(&parts[h * n_chunks..(h + 1) * n_chunks]));
    }
    (out, n_tasks)
}

/// Largest error against float64 attention over all heads, relative to the
/// largest output magnitude.
pub fn error(q: &[f32], k: &[f32], v: &[f32], shape: &Shape, got: &[f32]) -> f64 {
    let g = shape.q_heads / shape.kv_heads;
    let scale = 1.0 / (D as f64).sqrt();
    let (mut worst, mut peak) = (0.0f64, 1e-30f64);
    for h in 0..shape.q_heads {
        let kv = h / g;
        let base = kv * shape.s * D;
        let qh = &q[h * D..(h + 1) * D];
        let scores: Vec<f64> = (0..shape.s)
            .map(|j| {
                let key = &k[base + j * D..base + (j + 1) * D];
                qh.iter()
                    .zip(key)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
        let sum: f64 = w.iter().sum();
        for c in 0..D {
            let exact = w
                .iter()
                .enumerate()
                .map(|(j, p)| p * v[base + j * D + c] as f64)
                .sum::<f64>()
                / sum;
            worst = worst.max((got[h * D + c] as f64 - exact).abs());
            peak = peak.max(exact.abs());
        }
    }
    worst / peak
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
    let contexts = list("--contexts", &[512, 4096, 32768]);
    let threads = list("--threads", &[1, 2, 4, 8]);
    let q_heads: usize = arg("--q-heads").and_then(|v| v.parse().ok()).unwrap_or(16);
    let kv_heads: usize = arg("--kv-heads").and_then(|v| v.parse().ok()).unwrap_or(2);
    let chunk: usize = arg("--chunk").and_then(|v| v.parse().ok()).unwrap_or(256);
    let reps: usize = arg("--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    assert!(
        q_heads.is_multiple_of(kv_heads),
        "query heads must be a multiple of key/value heads"
    );
    let tol = 1e-5;
    println!(
        "decode attention, one layer: {q_heads} query heads, {kv_heads} key/value heads, \
         head width {D}, f32 cache; chunks of {chunk} tokens"
    );
    println!("every output checked against float64 (error relative to the output's scale)");
    for &s in &contexts {
        let shape = Shape {
            q_heads,
            kv_heads,
            s,
            chunk,
        };
        let q = fill(q_heads * D, 1, 4.0);
        let k = fill(kv_heads * s * D, 2, 4.0);
        let v = fill(kv_heads * s * D, 3, 1.0);
        let bytes = (2 * kv_heads * s * D * 4) as f64;
        println!(
            "\ncontext {s} (cache {:.1} MiB, counted once)",
            bytes / (1 << 20) as f64
        );
        println!(
            "  {:<13} {:>6} {:>8} {:>10} {:>8} {:>9}  bits across thread counts",
            "plan", "tasks", "threads", "best ms", "GB/s", "error"
        );
        let mut reference: Option<Vec<f32>> = None;
        for plan in [Plan::Heads, Plan::Packed, Plan::Split, Plan::PackedSplit] {
            let mut first: Option<Vec<f32>> = None;
            for &t in &threads {
                let (out, n_tasks) = decode(&q, &k, &v, &shape, plan, t);
                let e = error(&q, &k, &v, &shape, &out);
                if e > tol {
                    eprintln!(
                        "oracle failed: {} at S = {s}, {t} threads, error {e:.2e}",
                        plan.name()
                    );
                    std::process::exit(1);
                }
                let mut best = f64::INFINITY;
                for _ in 0..reps {
                    let t0 = Instant::now();
                    let _ = decode(&q, &k, &v, &shape, plan, t);
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                let same = match &first {
                    None => {
                        first = Some(out.clone());
                        "first".to_string()
                    }
                    Some(f) => {
                        if f.iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits()) {
                            "identical".to_string()
                        } else {
                            "DIFFERENT".to_string()
                        }
                    }
                };
                println!(
                    "  {:<13} {:>6} {:>8} {:>10.3} {:>8.1} {:>9.1e}  {same}",
                    plan.name(),
                    n_tasks,
                    t,
                    best * 1e3,
                    bytes / best / 1e9,
                    e
                );
            }
            let out = first.expect("at least one thread count");
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    let same = r
                        .iter()
                        .zip(&out)
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count();
                    let gap = r
                        .iter()
                        .zip(&out)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "  {:<13} against heads: {same} of {} outputs bit-identical, largest gap {gap:.1e}",
                        plan.name(),
                        out.len()
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(
        q_heads: usize,
        kv_heads: usize,
        s: usize,
        chunk: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Shape) {
        (
            fill(q_heads * D, 1, 4.0),
            fill(kv_heads * s * D, 2, 4.0),
            fill(kv_heads * s * D, 3, 1.0),
            Shape {
                q_heads,
                kv_heads,
                s,
                chunk,
            },
        )
    }

    #[test]
    fn every_plan_agrees_with_float64() {
        for (qh, kvh, s, chunk) in [
            (4, 4, 1, 256),
            (8, 2, 100, 32),
            (16, 2, 1000, 256),
            (6, 3, 777, 64),
        ] {
            let (q, k, v, shape) = setup(qh, kvh, s, chunk);
            for plan in [Plan::Heads, Plan::Packed, Plan::Split, Plan::PackedSplit] {
                for t in [1, 3] {
                    let (out, _) = decode(&q, &k, &v, &shape, plan, t);
                    assert!(
                        error(&q, &k, &v, &shape, &out) < 1e-5,
                        "{plan:?} {t} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn a_fixed_chunk_size_gives_the_same_bits_on_any_number_of_threads() {
        let (q, k, v, shape) = setup(16, 2, 3000, 256);
        for plan in [Plan::Split, Plan::PackedSplit] {
            let (one, _) = decode(&q, &k, &v, &shape, plan, 1);
            for t in [2, 4, 7] {
                let (many, _) = decode(&q, &k, &v, &shape, plan, t);
                assert!(one
                    .iter()
                    .zip(&many)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }

    #[test]
    fn packing_the_group_changes_nothing_but_the_reads() {
        let (q, k, v, shape) = setup(16, 2, 3000, 256);
        let (a, _) = decode(&q, &k, &v, &shape, Plan::Split, 4);
        let (b, _) = decode(&q, &k, &v, &shape, Plan::PackedSplit, 4);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn the_merge_of_the_chapter() {
        // Chapter 8's row, scores [1, 3 | 2, 5] against values [1, 2 | 3, 4], as two
        // chunks merged by log-sum-exp.
        let chunk = |x: [f64; 2], val: [f64; 2]| {
            let m = x[0].max(x[1]);
            let l = (x[0] - m).exp() + (x[1] - m).exp();
            let o = ((x[0] - m).exp() * val[0] + (x[1] - m).exp() * val[1]) / l;
            (o, m + l.ln())
        };
        let (oa, lse_a) = chunk([1.0, 3.0], [1.0, 2.0]);
        let (ob, lse_b) = chunk([2.0, 5.0], [3.0, 4.0]);
        let lse = lse_b + (1.0 + (lse_a - lse_b).exp()).ln();
        let out = oa * (lse_a - lse).exp() + ob * (lse_b - lse).exp();
        assert!((oa - 1.8808).abs() < 1e-4 && (ob - 3.9526).abs() < 1e-4);
        assert!((lse - 5.1852).abs() < 1e-4);
        assert!((out - 3.6881).abs() < 1e-4);
    }
}
