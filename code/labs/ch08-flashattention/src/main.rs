//! Chapter 8 lab: attention as a memory problem.
//!
//! Causal attention for one head of width 64, three ways:
//!
//! 1. `naive`: write the S x S score matrix, softmax each row, multiply by V;
//! 2. `online`: one pass over the keys per query row, keeping a running
//!    maximum, denominator and output (the online softmax);
//! 3. `tiled`: blocks of 64 queries against blocks of 64 keys, each block's
//!    scores held in a 64 x 64 buffer and every row's running state updated
//!    once per block, as FlashAttention does.
//!
//! Every result is checked against float64 attention on sampled rows (the
//! last row, which attends to every key, always included) before it is timed.
//! The program reports each version's temporary memory beside its time.
//!
//! `--forget-rescale` breaks the tiled kernel on purpose: it adopts a new
//! running maximum without rescaling what it has accumulated. `--threshold T`
//! does what FlashAttention-4 does: it keeps the old maximum unless a block
//! raises it by more than T, which is exact and skips most rescalings. The
//! default, ln 256, lets the exponentials grow by at most a factor of 256,
//! FlashAttention-4's usual setting. The threshold demonstration also runs on
//! "ramp" keys, scaled up along the sequence so that later blocks keep raising
//! every row's maximum.
//!
//!     cargo run --release -p ch08-flashattention -- [--max 8192] [--threshold 5.545]

use std::time::Instant;

pub const D: usize = 64;
pub const BLOCK: usize = 64;

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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Materializes all S x S scores (the causal half is used), then softmax, then P V.
pub fn naive(q: &[f32], k: &[f32], v: &[f32], s: usize) -> (Vec<f32>, usize) {
    let scale = 1.0 / (D as f32).sqrt();
    let mut scores = vec![0.0f32; s * s]; // the matrix that "exists only to be summed away"
    for i in 0..s {
        for j in 0..=i {
            scores[i * s + j] = dot(&q[i * D..(i + 1) * D], &k[j * D..(j + 1) * D]) * scale;
        }
    }
    let mut out = vec![0.0f32; s * D];
    for i in 0..s {
        let row = &mut scores[i * s..i * s + i + 1];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        let o = &mut out[i * D..(i + 1) * D];
        for (j, &p) in row.iter().enumerate() {
            for (oc, vc) in o.iter_mut().zip(&v[j * D..(j + 1) * D]) {
                *oc += p / sum * vc;
            }
        }
    }
    (out, s * s * 4)
}

/// One pass per query row: a running maximum `m`, denominator `l` and output `o`.
pub fn online(q: &[f32], k: &[f32], v: &[f32], s: usize) -> (Vec<f32>, usize) {
    let scale = 1.0 / (D as f32).sqrt();
    let mut out = vec![0.0f32; s * D];
    for i in 0..s {
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        let o = &mut out[i * D..(i + 1) * D];
        for j in 0..=i {
            let x = dot(&q[i * D..(i + 1) * D], &k[j * D..(j + 1) * D]) * scale;
            let m_new = m.max(x);
            let (old, new) = ((m - m_new).exp(), (x - m_new).exp());
            l = l * old + new;
            for (oc, vc) in o.iter_mut().zip(&v[j * D..(j + 1) * D]) {
                *oc = *oc * old + new * vc;
            }
            m = m_new;
        }
        o.iter_mut().for_each(|x| *x /= l);
    }
    (out, (D + 2) * 4)
}

/// How the tiled kernel treats a block that raises a row's maximum.
#[derive(Clone, Copy, PartialEq)]
pub enum Rescale {
    /// Adopt the new maximum and rescale the running sums: FlashAttention.
    Always,
    /// Keep the old maximum unless the block raises it by more than this:
    /// FlashAttention-4's conditional rescaling. Exact; exponentials stay below e^T.
    Threshold(f32),
    /// Adopt the new maximum but forget to rescale: wrong on purpose.
    Forget,
}

/// Blocks of BLOCK queries against blocks of BLOCK keys. Returns the output,
/// the bytes of temporaries, and how many row rescalings were performed.
pub fn tiled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    rescale: Rescale,
) -> (Vec<f32>, usize, usize) {
    let scale = 1.0 / (D as f32).sqrt();
    let mut out = vec![0.0f32; s * D];
    let mut tile = [[0.0f32; BLOCK]; BLOCK];
    let (mut m, mut l) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    let mut rescalings = 0;
    for qb in (0..s).step_by(BLOCK) {
        let rows = BLOCK.min(s - qb);
        m[..rows].fill(f32::NEG_INFINITY);
        l[..rows].fill(0.0);
        for kb in (0..=qb).step_by(BLOCK) {
            // Block scores, held on "chip"; causal: key j <= query i.
            let cols = BLOCK.min(s - kb);
            for (r, row) in tile.iter_mut().enumerate().take(rows) {
                let i = qb + r;
                for (c, t) in row.iter_mut().enumerate().take(cols) {
                    let j = kb + c;
                    *t = if j <= i {
                        dot(&q[i * D..(i + 1) * D], &k[j * D..(j + 1) * D]) * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            }
            for r in 0..rows {
                let block_max = tile[r][..cols]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if block_max == f32::NEG_INFINITY {
                    continue; // the whole block is masked for this row
                }
                let o = &mut out[(qb + r) * D..(qb + r + 1) * D];
                let raise = block_max > m[r];
                let adopt = match rescale {
                    Rescale::Always | Rescale::Forget => raise,
                    Rescale::Threshold(t) => m[r] == f32::NEG_INFINITY || block_max - m[r] > t,
                };
                if adopt {
                    if rescale != Rescale::Forget && m[r] != f32::NEG_INFINITY {
                        let f = (m[r] - block_max).exp();
                        l[r] *= f;
                        o.iter_mut().for_each(|x| *x *= f);
                        rescalings += 1;
                    }
                    m[r] = block_max;
                }
                for (c, &x) in tile[r][..cols].iter().enumerate() {
                    if x == f32::NEG_INFINITY {
                        continue;
                    }
                    let p = (x - m[r]).exp();
                    l[r] += p;
                    for (oc, vc) in o.iter_mut().zip(&v[(kb + c) * D..(kb + c + 1) * D]) {
                        *oc += p * vc;
                    }
                }
            }
        }
        for r in 0..rows {
            out[(qb + r) * D..(qb + r + 1) * D]
                .iter_mut()
                .for_each(|x| *x /= l[r]);
        }
    }
    (out, (BLOCK * BLOCK + 2 * BLOCK) * 4, rescalings)
}

/// Largest error against float64 attention over `rows` sampled rows, relative
/// to the largest output magnitude.
pub fn error(q: &[f32], k: &[f32], v: &[f32], s: usize, got: &[f32], rows: usize) -> f64 {
    let scale = 1.0 / (D as f64).sqrt();
    let mut worst = 0.0f64;
    let mut peak = 1e-30f64;
    for n in 0..rows {
        let i = if n == 0 { s - 1 } else { (n * 7919) % s };
        let scores: Vec<f64> = (0..=i)
            .map(|j| {
                (0..D)
                    .map(|c| q[i * D + c] as f64 * k[j * D + c] as f64)
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
        let sum: f64 = w.iter().sum();
        for c in 0..D {
            let exact: f64 = w
                .iter()
                .enumerate()
                .map(|(j, p)| p * v[j * D + c] as f64)
                .sum::<f64>()
                / sum;
            worst = worst.max((got[i * D + c] as f64 - exact).abs());
            peak = peak.max(exact.abs());
        }
    }
    worst / peak
}

/// Scales key j by 1 + 4j/S, so that scores grow along the sequence and later
/// blocks keep raising each row's running maximum.
pub fn ramp(k: &mut [f32], s: usize) {
    for (j, key) in k.chunks_mut(D).enumerate() {
        let f = 1.0 + 4.0 * j as f32 / s as f32;
        key.iter_mut().for_each(|x| *x *= f);
    }
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let max: usize = arg("--max").and_then(|v| v.parse().ok()).unwrap_or(8192);
    let threshold: f32 = arg("--threshold")
        .and_then(|v| v.parse().ok())
        .unwrap_or(256f32.ln());
    let forget = std::env::args().any(|a| a == "--forget-rescale");
    let tol = 1e-4;
    println!(
        "causal attention, one head of width {D}, blocks of {BLOCK}; errors relative to float64"
    );
    println!(
        "{:>6} {:>10} {:>13} {:>10} {:>12} {:>10} {:>12} {:>12}",
        "S",
        "naive ms",
        "naive temp",
        "online ms",
        "online temp",
        "tiled ms",
        "tiled temp",
        "rescalings"
    );
    let mut s = 256;
    while s <= max {
        let (q, k, v) = (
            fill(s * D, 1, 4.0),
            fill(s * D, 2, 4.0),
            fill(s * D, 3, 1.0),
        );
        let t = Instant::now();
        let (a, a_mem) = naive(&q, &k, &v, s);
        let t_naive = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (b, b_mem) = online(&q, &k, &v, s);
        let t_online = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (c, c_mem, n_resc) = tiled(&q, &k, &v, s, Rescale::Always);
        let t_tiled = t.elapsed().as_secs_f64();
        for (name, out) in [("naive", &a), ("online", &b), ("tiled", &c)] {
            let e = error(&q, &k, &v, s, out, 8);
            if e > tol {
                eprintln!("oracle failed: {name} at S = {s}, error {e:.2e}");
                std::process::exit(1);
            }
        }
        let mib = |b: usize| format!("{:.2} MiB", b as f64 / (1 << 20) as f64);
        println!(
            "{s:>6} {:>10.1} {:>13} {:>10.1} {:>12} {:>10.1} {:>12} {:>12}",
            t_naive * 1e3,
            mib(a_mem),
            t_online * 1e3,
            format!("{b_mem} B"),
            t_tiled * 1e3,
            format!("{c_mem} B"),
            n_resc
        );
        s *= 2;
    }
    let s = 2048;
    let (q, mut k, v) = (
        fill(s * D, 1, 4.0),
        fill(s * D, 2, 4.0),
        fill(s * D, 3, 1.0),
    );
    for keys in ["ordinary", "ramp"] {
        if keys == "ramp" {
            ramp(&mut k, s);
        }
        let (_, _, always) = tiled(&q, &k, &v, s, Rescale::Always);
        let (t_out, _, some) = tiled(&q, &k, &v, s, Rescale::Threshold(threshold));
        println!(
            "S = {s}, {keys} keys: rescaling only when a block raises the maximum by more than \
             {threshold:.3}: {some} rescalings instead of {always}, error {:.1e}",
            error(&q, &k, &v, s, &t_out, 8)
        );
    }
    let k = fill(s * D, 2, 4.0);
    if forget {
        let (f_out, _, _) = tiled(&q, &k, &v, s, Rescale::Forget);
        println!(
            "S = {s}: adopting new maxima without rescaling: error {:.1e} (tolerance {tol:.0e})",
            error(&q, &k, &v, s, &f_out, 8)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_agree_with_float64() {
        for s in [1, 7, 64, 65, 200] {
            let (q, k, v) = (
                fill(s * D, 1, 4.0),
                fill(s * D, 2, 4.0),
                fill(s * D, 3, 1.0),
            );
            assert!(error(&q, &k, &v, s, &naive(&q, &k, &v, s).0, 8) < 1e-4);
            assert!(error(&q, &k, &v, s, &online(&q, &k, &v, s).0, 8) < 1e-4);
            assert!(error(&q, &k, &v, s, &tiled(&q, &k, &v, s, Rescale::Always).0, 8) < 1e-4);
            assert!(
                error(
                    &q,
                    &k,
                    &v,
                    s,
                    &tiled(&q, &k, &v, s, Rescale::Threshold(256f32.ln())).0,
                    8
                ) < 1e-4
            );
        }
    }

    #[test]
    fn forgetting_the_rescale_is_wrong_once_a_later_block_raises_the_maximum() {
        let s = 256;
        let (q, k, v) = (
            fill(s * D, 1, 4.0),
            fill(s * D, 2, 4.0),
            fill(s * D, 3, 1.0),
        );
        assert!(error(&q, &k, &v, s, &tiled(&q, &k, &v, s, Rescale::Forget).0, 8) > 1e-3);
    }

    #[test]
    fn the_threshold_is_exact_and_rescales_less_on_ramp_keys() {
        let s = 512;
        let (q, mut k, v) = (
            fill(s * D, 1, 4.0),
            fill(s * D, 2, 4.0),
            fill(s * D, 3, 1.0),
        );
        ramp(&mut k, s);
        let (_, _, always) = tiled(&q, &k, &v, s, Rescale::Always);
        let (out, _, some) = tiled(&q, &k, &v, s, Rescale::Threshold(256f32.ln()));
        assert!(error(&q, &k, &v, s, &out, 8) < 1e-4);
        assert!(some < always);
    }

    #[test]
    fn the_worked_example_of_the_chapter() {
        // scores [1, 3, 2, 5] against values [1, 2, 3, 4], in blocks of two
        let (x, vals) = ([1.0f64, 3.0, 2.0, 5.0], [1.0f64, 2.0, 3.0, 4.0]);
        let (mut m, mut l, mut o) = (f64::NEG_INFINITY, 0.0f64, 0.0f64);
        for blk in [[0usize, 1], [2, 3]] {
            let bm = blk.iter().map(|&j| x[j]).fold(f64::NEG_INFINITY, f64::max);
            let m_new = m.max(bm);
            let f = (m - m_new).exp();
            l *= f;
            o *= f;
            for &j in &blk {
                l += (x[j] - m_new).exp();
                o += (x[j] - m_new).exp() * vals[j];
            }
            m = m_new;
        }
        let full: f64 = x
            .iter()
            .zip(&vals)
            .map(|(a, b)| (a - 5.0).exp() * b)
            .sum::<f64>()
            / x.iter().map(|a| (a - 5.0).exp()).sum::<f64>();
        assert!((o / l - full).abs() < 1e-12);
        assert!((o / l - 3.6881).abs() < 1e-4);
    }
}
