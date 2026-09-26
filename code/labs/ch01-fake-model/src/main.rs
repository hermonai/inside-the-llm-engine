//! Chapter 1 lab: a fake model with real costs.
//!
//! A decode step multiplies every weight matrix by one activation vector per
//! sequence. This program keeps one weight matrix the size of a small model
//! (512 MiB of f32 by default) and runs "decode steps" against it for rising
//! batch sizes, checking every kernel against an oracle before timing it.
//!
//!     cargo run --release -p ch01-fake-model -- [--mib 512] [--threads 1] [--steps 5]
//!
//! The weights are random, so the numbers mean nothing; the bytes are real,
//! so the costs are.

use std::time::Instant;

use capstone::kernels::{matmul_rows, matmul_rows_simple, split_rows};

/// Width of one weight row: the model width of a 7-billion-parameter model.
pub const COLS: usize = 4096;

/// Deterministic pseudo-random values in [-0.5, 0.5).
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Checks the fast kernel against two oracles on `sample` rows: the
/// one-output-at-a-time kernel (must agree bit for bit) and a float64 dot
/// product (must agree to a relative error of 1e-5 of the row's scale).
pub fn check(
    w: &[f32],
    rows: usize,
    x: &[f32],
    b: usize,
    y: &[f32],
    sample: usize,
) -> Result<(), String> {
    let step = (rows / sample).max(1);
    for r in (0..rows).step_by(step) {
        let w_row = &w[r * COLS..(r + 1) * COLS];
        let mut one = vec![0.0f32; b];
        matmul_rows_simple(w_row, 1, COLS, x, b, &mut one);
        for t in 0..b {
            let x_t = &x[t * COLS..(t + 1) * COLS];
            let fast = y[t * rows + r];
            if fast.to_bits() != one[t].to_bits() {
                return Err(format!(
                    "row {r}, sequence {t}: tiled {fast} vs one-at-a-time {}",
                    one[t]
                ));
            }
            let exact: f64 = w_row
                .iter()
                .zip(x_t)
                .map(|(&a, &c)| a as f64 * c as f64)
                .sum();
            let scale: f64 = w_row
                .iter()
                .zip(x_t)
                .map(|(&a, &c)| (a as f64 * c as f64).abs())
                .sum();
            if (fast as f64 - exact).abs() > 1e-5 * scale.max(1e-30) {
                return Err(format!("row {r}, sequence {t}: {fast} vs float64 {exact}"));
            }
        }
    }
    Ok(())
}

pub struct Point {
    pub batch: usize,
    pub step_ms: f64,
    pub tokens_per_s: f64,
    pub gb_per_s: f64,
    pub gflop_per_s: f64,
}

/// Runs `steps` decode steps of batch `b` and returns the fastest.
pub fn measure(
    w: &[f32],
    rows: usize,
    b: usize,
    threads: usize,
    steps: usize,
) -> Result<Point, String> {
    let x = fill(b * COLS, 1000 + b as u64);
    let mut y = vec![0.0f32; b * rows];
    split_rows(matmul_rows, w, rows, COLS, &x, b, &mut y, threads);
    check(w, rows, &x, b, &y, 64)?;
    let mut best = f64::INFINITY;
    for _ in 0..steps {
        let t = Instant::now();
        split_rows(matmul_rows, w, rows, COLS, &x, b, &mut y, threads);
        best = best.min(t.elapsed().as_secs_f64());
    }
    std::hint::black_box(&y);
    let bytes = (w.len() * 4) as f64;
    let flops = 2.0 * w.len() as f64 * b as f64;
    Ok(Point {
        batch: b,
        step_ms: best * 1e3,
        tokens_per_s: b as f64 / best,
        gb_per_s: bytes / best / 1e9,
        gflop_per_s: flops / best / 1e9,
    })
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let mib = arg("--mib", 512);
    let threads = arg("--threads", 1);
    let steps = arg("--steps", 5);
    let rows = mib * 1024 * 1024 / 4 / COLS;
    println!(
        "weights: {rows} rows x {COLS} f32 = {:.0} MB; threads {threads}; best of {steps} steps",
        (rows * COLS * 4) as f64 / 1e6
    );
    println!("predict first: at B = 1 each step reads every weight once, so tokens/s = bandwidth / {:.0} MB", (rows * COLS * 4) as f64 / 1e6);
    let w = fill(rows * COLS, 7);
    println!(
        "{:>4} {:>10} {:>12} {:>14} {:>10} {:>10}",
        "B", "step ms", "tokens/s", "per sequence", "GB/s", "GFLOP/s"
    );
    for b in [1usize, 2, 4, 8, 16, 32, 64] {
        match measure(&w, rows, b, threads, steps) {
            Ok(p) => println!(
                "{:>4} {:>10.1} {:>12.1} {:>14.1} {:>10.1} {:>10.1}",
                p.batch,
                p.step_ms,
                p.tokens_per_s,
                1e3 / p.step_ms,
                p.gb_per_s,
                p.gflop_per_s
            ),
            Err(e) => {
                eprintln!("oracle check failed at B = {b}: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_batch_size_matches_both_oracles() {
        let rows = 96;
        let w = fill(rows * COLS, 3);
        for b in [1, 3, 4, 7, 8] {
            for threads in [1, 2] {
                measure(&w, rows, b, threads, 1).unwrap();
            }
        }
    }
}
