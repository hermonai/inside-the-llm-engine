//! Chapter 7 lab: tile a GEMM and watch it climb.
//!
//! Part 1 climbs a ladder of kernels for C = A B with square f32 matrices,
//! each rung changing only the order of the same multiply-adds:
//!
//! 1. `naive`: the textbook i, j, k loops; the inner loop walks a column of B;
//! 2. `ikj`: the loops reordered so the inner loop walks rows contiguously;
//! 3. `blocked`: ikj inside 64 x 64 x 64 blocks that stay in cache;
//! 4. `tiled`: a register tile of 4 rows x 16 columns of C kept in
//!    accumulators while k runs, so every loaded element of B is used four
//!    times and every element of A sixteen times;
//! 5. `tiled`, split by rows across threads.
//!
//! Part 2 fixes a 4,096 x 4,096 weight matrix (64 MiB of f32) and multiplies it
//! by N vectors, N = 1 to 256, with two kernels: the tiled GEMM, whose tile is
//! 16 columns wide whatever N is, and the capstone engine's matrix-vector
//! kernel, which streams each weight row once and dots it with every vector.
//!
//! Every result is checked against a float64 product on a sample of entries
//! before it is timed.
//!
//!     cargo run --release -p ch07-gemm -- [--size 512] [--threads 4]

use std::time::Instant;

use capstone::kernels::{matmul_rows, split_rows};

pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Row-major `a` (m x k) times `b` (k x n) into `c` (m x n), textbook order.
pub fn naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/// The same products with the loops in i, k, j order: the inner loop reads a
/// row of B and updates a row of C, both contiguous.
pub fn ikj(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            let (brow, crow) = (&b[p * n..(p + 1) * n], &mut c[i * n..(i + 1) * n]);
            for (cv, bv) in crow.iter_mut().zip(brow) {
                *cv += aip * bv;
            }
        }
    }
}

/// ikj inside blocks of `bs` so each block of A, B and C stays in cache.
pub fn blocked(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, bs: usize) {
    c.fill(0.0);
    for i0 in (0..m).step_by(bs) {
        for p0 in (0..k).step_by(bs) {
            for j0 in (0..n).step_by(bs) {
                for i in i0..(i0 + bs).min(m) {
                    for p in p0..(p0 + bs).min(k) {
                        let aip = a[i * k + p];
                        let (j1, j2) = (j0, (j0 + bs).min(n));
                        let brow = &b[p * n + j1..p * n + j2];
                        let crow = &mut c[i * n + j1..i * n + j2];
                        for (cv, bv) in crow.iter_mut().zip(brow) {
                            *cv += aip * bv;
                        }
                    }
                }
            }
        }
    }
}

pub const TR: usize = 4; // rows of C per register tile
pub const TC: usize = 16; // columns of C per register tile

/// Register-tiled product of rows `i0..i1` of A. `n` must be a multiple of TC
/// and `i1 - i0` a multiple of TR; callers pad. The 4 x 16 accumulators stay in
/// registers for the whole k loop: per k step the tile loads 16 values of B and
/// 4 of A and performs 64 multiply-adds.
pub fn tiled_rows(a: &[f32], b: &[f32], c: &mut [f32], i0: usize, i1: usize, k: usize, n: usize) {
    for i in (i0..i1).step_by(TR) {
        for j in (0..n).step_by(TC) {
            let mut acc = [[0.0f32; TC]; TR];
            for p in 0..k {
                let brow: &[f32; TC] = b[p * n + j..p * n + j + TC].try_into().unwrap();
                for (r, row) in acc.iter_mut().enumerate() {
                    let av = a[(i + r) * k + p];
                    for (x, bv) in row.iter_mut().zip(brow) {
                        *x += av * bv;
                    }
                }
            }
            for (r, row) in acc.iter().enumerate() {
                c[(i - i0 + r) * n + j..(i - i0 + r) * n + j + TC].copy_from_slice(row);
            }
        }
    }
}

/// The tiled kernel for any shape: pads N up to the tile width and splits the
/// rows of A among `threads`.
pub fn tiled(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    assert_eq!(m % TR, 0, "rows must be a multiple of the tile");
    let np = n.div_ceil(TC) * TC;
    let bp: Vec<f32> = if np == n {
        b.to_vec()
    } else {
        let mut v = vec![0.0f32; k * np];
        for p in 0..k {
            v[p * np..p * np + n].copy_from_slice(&b[p * n..(p + 1) * n]);
        }
        v
    };
    let mut cp = vec![0.0f32; m * np];
    let per = (m / TR).div_ceil(threads) * TR;
    std::thread::scope(|s| {
        for (t, chunk) in cp.chunks_mut(per * np).enumerate() {
            let (bp, i0) = (&bp, t * per);
            let i1 = (i0 + per).min(m);
            s.spawn(move || tiled_rows(a, bp, chunk, i0, i1, k, np));
        }
    });
    if np == n {
        return cp;
    }
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        c[i * n..(i + 1) * n].copy_from_slice(&cp[i * np..i * np + n]);
    }
    c
}

/// The capstone engine's decode kernel for the same product: weights stored as
/// rows (the transposed layout of a weight matrix), each row streamed once and
/// dotted with every vector. Returns C (m x n) in row-major order.
pub fn rowwise(w: &[f32], x: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    split_rows(matmul_rows, w, m, k, x, n, &mut y, threads);
    let mut c = vec![0.0f32; m * n];
    for t in 0..n {
        for i in 0..m {
            c[i * n + t] = y[t * m + i];
        }
    }
    c
}

/// Checks `c` against a float64 product on `samples` entries spread over C.
pub fn check(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
    samples: usize,
) -> Result<(), String> {
    for s in 0..samples {
        let (i, j) = ((s * 7919) % m, (s * 104_729) % n);
        let (mut exact, mut scale) = (0.0f64, 0.0f64);
        for p in 0..k {
            let prod = a[i * k + p] as f64 * b[p * n + j] as f64;
            exact += prod;
            scale += prod.abs();
        }
        let got = c[i * n + j] as f64;
        if (got - exact).abs() > 1e-5 * scale.max(1e-30) {
            return Err(format!("C[{i}][{j}] = {got}, float64 gives {exact}"));
        }
    }
    Ok(())
}

/// Fraction of elements identical to the bit, and the largest difference
/// relative to the largest output (outputs near zero make element-wise
/// relative gaps meaningless).
pub fn bits_against(x: &[f32], y: &[f32]) -> (f64, f64) {
    let same = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let scale = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let gap = x
        .iter()
        .zip(y)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale;
    (same as f64 / x.len() as f64, gap as f64)
}

fn best_secs(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
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
    let size = arg("--size", 512);
    let threads = arg("--threads", 4);
    let fail = |e: String| -> ! {
        eprintln!("oracle failed: {e}");
        std::process::exit(1)
    };

    println!("Part 1: C = A B, {size} x {size} f32, best of 3 (GFLOP/s)");
    let (a, b) = (fill(size * size, 1), fill(size * size, 2));
    let flops = 2.0 * (size * size * size) as f64;
    let mut c = vec![0.0f32; size * size];
    type Kernel<'a> = Box<dyn Fn(&mut Vec<f32>) + 'a>;
    let rungs: Vec<(&str, Kernel)> = vec![
        (
            "naive (i, j, k)",
            Box::new(|c: &mut Vec<f32>| naive(&a, &b, c, size, size, size)),
        ),
        (
            "reordered (i, k, j)",
            Box::new(|c: &mut Vec<f32>| ikj(&a, &b, c, size, size, size)),
        ),
        (
            "blocked 64",
            Box::new(|c: &mut Vec<f32>| blocked(&a, &b, c, size, size, size, 64)),
        ),
        (
            "register tile 4 x 16",
            Box::new(|c: &mut Vec<f32>| *c = tiled(&a, &b, size, size, size, 1)),
        ),
        (
            "register tile, threads",
            Box::new(|c: &mut Vec<f32>| *c = tiled(&a, &b, size, size, size, threads)),
        ),
    ];
    let mut first = 0.0;
    let mut reference: Vec<f32> = Vec::new();
    for (name, run) in &rungs {
        run(&mut c);
        check(&a, &b, &c, size, size, size, 64).unwrap_or_else(|e| fail(format!("{name}: {e}")));
        if reference.is_empty() {
            reference = c.clone();
        }
        let (same, _) = bits_against(&reference, &c);
        let secs = best_secs(3, || run(&mut c));
        let g = flops / secs / 1e9;
        if first == 0.0 {
            first = g;
        }
        println!(
            "  {name:<24} {g:8.1}   ({:.0}x the naive loop; {:.1}% of elements bit-identical to it)",
            g / first,
            100.0 * same
        );
    }

    println!("Part 2: W (4096 x 4096 f32, 64 MiB) times N vectors, {threads} threads, best of 3");
    println!(
        "  {:>4} {:>13} {:>13} {:>13} {:>13} {:>14}",
        "N", "rowwise ms", "GFLOP/s", "tiled ms", "GFLOP/s", "same bits"
    );
    let (m, k) = (4096, 4096);
    let w = fill(m * k, 3);
    for n in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let x = fill(n * k, 10 + n as u64); // n vectors, one per row: the decode layout
        let mut xt = vec![0.0f32; k * n]; // the same vectors as the columns of B
        for t in 0..n {
            for p in 0..k {
                xt[p * n + t] = x[t * k + p];
            }
        }
        let cr = rowwise(&w, &x, m, k, n, threads);
        let ct = tiled(&w, &xt, m, k, n, threads);
        check(&w, &xt, &cr, m, k, n, 32).unwrap_or_else(|e| fail(format!("rowwise N={n}: {e}")));
        check(&w, &xt, &ct, m, k, n, 32).unwrap_or_else(|e| fail(format!("tiled N={n}: {e}")));
        let tr = best_secs(3, || drop(rowwise(&w, &x, m, k, n, threads)));
        let tt = best_secs(3, || drop(tiled(&w, &xt, m, k, n, threads)));
        let f = 2.0 * (m * k * n) as f64;
        let (same, gap) = bits_against(&ct, &cr);
        println!(
            "  {n:>4} {:>13.2} {:>13.1} {:>13.2} {:>13.1} {:>6.1}% ({gap:.0e})",
            tr * 1e3,
            f / tr / 1e9,
            tt * 1e3,
            f / tt / 1e9,
            100.0 * same
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rung_agrees_with_float64() {
        let (m, k, n) = (36, 40, 48);
        let (a, b) = (fill(m * k, 1), fill(k * n, 2));
        let mut c = vec![0.0f32; m * n];
        naive(&a, &b, &mut c, m, k, n);
        check(&a, &b, &c, m, k, n, 50).unwrap();
        ikj(&a, &b, &mut c, m, k, n);
        check(&a, &b, &c, m, k, n, 50).unwrap();
        blocked(&a, &b, &mut c, m, k, n, 16);
        check(&a, &b, &c, m, k, n, 50).unwrap();
        for threads in [1, 3] {
            check(&a, &b, &tiled(&a, &b, m, k, n, threads), m, k, n, 50).unwrap();
        }
    }

    #[test]
    fn skinny_shapes_agree_both_ways() {
        let (m, k) = (64, 96);
        let w = fill(m * k, 4);
        for n in [1, 3, 17] {
            let x = fill(n * k, 5);
            let mut xt = vec![0.0f32; k * n];
            for t in 0..n {
                for p in 0..k {
                    xt[p * n + t] = x[t * k + p];
                }
            }
            check(&w, &xt, &rowwise(&w, &x, m, k, n, 2), m, k, n, 40).unwrap();
            check(&w, &xt, &tiled(&w, &xt, m, k, n, 2), m, k, n, 40).unwrap();
        }
    }
}
