//! Chapter 3 lab: draw your own roofline.
//!
//! Measures the two roofs of this machine's CPU and the points between them:
//!
//! 1. the bandwidth roof: summing (read) and copying arrays far larger than
//!    the caches, with 1 to `--threads` threads;
//! 2. two compute roofs: a loop of independent fused multiply-adds (what the
//!    arithmetic units can do) and the capstone engine's matrix kernel on
//!    weights that fit in cache (what this book's code does);
//! 3. the sweep: decode steps --- one weight matrix in memory times B
//!    activation vectors --- for B = 1 to 256, each checked against two
//!    oracles before it is timed.
//!
//! From roofs 1 and 2 it predicts the batch at which the sweep stops being
//! free, then prints the sweep so you can see whether it bends there.
//!
//!     cargo run --release -p ch03-roofline -- [--threads 4] [--mib 256] [--csv out.csv]

use std::time::Instant;

use capstone::kernels::{matmul_rows, matmul_rows_simple, split_rows};

/// Width of one weight row, as in Chapter 1's lab.
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

/// Sum of a slice with sixteen independent running sums, so that the adds
/// never wait for each other and the loop is limited by loads, not latency.
pub fn sum16(v: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    let chunks = v.chunks_exact(16);
    let tail: f32 = chunks.remainder().iter().sum();
    for c in chunks {
        for (a, x) in acc.iter_mut().zip(c) {
            *a += x;
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Sum with one running total: each addition waits for the one before it,
/// so this loop is limited by the adder's latency, not by memory.
pub fn sum1(v: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for x in v {
        total += x;
    }
    total
}

/// Runs `work` on `threads` equal parts of `0..n` at once and returns the
/// fastest of `reps` wall-clock times in seconds.
fn timed_parts(n: usize, threads: usize, reps: usize, work: impl Fn(usize, usize) + Sync) -> f64 {
    let per = n.div_ceil(threads);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::thread::scope(|s| {
            for i in 0..threads {
                let (lo, hi) = (i * per, ((i + 1) * per).min(n));
                let work = &work;
                s.spawn(move || work(lo, hi));
            }
        });
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Read bandwidth in GB/s: every thread sums its share of `buf`.
pub fn read_gbps(buf: &[f32], threads: usize, reps: usize) -> f64 {
    let secs = timed_parts(buf.len(), threads, reps, |lo, hi| {
        std::hint::black_box(sum16(&buf[lo..hi]));
    });
    (buf.len() * 4) as f64 / secs / 1e9
}

/// Copy bandwidth in GB/s, counting the bytes read and the bytes written.
pub fn copy_gbps(src: &[f32], dst: &mut [f32], threads: usize, reps: usize) -> f64 {
    let n = src.len();
    let dst_addr = dst.as_mut_ptr() as usize;
    let secs = timed_parts(n, threads, reps, |lo, hi| {
        // SAFETY: the parts are disjoint, so no two threads write the same element.
        let d = unsafe { std::slice::from_raw_parts_mut((dst_addr as *mut f32).add(lo), hi - lo) };
        d.copy_from_slice(&src[lo..hi]);
    });
    std::hint::black_box(&dst);
    (2 * n * 4) as f64 / secs / 1e9
}

/// Peak arithmetic in GFLOP/s: each thread keeps 64 independent running
/// sums and adds a product to each, `iters` times. On AArch64 the update is
/// one fused multiply-add (2 FLOPs), which the base instruction set has; on
/// other targets it is a separate multiply and add, because the baseline
/// x86-64 instruction set has no fused form.
pub fn fma_gflops(threads: usize, iters: usize, reps: usize) -> f64 {
    let secs = timed_parts(threads, threads, reps, |_, _| {
        let x = std::hint::black_box(1e-7f32);
        let m = std::hint::black_box(0.5f32);
        let mut acc = [0.0f32; 64];
        for _ in 0..iters {
            for a in acc.iter_mut() {
                #[cfg(target_arch = "aarch64")]
                {
                    *a = x.mul_add(m, *a);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    *a += x * m;
                }
            }
        }
        std::hint::black_box(acc);
    });
    (threads * iters * 64 * 2) as f64 / secs / 1e9
}

/// The capstone kernel's own compute rate in GFLOP/s at batch `b`, with
/// nothing waiting for memory. Each thread multiplies a private 1 MiB slice
/// of weights by the same `b` shared vectors, `reps` times, as the sweep's
/// threads do; the weights and vectors stay in the shared cache up to about
/// B = 128. The threads start together and the slowest one's time is the time.
pub fn kernel_gflops(threads: usize, b: usize, reps: usize) -> f64 {
    const ROWS: usize = 64; // 64 x 4,096 f32 = 1 MiB
    let x = fill(b * COLS, 30);
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = std::sync::Barrier::new(threads);
        let secs = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|i| {
                    let (start, x) = (&start, &x);
                    s.spawn(move || {
                        let w = fill(ROWS * COLS, 20 + i as u64);
                        let mut y = vec![0.0f32; b * ROWS];
                        matmul_rows(&w, ROWS, COLS, x, b, &mut y); // warm the cache
                        start.wait();
                        let t = Instant::now();
                        for _ in 0..reps {
                            matmul_rows(&w, ROWS, COLS, x, b, &mut y);
                        }
                        std::hint::black_box(&y);
                        t.elapsed().as_secs_f64()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .fold(0.0, f64::max)
        });
        best = best.min(secs);
    }
    (threads * reps * 2 * ROWS * COLS * b) as f64 / best / 1e9
}

/// Checks the fast kernel against two oracles on `sample` rows: the
/// one-output-at-a-time kernel (bit for bit) and a float64 dot product
/// (relative 1e-5 of the row's scale).
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
                    "row {r}, vector {t}: {fast} vs one-at-a-time {}",
                    one[t]
                ));
            }
            let (exact, scale) =
                w_row
                    .iter()
                    .zip(x_t)
                    .fold((0.0f64, 0.0f64), |(e, s), (&a, &c)| {
                        let p = a as f64 * c as f64;
                        (e + p, s + p.abs())
                    });
            if (fast as f64 - exact).abs() > 1e-5 * scale.max(1e-30) {
                return Err(format!("row {r}, vector {t}: {fast} vs float64 {exact}"));
            }
        }
    }
    Ok(())
}

/// One decode step: `rows` x COLS weights times `b` vectors, checked, then
/// the best of `reps` timings. Returns (seconds, FLOPs, weight bytes).
pub fn step(
    w: &[f32],
    rows: usize,
    b: usize,
    threads: usize,
    reps: usize,
) -> Result<(f64, f64, f64), String> {
    let x = fill(b * COLS, 1000 + b as u64);
    let mut y = vec![0.0f32; b * rows];
    split_rows(matmul_rows, w, rows, COLS, &x, b, &mut y, threads);
    check(w, rows, &x, b, &y, 32)?;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        split_rows(matmul_rows, w, rows, COLS, &x, b, &mut y, threads);
        best = best.min(t.elapsed().as_secs_f64());
    }
    std::hint::black_box(&y);
    Ok((
        best,
        2.0 * (rows * COLS * b) as f64,
        (rows * COLS * 4) as f64,
    ))
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let threads: usize = arg("--threads", 4);
    let mib: usize = arg("--mib", 256);
    let csv_path: String = arg("--csv", String::new());
    let mut csv = vec!["kind,x,intensity_flop_per_byte,gflop_per_s,gb_per_s,step_ms".to_string()];
    let fail = |e: String| -> ! {
        eprintln!("oracle check failed: {e}");
        std::process::exit(1)
    };

    println!("1. bandwidth roof: {mib} MiB arrays (the default, 256, is far larger than a laptop's caches)");
    let n = mib * 1024 * 1024 / 4;
    let big = fill(n, 11);
    let (src, mut dst) = (fill(n / 2, 12), vec![0.0f32; n / 2]);
    let mut read_roof: f64 = 0.0;
    for t in 1..=threads {
        let (r, c) = (read_gbps(&big, t, 5), copy_gbps(&src, &mut dst, t, 5));
        println!(
            "   {t} thread{}: read {r:6.1} GB/s   copy {c:6.1} GB/s",
            if t > 1 { "s" } else { " " }
        );
        csv.push(format!("read,{t},,,{r:.2},"));
        csv.push(format!("copy,{t},,,{c:.2},"));
        read_roof = read_roof.max(r);
    }
    let one_sum = {
        let secs = timed_parts(n, 1, 3, |lo, hi| {
            std::hint::black_box(sum1(&big[lo..hi]));
        });
        (n * 4) as f64 / secs / 1e9
    };
    println!("   1 thread, one running sum instead of sixteen: {one_sum:5.1} GB/s");
    csv.push(format!("read_one_sum,1,,,{one_sum:.2},"));
    assert_eq!(dst, src, "the copy must be exact");
    drop((big, src, dst));

    println!("2. compute roofs, {threads} threads");
    let peak = fma_gflops(threads, 20_000_000, 5);
    println!("   independent multiply-adds: {peak:6.1} GFLOP/s");
    csv.push(format!("fma_peak,{threads},,{peak:.2},,"));
    let batches = [1usize, 2, 4, 8, 16, 32, 64, 128, 256];
    let in_cache: Vec<f64> = batches
        .iter()
        .map(|&b| kernel_gflops(threads, b, (4096 / b).max(16)))
        .collect();
    let kernel = in_cache.iter().copied().fold(0.0, f64::max);
    println!("   capstone kernel, weights in cache, by batch (GFLOP/s):");
    for (&b, &g) in batches.iter().zip(&in_cache) {
        println!("     B = {b:>3}: {g:6.1}");
        csv.push(format!("kernel_cached,{b},,{g:.2},,"));
    }
    let ridge = kernel / read_roof;
    // f32 weights: B vectors give 2B FLOPs per 4 bytes of weight read.
    println!(
        "   ridge = {kernel:.1} / {read_roof:.1} = {ridge:.2} FLOP/byte; with 4-byte weights \
         I(B) = B/2, so the sweep should stop being free near B = {:.1}",
        2.0 * ridge
    );

    println!("3. the sweep: {mib} MiB of weights times B vectors, {threads} threads, best of 5");
    println!("   predicted step = the longer of (weight bytes / read roof) and (FLOPs / in-cache rate at B)");
    let rows = mib * 1024 * 1024 / 4 / COLS;
    let w = fill(rows * COLS, 7);
    // Freshly written weights are slow to read for the first few passes on
    // some machines (the record shows it on the M1), so warm them first.
    step(&w, rows, 1, threads, 5).unwrap_or_else(|e| fail(e));
    println!(
        "   {:>4} {:>10} {:>9} {:>10} {:>9} {:>9} {:>8}",
        "B", "FLOP/byte", "step ms", "predicted", "GFLOP/s", "GB/s", "bound"
    );
    for (&b, &cached) in batches.iter().zip(&in_cache) {
        let (secs, flops, bytes) = step(&w, rows, b, threads, 5).unwrap_or_else(|e| fail(e));
        let (i, g, bw) = (flops / bytes, flops / secs / 1e9, bytes / secs / 1e9);
        let (t_mem, t_math) = (bytes / read_roof / 1e9, flops / cached / 1e9);
        let bound = if t_mem >= t_math { "memory" } else { "compute" };
        println!(
            "   {b:>4} {i:>10.1} {:>9.2} {:>10.2} {g:>9.1} {bw:>9.1} {bound:>8}",
            secs * 1e3,
            t_mem.max(t_math) * 1e3
        );
        csv.push(format!("sweep,{b},{i:.3},{g:.2},{bw:.2},{:.3}", secs * 1e3));
    }
    if !csv_path.is_empty() {
        std::fs::write(&csv_path, csv.join("\n") + "\n").expect("write the CSV");
        println!("wrote {csv_path}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sum_is_exact_on_small_integers() {
        let v: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        assert_eq!(sum16(&v), 500_500.0); // every partial sum is an integer below 2^24
    }

    #[test]
    fn both_sums_agree_on_small_integers() {
        let v: Vec<f32> = (1..=1000).map(|i| (i % 17) as f32).collect();
        assert_eq!(sum1(&v), sum16(&v));
    }

    #[test]
    fn copying_in_parts_copies_everything() {
        let src = fill(10_001, 5);
        let mut dst = vec![0.0; src.len()];
        copy_gbps(&src, &mut dst, 3, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn every_batch_size_matches_both_oracles() {
        let rows = 64;
        let w = fill(rows * COLS, 3);
        for b in [1, 2, 4, 5, 9, 16] {
            for threads in [1, 3] {
                step(&w, rows, b, threads, 1).unwrap();
            }
        }
    }
}
