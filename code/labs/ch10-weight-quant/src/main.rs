//! Chapter 10 lab: quantize a real weight matrix.
//!
//! Part 1 decodes real K-quant blocks from a GGUF file --- a Q4_K block of
//! layer 0's query projection and a Q6_K block of its value projection --- with
//! decoders transcribed from ggml, and prints what the bytes mean.
//!
//! Part 2 quantizes layer 0's value projection (1,024 x 3,072, stored as Q6_K)
//! to 4 and 3 bits five ways --- round-to-nearest with one scale per tensor, per
//! row and per group of 32; an AWQ-style activation-aware scaling; and GPTQ ---
//! and measures the error in the layer's *output* on real inputs: the token
//! embeddings of the book's own AUTHORING.md, RMS-normalized with the layer's
//! weights, exactly what layer 0 sees. It calibrates on the first 1,024 tokens
//! and evaluates on the other 2,193.
//!
//! Part 3 times a 4,096 x 4,096 matrix times 1, 4 and 16 vectors with the
//! weights in f32; in Q8_0 and in Q4_0, dequantized inside the kernel (Q4_0 two
//! ways: each 32-weight block reduced on its own, or the scale folded into the
//! unpacked weights and each row reduced once); and in Q4_0 against activations
//! quantized to Q8_0 (integer dot products). Each is checked against float64.
//!
//!     cargo run --release -p ch10-weight-quant -- --gguf <llama-3.2-3b.gguf>
//!         [--threads 4] [--parts 1,2,3] [--dump <dir>]

mod gguf;
mod quant;

use gguf::{decode, f16, to_f16, Gguf, Q4_0, Q4_K, Q6_K, Q8_0};
use quant::{awq, gptq, output_error, par_rows, rtn, Grouping};
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

const TOKENS: &str = include_str!("../tokens.txt");

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
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

// ---------------------------------------------------------------- part 1

fn part1(g: &mut Gguf, dump: Option<&str>) -> std::io::Result<()> {
    println!("1. Real K-quant blocks, decoded with ggml's reference layouts");
    for name in ["blk.0.attn_q.weight", "blk.0.attn_v.weight"] {
        let i = g.index(name).expect("tensor present");
        let t = &g.tensors[i];
        let (ty, dims) = (t.ty, t.dims.clone());
        let bpw = t.row_bytes() as f64 * 8.0 / dims[0] as f64;
        println!(
            "\n{name}: {} x {} as {} ({} bytes per row of {}, {bpw:.4} bits per weight)",
            dims[1],
            dims[0],
            gguf::type_name(ty),
            t.row_bytes(),
            dims[0]
        );
        let raw = g.raw_rows(i, 0, 1)?;
        let y = decode(ty, &raw);
        if ty == Q4_K {
            let b = &raw[..144];
            let (d, dmin) = (
                f16(u16::from_le_bytes([b[0], b[1]])),
                f16(u16::from_le_bytes([b[2], b[3]])),
            );
            println!("  first super-block: d = {d:.6e}, dmin = {dmin:.6e}");
            let sm: Vec<String> = (0..8)
                .map(|j| {
                    let (s, m) = gguf::scale_min_k4(j, &b[4..16]);
                    format!("({s},{m})")
                })
                .collect();
            println!(
                "  six-bit (scale, min) of its 8 sub-blocks: {}",
                sm.join(" ")
            );
            let codes: Vec<u8> = b[16..24].iter().map(|q| q & 0xf).collect();
            println!("  first 8 codes: {codes:?}");
        } else if ty == Q6_K {
            let b = &raw[..210];
            let d = f16(u16::from_le_bytes([b[208], b[209]]));
            let sc: Vec<i8> = b[192..208].iter().map(|&s| s as i8).collect();
            println!(
                "  first super-block: d = {d:.6e}; eight-bit scales of its 16 sub-blocks: {sc:?}"
            );
        }
        let first: Vec<String> = y[..8].iter().map(|v| format!("{v:+.6}")).collect();
        println!("  first 8 weights: {}", first.join(" "));
        if let Some(dir) = dump {
            let rows = g.rows(i, 0, 4)?;
            let path = format!("{dir}/{name}.rows0-3.txt");
            let mut f = std::fs::File::create(&path)?;
            for v in rows {
                writeln!(f, "{:08x}", v.to_bits())?;
            }
            println!("  wrote rows 0-3 as f32 bits to {path}");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- part 2

/// A labelled quantizer: returns the weights, bits per weight and a note.
type Method<'a> = (String, Box<dyn Fn() -> (Vec<f32>, f64, String) + 'a>);

/// Distinct tokens with their counts, and their layer-0 inputs.
fn inputs(
    g: &mut Gguf,
    ids: &[u32],
    emb: usize,
    gamma: &[f32],
    eps: f32,
) -> std::io::Result<(Vec<f32>, Vec<f64>)> {
    let mut counts: BTreeMap<u32, f64> = BTreeMap::new();
    for &t in ids {
        *counts.entry(t).or_insert(0.0) += 1.0;
    }
    let cols = gamma.len();
    let mut x = Vec::with_capacity(counts.len() * cols);
    let mut c = Vec::with_capacity(counts.len());
    for (&t, &n) in &counts {
        let e = g.rows(emb, t as usize, 1)?;
        let ms = e.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cols as f64;
        let inv = 1.0 / ((ms + eps as f64).sqrt());
        x.extend(
            e.iter()
                .zip(gamma)
                .map(|(v, gm)| ((*v as f64) * inv) as f32 * gm),
        );
        c.push(n);
    }
    Ok((x, c))
}

fn part2(g: &mut Gguf, threads: usize) -> std::io::Result<()> {
    println!("\n2. Quantizing layer 0's value projection, error measured in its output");
    let ids: Vec<u32> = TOKENS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(|v| v.parse().expect("token id")))
        .collect();
    let (cal_ids, eval_ids) = ids.split_at(1024.min(ids.len()));
    let emb = g.index("token_embd.weight").expect("token_embd");
    let norm = g.index("blk.0.attn_norm.weight").expect("attn_norm");
    let wv = g.index("blk.0.attn_v.weight").expect("attn_v");
    let eps = g
        .numbers
        .get("llama.attention.layer_norm_rms_epsilon")
        .copied()
        .unwrap_or(1e-5) as f32;
    let gamma = g.rows(norm, 0, 1)?;
    let (cols, rows) = (
        g.tensors[wv].dims[0] as usize,
        g.tensors[wv].dims[1] as usize,
    );
    let w = g.rows(wv, 0, rows)?;
    let (xc, cc) = inputs(g, cal_ids, emb, &gamma, eps)?;
    let (xe, ce) = inputs(g, eval_ids, emb, &gamma, eps)?;
    // A wider calibration set: 4,096 vocabulary entries spread evenly over the
    // first 128,000 IDs (the rest are special tokens).
    let vocab_ids: Vec<u32> = (0..4096u32).map(|i| i * 31 + 7).collect();
    let (xv, cv) = inputs(g, &vocab_ids, emb, &gamma, eps)?;
    let unseen = eval_ids.iter().filter(|t| !cal_ids.contains(t)).count();
    println!(
        "W: {rows} x {cols} (Q6_K in the file, 6.5625 bits per weight, used as the reference)"
    );
    println!(
        "inputs: RMSNorm(embedding) x attn_norm, eps {eps:e}; calibration {} tokens ({} distinct), \
         evaluation {} tokens ({} distinct, {unseen} not in calibration)",
        cal_ids.len(),
        cc.len(),
        eval_ids.len(),
        ce.len()
    );
    let wnorm = w.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    println!(
        "\n{:<34} {:>6} {:>12} {:>12} {:>12} {:>8}",
        "method", "bpw", "weight err", "output (cal)", "output (eval)", "seconds"
    );
    for bits in [8u32, 4, 3] {
        let mut methods: Vec<Method> = Vec::new();
        let wr = &w;
        if bits == 8 {
            methods.push((
                "round-to-nearest, groups of 32".into(),
                Box::new(move || {
                    let (q, b) = rtn(wr, rows, cols, 8, Grouping::Group(32));
                    (q, b, String::new())
                }),
            ));
        } else {
            for (label, gr) in [
                ("round-to-nearest, one scale", Grouping::Tensor),
                ("round-to-nearest, scale per row", Grouping::Row),
                ("round-to-nearest, groups of 32", Grouping::Group(32)),
            ] {
                methods.push((
                    label.into(),
                    Box::new(move || {
                        let (q, b) = rtn(wr, rows, cols, bits, gr);
                        (q, b, String::new())
                    }),
                ));
            }
            let (xcr, ccr) = (&xc, &cc);
            methods.push((
                "AWQ-style scaling, groups of 32".into(),
                Box::new(move || {
                    let (q, alpha, b) = awq(wr, rows, cols, bits, 32, xcr, ccr, threads);
                    (q, b, format!("alpha {alpha:.1}"))
                }),
            ));
            methods.push((
                "GPTQ, groups of 32".into(),
                Box::new(move || {
                    let (q, b) = gptq(wr, rows, cols, bits, 32, xcr, ccr, threads);
                    (q, b, String::new())
                }),
            ));
            let (xvr, cvr) = (&xv, &cv);
            methods.push((
                "GPTQ, 4,096 vocabulary rows".into(),
                Box::new(move || {
                    let (q, b) = gptq(wr, rows, cols, bits, 32, xvr, cvr, threads);
                    (q, b, "calibrated on vocabulary".into())
                }),
            ));
        }
        for (label, run) in methods {
            let t0 = Instant::now();
            let (q, bpw, note) = run();
            let secs = t0.elapsed().as_secs_f64();
            let werr = w
                .iter()
                .zip(&q)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / wnorm;
            let (ec, nc) = output_error(&w, &q, cols, &xc, &cc, threads);
            let (ee, ne) = output_error(&w, &q, cols, &xe, &ce, threads);
            println!(
                "{:<34} {:>6.3} {:>12.4} {:>12.4} {:>12.4} {:>8.1}{}",
                format!("{bits}-bit {label}"),
                bpw,
                werr,
                (ec / nc).sqrt(),
                (ee / ne).sqrt(),
                secs,
                if note.is_empty() {
                    String::new()
                } else {
                    format!("  {note}")
                }
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- part 3

/// ggml's reference Q8_0 and Q4_0 quantizers (quantize_row_q8_0_ref, _q4_0_ref).
fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&to_f16(d).to_le_bytes());
        out.extend(b.iter().map(|v| (v * id).round() as i8 as u8));
    }
    out
}

fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for b in x.chunks_exact(32) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in b {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&to_f16(d).to_le_bytes());
        for j in 0..16 {
            let q0 = ((b[j] * id + 8.5) as i8).min(15) as u8;
            let q1 = ((b[16 + j] * id + 8.5) as i8).min(15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
    out
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for ((s, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }
    acc.iter().sum()
}

#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    F32,
    /// Q8_0 or Q4_0 unpacked to floats with the scale folded in; one reduction per row.
    Q8Deq,
    Q4Deq,
    /// Q4_0 unpacked to floats, each 32-weight block reduced on its own.
    Q4Block,
    /// Q4_0 codes against activations quantized to Q8_0: integer dot products.
    Q4Int,
}

/// y[b][r] for every row r of the weights and vector b.
fn matmul(k: Kernel, w32: &[f32], wq: &[u8], xs: &[f32], n: usize, threads: usize) -> Vec<f32> {
    let batch = xs.len() / n;
    let rowsum = std::sync::Mutex::new(vec![0.0f32; n * batch]);
    // Activations quantized once per call for the integer kernel.
    let xq: Vec<(f32, [i8; 32])> = if k == Kernel::Q4Int {
        xs.chunks_exact(32)
            .map(|b| {
                let amax = b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let d = f16(to_f16(amax / 127.0));
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut q = [0i8; 32];
                for (qj, v) in q.iter_mut().zip(b) {
                    *qj = (v * id).round() as i8;
                }
                (d, q)
            })
            .collect()
    } else {
        Vec::new()
    };
    let per = n.div_ceil(threads.max(1));
    par_rows(threads.max(1), threads.max(1), |t| {
        let (r0, r1) = (t * per, ((t + 1) * per).min(n));
        let mut local = vec![0.0f32; (r1.saturating_sub(r0)) * batch];
        let mut unpacked = [0.0f32; 32];
        let mut ints = [0i8; 32];
        for r in r0..r1 {
            let out = &mut local[(r - r0) * batch..(r - r0 + 1) * batch];
            match k {
                Kernel::F32 => {
                    let row = &w32[r * n..(r + 1) * n];
                    for (b, o) in out.iter_mut().enumerate() {
                        *o = dot(row, &xs[b * n..(b + 1) * n]);
                    }
                }
                Kernel::Q4Block => {
                    let row = &wq[r * (n / 32) * 18..(r + 1) * (n / 32) * 18];
                    for (kb, blk) in row.chunks_exact(18).enumerate() {
                        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
                        for j in 0..16 {
                            unpacked[j] = ((blk[2 + j] & 0xf) as i32 - 8) as f32;
                            unpacked[16 + j] = ((blk[2 + j] >> 4) as i32 - 8) as f32;
                        }
                        for (b, o) in out.iter_mut().enumerate() {
                            let x = &xs[b * n + kb * 32..b * n + kb * 32 + 32];
                            *o += d * dot(&unpacked, x);
                        }
                    }
                }
                Kernel::Q8Deq | Kernel::Q4Deq => {
                    // One vector at a time: sixteen accumulators stay in registers
                    // for the whole row and are reduced once; each block is unpacked
                    // with its scale folded in, once per vector.
                    let bb = if k == Kernel::Q8Deq { 34 } else { 18 };
                    let row = &wq[r * (n / 32) * bb..(r + 1) * (n / 32) * bb];
                    for (b, o) in out.iter_mut().enumerate() {
                        let x = &xs[b * n..(b + 1) * n];
                        let mut acc = [0.0f32; 16];
                        for (blk, xb) in row.chunks_exact(bb).zip(x.chunks_exact(32)) {
                            let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
                            if k == Kernel::Q8Deq {
                                for (u, q) in unpacked.iter_mut().zip(&blk[2..34]) {
                                    *u = (*q as i8) as f32 * d;
                                }
                            } else {
                                for j in 0..16 {
                                    unpacked[j] = ((blk[2 + j] & 0xf) as i32 - 8) as f32 * d;
                                    unpacked[16 + j] = ((blk[2 + j] >> 4) as i32 - 8) as f32 * d;
                                }
                            }
                            for (wc, xc) in unpacked.chunks_exact(16).zip(xb.chunks_exact(16)) {
                                for ((a, wi), xi) in acc.iter_mut().zip(wc).zip(xc) {
                                    *a += wi * xi;
                                }
                            }
                        }
                        *o = acc.iter().sum();
                    }
                }
                Kernel::Q4Int => {
                    let row = &wq[r * (n / 32) * 18..(r + 1) * (n / 32) * 18];
                    for (kb, blk) in row.chunks_exact(18).enumerate() {
                        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
                        for j in 0..16 {
                            ints[j] = (blk[2 + j] & 0xf) as i8 - 8;
                            ints[16 + j] = (blk[2 + j] >> 4) as i8 - 8;
                        }
                        for (b, o) in out.iter_mut().enumerate() {
                            let (dx, qx) = &xq[b * (n / 32) + kb];
                            let s: i32 = ints
                                .iter()
                                .zip(qx)
                                .map(|(a, c)| *a as i32 * *c as i32)
                                .sum();
                            *o += d * dx * s as f32;
                        }
                    }
                }
            }
        }
        let mut all = rowsum.lock().unwrap();
        for (i, v) in local.iter().enumerate() {
            let (r, b) = (r0 + i / batch, i % batch);
            all[b * n + r] = *v;
        }
    });
    rowsum.into_inner().unwrap()
}

fn part3(threads: usize) {
    let n = 4096;
    println!(
        "\n3. A {n} x {n} matrix times B vectors, {threads} threads, best of 5; \
         error relative to the largest float64 output"
    );
    let w = fill(n * n, 11);
    let (q8, q4) = (quantize_q8_0(&w), quantize_q4_0(&w));
    let (d8, d4) = (decode(Q8_0, &q8), decode(Q4_0, &q4));
    println!(
        "{:<24} {:>3} {:>9} {:>9} {:>9} {:>10}",
        "kernel", "B", "ms", "GB/s", "GFLOP/s", "error"
    );
    for batch in [1usize, 4, 16] {
        let xs = fill(n * batch, 12 + batch as u64);
        for (name, k, weights, bytes) in [
            ("f32", Kernel::F32, &w, n * n * 4),
            ("Q8_0, dequantized", Kernel::Q8Deq, &d8, q8.len()),
            ("Q4_0, reduced per block", Kernel::Q4Block, &d4, q4.len()),
            ("Q4_0, dequantized", Kernel::Q4Deq, &d4, q4.len()),
            ("Q4_0 x Q8_0, integer", Kernel::Q4Int, &d4, q4.len()),
        ] {
            let wq: &[u8] = match k {
                Kernel::F32 => &[],
                Kernel::Q8Deq => &q8,
                _ => &q4,
            };
            let y = matmul(k, &w, wq, &xs, n, threads);
            // float64 reference with the weights the kernel represents
            let (mut err, mut peak) = (0.0f64, 1e-30f64);
            for b in 0..batch {
                let x = &xs[b * n..(b + 1) * n];
                for r in (0..n).step_by(61) {
                    let exact: f64 = weights[r * n..(r + 1) * n]
                        .iter()
                        .zip(x)
                        .map(|(a, v)| *a as f64 * *v as f64)
                        .sum();
                    err = err.max((y[b * n + r] as f64 - exact).abs());
                    peak = peak.max(exact.abs());
                }
            }
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = Instant::now();
                std::hint::black_box(matmul(k, &w, wq, &xs, n, threads));
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!(
                "{:<24} {:>3} {:>9.3} {:>9.1} {:>9.1} {:>10.1e}",
                name,
                batch,
                best * 1e3,
                bytes as f64 / best / 1e9,
                2.0 * (n * n * batch) as f64 / best / 1e9,
                err / peak
            );
        }
    }
}

fn main() -> std::io::Result<()> {
    let threads: usize = arg("--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let parts = arg("--parts").unwrap_or_else(|| "1,2,3".into());
    let dump = arg("--dump");
    let path = arg("--gguf");
    if parts.contains('1') || parts.contains('2') {
        match &path {
            Some(p) => {
                let mut g = Gguf::open(p)?;
                if parts.contains('1') {
                    part1(&mut g, dump.as_deref())?;
                }
                if parts.contains('2') {
                    part2(&mut g, threads)?;
                }
            }
            None => println!("parts 1 and 2 need --gguf <Llama 3.2 3B file>; skipping them"),
        }
    }
    if parts.contains('3') {
        part3(threads);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_quantizers_round_trip_within_their_step() {
        let x = fill(4096, 5);
        let (d8, d4) = (
            decode(Q8_0, &quantize_q8_0(&x)),
            decode(Q4_0, &quantize_q4_0(&x)),
        );
        for (a, (b, c)) in x.iter().zip(d8.iter().zip(&d4)) {
            assert!((a - b).abs() <= 0.5 / 127.0 * 0.5 + 1e-3);
            // one full step: a value of the other sign than the block's extreme can clamp
            assert!((a - c).abs() <= 0.5 / 8.0 + 2e-3);
        }
    }

    #[test]
    fn every_kernel_matches_float64() {
        let n = 256;
        let w = fill(n * n, 3);
        let (q8, q4) = (quantize_q8_0(&w), quantize_q4_0(&w));
        let (d8, d4) = (decode(Q8_0, &q8), decode(Q4_0, &q4));
        let xs = fill(n * 3, 4);
        for (k, wq, weights, tol) in [
            (Kernel::F32, &[][..], &w, 1e-5),
            (Kernel::Q8Deq, &q8[..], &d8, 1e-5),
            (Kernel::Q4Deq, &q4[..], &d4, 1e-5),
            (Kernel::Q4Block, &q4[..], &d4, 1e-5),
            (Kernel::Q4Int, &q4[..], &d4, 2e-2),
        ] {
            for t in [1, 3] {
                let y = matmul(k, &w, wq, &xs, n, t);
                for b in 0..3 {
                    for r in 0..n {
                        let exact: f64 = (0..n)
                            .map(|j| weights[r * n + j] as f64 * xs[b * n + j] as f64)
                            .sum();
                        assert!((y[b * n + r] as f64 - exact).abs() < tol * 8.0);
                    }
                }
            }
        }
    }
}
