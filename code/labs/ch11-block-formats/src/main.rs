//! Chapter 11 lab: round-trip real weights through FP8, MX and NVFP4.
//!
//! Part 1 prints each element format's grid, from its specification.
//! Part 2 quantizes layer 0's value projection of Llama 3.2 3B (Chapter 10's
//! lab loads it, with the same real inputs) into every format, and measures
//! bits per value, the weight error and the error in the layer's output.
//! Part 3 quantizes the layer's inputs instead, with the weights exact: the
//! activation side of the same comparison, with the inputs' outlier statistics.
//! Part 4 puts one large value in a block of small ones and shows what each
//! block format does to the small values.
//!
//!     cargo run --release -p ch11-block-formats -- --gguf <llama-3.2-3b.gguf>
//!         [--threads 4] [--outlier 30]

use ch10_weight_quant::gguf::Gguf;
use ch10_weight_quant::layer0;
use ch10_weight_quant::quant::{output_error, rtn, Grouping};
use ch11_block_formats::formats::{
    bf16_round, mx, nvfp4, per_tensor, Minifloat, Quantized, E2M1, E2M3, E3M2, E4M3, E5M2,
};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn part1() {
    println!("1. Element formats, from their specifications");
    for f in [E4M3, E5M2, E2M3, E3M2, E2M1] {
        let mags = f.magnitudes();
        let smallest = mags[1];
        let min_normal = f.decode(1 << f.m);
        println!(
            "  {}: {} bits, {} finite magnitudes, largest {}, smallest normal {:e}, smallest subnormal {:e}",
            f.name,
            f.bits(),
            mags.len(),
            f.max(),
            min_normal,
            smallest
        );
    }
    let grid: Vec<String> = E2M1.magnitudes().iter().map(|v| format!("{v}")).collect();
    println!("  E2M1 non-negative values: {}", grid.join(", "));
}

fn report(
    name: &str,
    q: &Quantized,
    w: &[f32],
    cols: usize,
    xe: &[f32],
    ce: &[f64],
    threads: usize,
) {
    let wn = w.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let we = w
        .iter()
        .zip(&q.values)
        .map(|(a, b)| ((*a - *b) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / wn;
    let (e, n) = output_error(w, &q.values, cols, xe, ce, threads);
    let sat = if q.saturated_blocks > 0 {
        format!(
            "  {} of {} blocks saturated",
            q.saturated_blocks,
            w.len() / 32
        )
    } else {
        String::new()
    };
    println!(
        "  {:<36} {:>7.3} {:>11.4} {:>11.4}{sat}",
        name,
        q.bits,
        we,
        (e / n).sqrt()
    );
}

fn part2(path: &str, threads: usize) -> std::io::Result<()> {
    let mut g = Gguf::open(path)?;
    let l0 = layer0::load(&mut g)?;
    let (w, cols) = (&l0.w, l0.cols);
    let (xe, ce) = (&l0.eval.0, &l0.eval.1);
    println!(
        "\n2. Layer 0's value projection ({} x {}, Q6_K in the file) in each format; output error \
         on {} evaluation tokens of AUTHORING.md",
        l0.rows, cols, l0.eval_tokens
    );
    println!(
        "  {:<36} {:>7} {:>11} {:>11}",
        "format", "bits", "weight err", "output err"
    );
    let bf16 = Quantized {
        values: w.iter().map(|v| bf16_round(*v)).collect(),
        bits: 16.0,
        saturated_blocks: 0,
    };
    report("BF16", &bf16, w, cols, xe, ce, threads);
    report(
        "FP8 E4M3, one scale per tensor",
        &per_tensor(w, &E4M3),
        w,
        cols,
        xe,
        ce,
        threads,
    );
    report(
        "FP8 E5M2, one scale per tensor",
        &per_tensor(w, &E5M2),
        w,
        cols,
        xe,
        ce,
        threads,
    );
    let mx_formats: [(&str, Minifloat); 5] = [
        ("MXFP8 (E4M3)", E4M3),
        ("MXFP8 (E5M2)", E5M2),
        ("MXFP6 (E2M3)", E2M3),
        ("MXFP6 (E3M2)", E3M2),
        ("MXFP4 (E2M1)", E2M1),
    ];
    for (name, f) in mx_formats {
        report(name, &mx(w, &f), w, cols, xe, ce, threads);
    }
    report(
        "NVFP4 (E2M1, E4M3 scales per 16)",
        &nvfp4(w),
        w,
        cols,
        xe,
        ce,
        threads,
    );
    for (name, bits) in [
        ("INT8, a scale per 32 (like Q8_0)", 8),
        ("INT4, a scale per 32 (like Q4_0)", 4),
    ] {
        let (values, bpw) = rtn(w, l0.rows, cols, bits, Grouping::Group(32));
        report(
            name,
            &Quantized {
                values,
                bits: bpw,
                saturated_blocks: 0,
            },
            w,
            cols,
            xe,
            ce,
            threads,
        );
    }
    Ok(())
}

/// FP8 with one FP32 scale per `block` consecutive values (DeepSeek-V3 scales
/// activations per token per 128 channels).
fn fp8_blocks(x: &[f32], f: &Minifloat, block: usize) -> Vec<f32> {
    x.chunks(block)
        .flat_map(|b| per_tensor(b, f).values)
        .collect()
}

fn part3(path: &str, threads: usize) -> std::io::Result<()> {
    let mut g = Gguf::open(path)?;
    let l0 = layer0::load(&mut g)?;
    let (w, cols) = (&l0.w, l0.cols);
    let (xe, ce) = (&l0.eval.0, &l0.eval.1);
    let tokens = ce.len();
    // Outlier statistics of the inputs: per token, largest |x| over RMS; per channel,
    // the largest mean |x|.
    let mut ratio = 0.0f64;
    let mut chan = vec![0.0f64; cols];
    for t in xe.chunks(cols) {
        let rms = (t.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cols as f64).sqrt();
        let amax = t.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
        ratio += amax / rms / tokens as f64;
        for (c, v) in chan.iter_mut().zip(t) {
            *c += (*v as f64).abs() / tokens as f64;
        }
    }
    let mean_chan = chan.iter().sum::<f64>() / cols as f64;
    let top = chan.iter().fold(0.0f64, |a, v| a.max(*v));
    println!(
        "\n3. The inputs instead ({tokens} distinct evaluation tokens x {cols} channels), weights exact"
    );
    println!(
        "  per token, the largest |x| is {ratio:.1} times the RMS on average; the loudest channel's \
         mean |x| is {:.1} times the average channel's",
        top / mean_chan
    );
    println!(
        "  {:<40} {:>7} {:>11}",
        "format for the inputs", "bits", "output err"
    );
    let quantized: Vec<(&str, f64, Vec<f32>)> = vec![
        (
            "FP8 E4M3, a scale per token",
            8.0 + 32.0 / cols as f64,
            fp8_blocks(xe, &E4M3, cols),
        ),
        (
            "FP8 E4M3, a scale per 128 channels",
            8.0 + 32.0 / 128.0,
            fp8_blocks(xe, &E4M3, 128),
        ),
        ("MXFP8 (E4M3)", 8.25, mx(xe, &E4M3).values),
        ("MXFP6 (E2M3)", 6.25, mx(xe, &E2M3).values),
        ("MXFP4 (E2M1)", 4.25, mx(xe, &E2M1).values),
        (
            "NVFP4, a tensor scale per token",
            4.5,
            xe.chunks(cols).flat_map(|t| nvfp4(t).values).collect(),
        ),
    ];
    for (name, bits, xq) in &quantized {
        // output error with W exact: compare W x and W x_hat, weighted by counts
        let (e, n) = activation_error(w, cols, xe, xq, ce, threads);
        println!("  {:<40} {:>7.3} {:>11.4}", name, bits, (e / n).sqrt());
    }
    let (q8, _) = rtn(xe, tokens, cols, 8, Grouping::Group(32));
    let (e, n) = activation_error(w, cols, xe, &q8, ce, threads);
    println!(
        "  {:<40} {:>7.3} {:>11.4}",
        "INT8, a scale per 32 (like Q8_0)",
        8.5,
        (e / n).sqrt()
    );
    let (q4, _) = rtn(xe, tokens, cols, 4, Grouping::Group(32));
    let (e, n) = activation_error(w, cols, xe, &q4, ce, threads);
    println!(
        "  {:<40} {:>7.3} {:>11.4}",
        "INT4, a scale per 32 (like Q4_0)",
        4.5,
        (e / n).sqrt()
    );
    Ok(())
}

/// Squared output error of W x_hat against W x, and the squared output norm.
fn activation_error(
    w: &[f32],
    cols: usize,
    x: &[f32],
    xq: &[f32],
    c: &[f64],
    threads: usize,
) -> (f64, f64) {
    let rows = w.len() / cols;
    let parts = std::sync::Mutex::new((0.0f64, 0.0f64));
    ch10_weight_quant::quant::par_rows(rows, threads, |r| {
        let wr = &w[r * cols..(r + 1) * cols];
        let (mut e, mut n) = (0.0f64, 0.0f64);
        for ((xt, qt), &ct) in x.chunks_exact(cols).zip(xq.chunks_exact(cols)).zip(c) {
            let (mut y, mut yq) = (0.0f64, 0.0f64);
            for ((a, b), v) in wr.iter().zip(qt).zip(xt) {
                y += (*a as f64) * (*v as f64);
                yq += (*a as f64) * (*b as f64);
            }
            e += ct * (y - yq) * (y - yq);
            n += ct * y * y;
        }
        let mut p = parts.lock().unwrap();
        p.0 += e;
        p.1 += n;
    });
    parts.into_inner().unwrap()
}

fn part4(outlier: f32) {
    println!("\n4. One value of {outlier} in a block of 32 values between -1 and 1");
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut block: Vec<f32> = (0..32)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    block[5] = outlier;
    let small = |q: &[f32]| {
        let (mut e, mut n, mut zero) = (0.0f64, 0.0f64, 0);
        for (i, (a, b)) in block.iter().zip(q).enumerate() {
            if i == 5 {
                continue;
            }
            e += ((a - b) as f64).powi(2);
            n += (*a as f64).powi(2);
            if *b == 0.0 && *a != 0.0 {
                zero += 1;
            }
        }
        ((e / n).sqrt(), zero)
    };
    for (name, q) in [
        ("MXFP8 (E4M3)", mx(&block, &E4M3)),
        ("MXFP4 (E2M1)", mx(&block, &E2M1)),
        ("NVFP4", nvfp4(&block)),
    ] {
        let (err, zeros) = small(&q.values);
        println!(
            "  {:<14} the outlier becomes {:>8.3}; the 31 small values: error {:>6.1}%, {zeros} set to zero",
            name,
            q.values[5],
            100.0 * err
        );
    }
}

fn main() -> std::io::Result<()> {
    let threads: usize = arg("--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let outlier: f32 = arg("--outlier")
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0);
    part1();
    match arg("--gguf") {
        Some(p) => {
            part2(&p, threads)?;
            part3(&p, threads)?;
        }
        None => println!("\nparts 2 and 3 need --gguf <Llama 3.2 3B file>; skipping them"),
    }
    part4(outlier);
    Ok(())
}
