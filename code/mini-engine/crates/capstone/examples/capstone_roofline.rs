//! Chapter 42's measurement: the capstone engine against the roofline of the
//! machine it runs on.
//!
//!     cargo run --release -p capstone --example capstone_roofline -- [out.json] [threads...]
//!
//! 1. The memory ceiling: summing a 256 MiB array with 32 independent running
//!    sums, so arithmetic cannot be the limit, on each thread count.
//! 2. The compute ceilings of the engine's two matrix kernels, one output at
//!    a time (`matmul_rows_simple`) and tiled four tokens at a time
//!    (`matmul_rows`), on operands that fit in cache, on each thread count;
//!    the tiled kernel as its activations outgrow the L1 cache; and the cost
//!    of spawning the threads.
//! 3. Decode steps of the 97-million-parameter configuration at batch sizes
//!    1 to 64, timed after every prompt is in the cache, for each thread count,
//!    with the FLOPs and bytes of each step counted from the model's shapes.
//!
//! The thread counts default to `1`.

use std::time::Instant;

use capstone::engine::{Event, Sampling};
use capstone::kernels::{matmul_rows, matmul_rows_simple, split_rows, MatmulKernel};
use capstone::{Engine, EngineConfig, GenerationRequest, Model, ModelConfig};

fn best_of<F: FnMut() -> f64>(runs: usize, mut f: F) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// Read every byte of a slice as fast as the core can: a sum of 32-bit
/// integers. Integer addition is associative, so the compiler is free to split
/// the sum across vector registers and several running totals; a float sum in
/// index order would forbid that (Chapter 38), and would measure the adder
/// instead of memory.
fn stream_sum(a: &[u32]) -> u32 {
    a.iter().fold(0u32, |s, &x| s.wrapping_add(x))
}

fn stream_sum_threads(a: &[u32], threads: usize) -> u32 {
    let per = a.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = a
            .chunks(per)
            .map(|part| scope.spawn(move || stream_sum(part)))
            .collect();
        handles
            .into_iter()
            .fold(0u32, |s, h| s.wrapping_add(h.join().unwrap()))
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args.first().filter(|a| a.ends_with(".json")).cloned();
    let mut thread_counts: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    if thread_counts.is_empty() {
        thread_counts.push(1);
    }
    let mut json = String::from("{\n");
    let mut sink = 0.0f32;

    // 1. memory: one 256 MiB read per pass
    let a: Vec<u32> = (0..64u32 << 20).map(|i| i % 7).collect();
    let mut checksum = 0u32;
    let mut bandwidth = Vec::new();
    for &t in &thread_counts {
        let secs = best_of(5, || {
            let start = Instant::now();
            checksum = checksum.wrapping_add(stream_sum_threads(&a, t));
            start.elapsed().as_secs_f64()
        });
        let gb_s = (a.len() * 4) as f64 / secs / 1e9;
        println!("memory ceiling, {t} thread(s): {gb_s:.1} GB/s (256 MiB read, best of 5)");
        bandwidth.push(gb_s);
    }
    drop(a);
    sink += checksum as f32;

    // 2. compute: a 512 x 1024 weight block (2 MiB, fits the shared L2)
    // against 16 tokens (64 KiB, fits one core's L1), the model's row width.
    // The engine uses the one-token kernel for steps of fewer than 4 tokens
    // and the tiled kernel otherwise, so both ceilings matter.
    let operands = |rows: usize, cols: usize, n: usize| {
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3)
            .collect();
        let x: Vec<f32> = (0..n * cols)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3)
            .collect();
        (w, x)
    };
    let mut gflops_of =
        |kernel: MatmulKernel, rows: usize, cols: usize, n: usize, threads: usize| {
            let (w, x) = operands(rows, cols, n);
            let mut y = vec![0.0f32; n * rows];
            let secs = best_of(20, || {
                let start = Instant::now();
                split_rows(kernel, &w, rows, cols, &x, n, &mut y, threads);
                start.elapsed().as_secs_f64()
            });
            sink += y[0];
            2.0 * (rows * cols * n) as f64 / secs / 1e9
        };
    let (mut simple, mut tiled) = (Vec::new(), Vec::new());
    for &t in &thread_counts {
        let g1 = gflops_of(matmul_rows_simple, 512, 1024, 16, t);
        let g4 = gflops_of(matmul_rows, 512, 1024, 16, t);
        println!(
            "compute ceiling, {t} thread(s): one token at a time {g1:.1} GFLOP/s, tiled 4 tokens {g4:.1} GFLOP/s (512x1024 x 16 tokens, best of 20)"
        );
        simple.push(g1);
        tiled.push(g4);
    }
    // 2b. the tiled kernel re-reads every token's activations for every weight
    // row; once n tokens of `cols` floats outgrow the L1 cache, those reads
    // come from L2 instead.
    let mut cache_json = Vec::new();
    for cols in [1024usize, 3072] {
        let row: Vec<String> = [4usize, 8, 16, 32, 64]
            .iter()
            .map(|&n| {
                let g = gflops_of(matmul_rows, 256, cols, n, 1);
                cache_json.push(format!("{{\"cols\": {cols}, \"tokens\": {n}, \"activation_kib\": {}, \"gflop_s\": {g:.3}}}", n * cols * 4 / 1024));
                format!("n={n} ({} KiB) {g:.1}", n * cols * 4 / 1024)
            })
            .collect();
        println!("tiled kernel, 1 thread, 256 x {cols}: {}", row.join(", "));
    }
    // 2c. what a call to split_rows pays before any arithmetic: spawning and
    // joining its threads.
    let mut spawn_us = Vec::new();
    for &t in &thread_counts {
        let secs = best_of(200, || {
            let start = Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..t {
                    scope.spawn(|| std::hint::black_box(0u32));
                }
            });
            start.elapsed().as_secs_f64()
        });
        println!(
            "spawn and join {t} thread(s): {:.1} us (best of 200)",
            secs * 1e6
        );
        spawn_us.push(secs * 1e6);
    }
    let list = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    json += &format!(
        "  \"threads\": {:?},\n  \"bandwidth_gb_s\": [{}],\n  \"kernel_simple_gflop_s\": [{}],\n  \"kernel_tiled_gflop_s\": [{}],\n  \"spawn_join_us\": [{}],\n  \"tiled_by_tokens\": [{}],\n",
        thread_counts,
        list(&bandwidth),
        list(&simple),
        list(&tiled),
        list(&spawn_us),
        cache_json.join(", ")
    );

    // 3. decode steps at rising batch sizes
    let cfg = ModelConfig::bench();
    println!(
        "model: {:.1} M parameters, {:.0} MB of f32 weights read per step, {} KB of cache per token",
        cfg.parameters() as f64 / 1e6,
        cfg.weight_bytes_per_step() as f64 / 1e6,
        cfg.kv_bytes_per_token() / 1024
    );
    let (prompt_len, warm_steps, timed_steps) = (16usize, 4usize, 12usize);
    json += "  \"runs\": [\n";
    let batches = [1usize, 2, 4, 8, 16, 32, 64];
    let mut first = true;
    for (ti, &threads) in thread_counts.iter().enumerate() {
        let bandwidth = bandwidth[ti];
        println!(
            "-- {threads} thread(s): ridge points {:.2} (one token) and {:.2} (tiled) FLOP/byte",
            simple[ti] / bandwidth,
            tiled[ti] / bandwidth
        );
        for &b in &batches {
            let model = Model::new(cfg, 7).with_threads(threads);
            let max_new = warm_steps + timed_steps + 2;
            let block_size = 16;
            let per_seq = (prompt_len + max_new).div_ceil(block_size);
            let ecfg = EngineConfig {
                n_blocks: b * per_seq,
                block_size,
                max_running: b,
                token_budget: b * prompt_len,
            };
            let mut engine = Engine::new(model, ecfg);
            for s in 0..b {
                let prompt: Vec<u32> = (0..prompt_len as u32)
                    .map(|i| (s as u32 * 13 + i * 29 + 5) % 256)
                    .collect();
                engine
                    .submit(GenerationRequest {
                        prompt,
                        max_new_tokens: max_new,
                        stop: None,
                        sampling: Sampling::Greedy,
                    })
                    .expect("fits by construction");
            }
            // Prefill every prompt in one step (the budget covers them all), then warm up.
            let mut produced = 0usize;
            for _ in 0..1 + warm_steps {
                produced += engine
                    .step()
                    .unwrap()
                    .iter()
                    .filter(|e| matches!(e, Event::Token { .. }))
                    .count();
            }
            assert_eq!(engine.running(), b);
            let context_start = prompt_len + warm_steps; // tokens already cached when timing starts
            let t = Instant::now();
            for _ in 0..timed_steps {
                produced += engine
                    .step()
                    .unwrap()
                    .iter()
                    .filter(|e| matches!(e, Event::Token { .. }))
                    .count();
            }
            let secs = t.elapsed().as_secs_f64();
            let step_s = secs / timed_steps as f64;
            // FLOPs and bytes of the timed steps, counted from the shapes
            let mut flops = 0.0;
            let mut bytes = 0.0;
            for k in 0..timed_steps {
                let context = context_start + k;
                flops += b as f64 * cfg.flops_per_token(context);
                bytes += cfg.weight_bytes_per_step() as f64
                    + (b * (context + 1) * cfg.kv_bytes_per_token()) as f64;
            }
            let intensity = flops / bytes;
            let gflops = flops / secs / 1e9;
            let tok_s = b as f64 / step_s;
            let peak = if b < 4 { simple[ti] } else { tiled[ti] };
            let bound = (bandwidth * intensity).min(peak);
            println!(
                "B={b:>2}: {:7.1} ms/step  {tok_s:7.1} tok/s  intensity {intensity:6.2} FLOP/byte  \
                 {gflops:6.1} GFLOP/s  roofline {bound:6.1}  ({:.0}%)",
                step_s * 1e3,
                100.0 * gflops / bound
            );
            json += &format!(
                "{}    {{\"threads\": {threads}, \"batch\": {b}, \"step_ms\": {:.3}, \
                 \"tokens_per_s\": {tok_s:.2}, \"intensity\": {intensity:.4}, \"gflop_s\": {gflops:.3}, \
                 \"roofline_gflop_s\": {bound:.3}, \"tokens_generated\": {produced}}}",
                if first { "" } else { ",\n" },
                step_s * 1e3
            );
            first = false;
        }
    }
    json += &format!("\n  ],\n  \"sink\": {sink}\n}}\n");
    if let Some(path) = out_path {
        std::fs::write(&path, json).expect("write results");
        println!("wrote {path}");
    }
}
