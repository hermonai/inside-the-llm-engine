//! Chapter 42's measurement: the capstone engine on the roofline of one core.
//!
//!     cargo run --release -p capstone --example capstone_roofline [out.json]
//!
//! 1. The memory ceiling: one core streaming a 256 MiB array.
//! 2. The compute ceiling: the engine's own matrix kernel on operands that fit
//!    in cache, so it is limited by arithmetic rather than memory.
//! 3. Decode steps of the 97-million-parameter configuration at batch sizes
//!    1 to 64, timed after every prompt is in the cache, with the FLOPs and
//!    bytes of each step counted from the model's shapes.
//!
//! Everything runs on one thread; the engine has no threads of its own.

use std::time::Instant;

use capstone::engine::{Event, Sampling};
use capstone::kernels::{dot, matmul_rows};
use capstone::{Engine, EngineConfig, GenerationRequest, Model, ModelConfig};

fn best_of<F: FnMut() -> f64>(runs: usize, mut f: F) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

fn main() {
    let out_path = std::env::args().nth(1);
    let mut json = String::from("{\n");

    // 1. memory: dot(a, a) reads one 256 MiB stream once per pass
    let a: Vec<f32> = (0..64usize << 20).map(|i| (i % 7) as f32 * 0.5).collect();
    let mut sink = 0.0f32;
    let t_mem = best_of(5, || {
        let t = Instant::now();
        sink += dot(&a, &a);
        t.elapsed().as_secs_f64()
    });
    let bandwidth = (a.len() * 4) as f64 / t_mem / 1e9;
    drop(a);
    println!("memory ceiling: {bandwidth:.1} GB/s (one core, 256 MiB read, best of 5)");

    // 2. compute: a 512x512 weight block against 64 tokens, resident in cache
    let (rows, cols, n) = (512, 512, 64);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
    let x: Vec<f32> = (0..n * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
    let mut y = vec![0.0f32; n * rows];
    let t_mm = best_of(20, || {
        let t = Instant::now();
        matmul_rows(&w, rows, cols, &x, n, &mut y);
        t.elapsed().as_secs_f64()
    });
    sink += y[0];
    let peak = 2.0 * (rows * cols * n) as f64 / t_mm / 1e9;
    println!("compute ceiling: {peak:.1} GFLOP/s (matmul_rows 512x512 x 64 tokens, best of 20)");
    json += &format!("  \"bandwidth_gb_s\": {bandwidth:.3},\n  \"kernel_peak_gflop_s\": {peak:.3},\n");

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
    for (bi, &b) in batches.iter().enumerate() {
        let model = Model::new(cfg, 7);
        let max_new = warm_steps + timed_steps + 2;
        let block_size = 16;
        let per_seq = (prompt_len + max_new).div_ceil(block_size);
        let ecfg = EngineConfig { n_blocks: b * per_seq, block_size, max_running: b, token_budget: b * prompt_len };
        let mut engine = Engine::new(model, ecfg);
        for s in 0..b {
            let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| (s as u32 * 13 + i * 29 + 5) % 256).collect();
            engine
                .submit(GenerationRequest { prompt, max_new_tokens: max_new, stop: None, sampling: Sampling::Greedy })
                .expect("fits by construction");
        }
        // Prefill every prompt in one step (the budget covers them all), then warm up.
        let mut produced = 0usize;
        for _ in 0..1 + warm_steps {
            produced += engine.step().unwrap().iter().filter(|e| matches!(e, Event::Token { .. })).count();
        }
        assert_eq!(engine.running(), b);
        let context_start = prompt_len + warm_steps; // tokens already cached when timing starts
        let t = Instant::now();
        for _ in 0..timed_steps {
            produced += engine.step().unwrap().iter().filter(|e| matches!(e, Event::Token { .. })).count();
        }
        let secs = t.elapsed().as_secs_f64();
        let step_s = secs / timed_steps as f64;
        // FLOPs and bytes of the timed steps, counted from the shapes
        let mut flops = 0.0;
        let mut bytes = 0.0;
        for k in 0..timed_steps {
            let context = context_start + k;
            flops += b as f64 * cfg.flops_per_token(context);
            bytes += cfg.weight_bytes_per_step() as f64 + (b * (context + 1) * cfg.kv_bytes_per_token()) as f64;
        }
        let intensity = flops / bytes;
        let gflops = flops / secs / 1e9;
        let tok_s = b as f64 / step_s;
        let bound = (bandwidth * intensity).min(peak);
        println!(
            "B={b:>2}: {:7.1} ms/step  {tok_s:7.1} tok/s  intensity {intensity:6.2} FLOP/byte  {gflops:6.1} GFLOP/s  roofline {bound:6.1}  ({:.0}%)",
            step_s * 1e3,
            100.0 * gflops / bound
        );
        json += &format!(
            "    {{\"batch\": {b}, \"step_ms\": {:.3}, \"tokens_per_s\": {tok_s:.2}, \"intensity\": {intensity:.4}, \"gflop_s\": {gflops:.3}, \"roofline_gflop_s\": {bound:.3}, \"tokens_generated\": {produced}}}{}\n",
            step_s * 1e3,
            if bi + 1 < batches.len() { "," } else { "" }
        );
    }
    json += &format!("  ],\n  \"sink\": {sink}\n}}\n");
    if let Some(path) = out_path {
        std::fs::write(&path, json).expect("write results");
        println!("wrote {path}");
    }
}
