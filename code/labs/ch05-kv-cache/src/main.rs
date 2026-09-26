//! Chapter 5 lab: a cache you can prove correct.
//!
//! A small Llama-style model (the capstone engine's code, seeded random
//! weights) generates tokens greedily two ways:
//!
//! * **with a KV cache**: the prompt is prefilled once, then every step
//!   projects only the new token and reads the earlier tokens' keys and
//!   values from the cache;
//! * **by recomputation**: every step runs the whole sequence through the
//!   model from scratch (the capstone's oracle), as a model runner without a
//!   cache would.
//!
//! The oracle is the recomputation: at every step the cached path must pick
//! the same token and produce the same logits, bit for bit. The program times
//! both and counts the token positions each projects. With `--evict P` it
//! zeroes the cached keys and values of prompt position P after prefill and
//! reports how far the outputs drift.
//!
//!     cargo run --release -p ch05-kv-cache -- [--new 128] [--evict P] [--csv out.csv]

use std::time::Instant;

use capstone::kernels::argmax;
use capstone::kv::{BlockId, KvPool};
use capstone::model::SeqChunk;
use capstone::oracle::oracle_logits;
use capstone::{Model, ModelConfig};

const BLOCK: usize = 16;

/// Four layers of width 256: small enough that recomputing a few hundred
/// tokens takes seconds, large enough that the costs are the model's.
pub fn lab_config() -> ModelConfig {
    ModelConfig {
        vocab: 256,
        d_model: 256,
        n_layers: 4,
        n_heads: 8,
        n_kv_heads: 2,
        d_head: 32,
        d_ff: 768,
        rope_base: 10_000.0,
        norm_eps: 1e-5,
    }
}

pub fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 61 + 7) % 256) as u32).collect()
}

fn logits_of(
    model: &Model,
    pool: &mut KvPool,
    table: &[BlockId],
    tokens: &[u32],
    start: usize,
) -> Vec<f32> {
    let chunk = SeqChunk {
        tokens,
        start,
        table,
        want_logits: true,
    };
    model.forward(&[chunk], pool).expect("cache has room")[0]
        .take()
        .expect("logits were requested")
}

/// One generation's record: the tokens, each step's logits and each step's time.
pub struct Run {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub step_secs: Vec<f64>,
}

/// Greedy generation with a KV cache. `evict` zeroes one prompt position's
/// cached keys and values, in every layer, after the prefill.
pub fn with_cache(model: &Model, prompt: &[u32], new: usize, evict: Option<usize>) -> Run {
    let cfg = model.cfg;
    let blocks = (prompt.len() + new).div_ceil(BLOCK);
    let mut pool = KvPool::new(blocks, BLOCK, cfg.n_layers, cfg.kv_width());
    let table = pool.alloc(blocks).unwrap();
    let mut run = Run {
        tokens: Vec::new(),
        logits: Vec::new(),
        step_secs: Vec::new(),
    };
    let t = Instant::now();
    let mut logits = logits_of(model, &mut pool, &table, prompt, 0);
    run.step_secs.push(t.elapsed().as_secs_f64());
    if let Some(p) = evict {
        let zeros = vec![0.0f32; cfg.kv_width()];
        for layer in 0..cfg.n_layers {
            pool.write(&table, layer, p, &zeros, &zeros).unwrap();
        }
    }
    for pos in prompt.len()..prompt.len() + new {
        let next = argmax(&logits) as u32;
        run.tokens.push(next);
        run.logits.push(logits);
        if run.tokens.len() == new {
            break;
        }
        let t = Instant::now();
        logits = logits_of(model, &mut pool, &table, &[next], pos);
        run.step_secs.push(t.elapsed().as_secs_f64());
    }
    run
}

/// Greedy generation by recomputing the whole sequence at every step.
pub fn by_recomputation(model: &Model, prompt: &[u32], new: usize) -> Run {
    let mut seq = prompt.to_vec();
    let mut run = Run {
        tokens: Vec::new(),
        logits: Vec::new(),
        step_secs: Vec::new(),
    };
    for _ in 0..new {
        let t = Instant::now();
        let logits = oracle_logits(model, &seq);
        run.step_secs.push(t.elapsed().as_secs_f64());
        let next = argmax(&logits) as u32;
        run.tokens.push(next);
        run.logits.push(logits);
        seq.push(next);
    }
    run
}

/// The first step at which two runs differ, and the largest logit gap.
pub fn compare(a: &Run, b: &Run) -> (Option<usize>, f32) {
    let mut first = None;
    let mut gap = 0.0f32;
    for (i, (la, lb)) in a.logits.iter().zip(&b.logits).enumerate() {
        let same = la.iter().zip(lb).all(|(x, y)| x.to_bits() == y.to_bits());
        if !same && first.is_none() {
            first = Some(i);
        }
        gap = la
            .iter()
            .zip(lb)
            .fold(gap, |g, (x, y)| g.max((x - y).abs()));
    }
    (first, gap)
}

fn csv_path() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1).cloned())
}

fn arg(name: &str) -> Option<usize> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() {
    let new = arg("--new").unwrap_or(128);
    let evict = arg("--evict");
    let cfg = lab_config();
    let model = Model::new(cfg, 5);
    let p = prompt(32);
    println!(
        "model: {} layers, width {}, {} parameters; prompt {} tokens, {} generated; KV cache {} bytes per token",
        cfg.n_layers,
        cfg.d_model,
        cfg.parameters(),
        p.len(),
        new,
        cfg.kv_bytes_per_token()
    );
    let cached = with_cache(&model, &p, new, None);
    let slow = by_recomputation(&model, &p, new);
    match compare(&cached, &slow) {
        (None, _) => println!("oracle: all {new} steps agree bit for bit (tokens and logits)"),
        (Some(i), g) => {
            eprintln!("oracle failed at step {i}: largest logit gap {g}");
            std::process::exit(1);
        }
    }
    let (proj_cached, proj_slow) = (p.len() + new - 1, (p.len()..p.len() + new).sum::<usize>());
    println!(
        "token positions projected: {proj_cached} with the cache, {proj_slow} without ({:.0}x)",
        proj_slow as f64 / proj_cached as f64
    );
    println!(
        "{:>9} {:>16} {:>18}",
        "position", "cached ms/token", "recompute ms/token"
    );
    println!(
        "{:>9} {:>16.3} {:>18.3}   (both process the {}-token prompt)",
        "prompt",
        cached.step_secs[0] * 1e3,
        slow.step_secs[0] * 1e3,
        p.len()
    );
    for i in [1usize, new / 4, new / 2, 3 * new / 4, new - 1] {
        println!(
            "{:>9} {:>16.3} {:>18.3}",
            p.len() + i,
            cached.step_secs[i] * 1e3,
            slow.step_secs[i] * 1e3
        );
    }
    let (tc, ts) = (
        cached.step_secs.iter().sum::<f64>(),
        slow.step_secs.iter().sum::<f64>(),
    );
    println!(
        "total: {:.3} s with the cache, {:.3} s without ({:.0}x)",
        tc,
        ts,
        ts / tc
    );

    if let Some(path) = csv_path() {
        let mut rows = vec!["position,cached_ms,recompute_ms".to_string()];
        for (i, (c, r)) in cached.step_secs.iter().zip(&slow.step_secs).enumerate() {
            rows.push(format!("{},{:.4},{:.4}", p.len() + i, c * 1e3, r * 1e3));
        }
        std::fs::write(&path, rows.join("\n") + "\n").expect("write the CSV");
        println!("wrote {path}");
    }

    if let Some(pos) = evict {
        let broken = with_cache(&model, &p, new, Some(pos));
        let (first, gap) = compare(&broken, &slow);
        let first_token = broken
            .tokens
            .iter()
            .zip(&slow.tokens)
            .position(|(a, b)| a != b);
        println!(
            "evicting prompt position {pos}: logits differ from step {:?}, largest gap {gap:.4}; first different token at step {:?}",
            first, first_token
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_reproduces_recomputation_bit_for_bit() {
        let model = Model::new(ModelConfig::tiny(), 3);
        let p = prompt(9);
        let (a, b) = (
            with_cache(&model, &p, 12, None),
            by_recomputation(&model, &p, 12),
        );
        assert_eq!(compare(&a, &b).0, None);
        assert_eq!(a.tokens, b.tokens);
    }

    #[test]
    fn a_corrupted_entry_changes_the_logits() {
        let model = Model::new(ModelConfig::tiny(), 3);
        let p = prompt(9);
        let (a, b) = (
            with_cache(&model, &p, 4, Some(2)),
            by_recomputation(&model, &p, 4),
        );
        // step 0's logits come from the prefill, before the entry is zeroed
        assert_eq!(compare(&a, &b).0, Some(1));
    }
}
