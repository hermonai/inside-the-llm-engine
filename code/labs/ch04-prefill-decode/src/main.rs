//! Chapter 4 lab: one model, read and written.
//!
//! The capstone engine's model (Chapter 42's measured configuration: 97
//! million parameters, 388 MB of f32 weights) processes a prompt of S tokens
//! two ways:
//!
//! * as **prefill**: one forward pass over all S tokens, which multiplies every
//!   weight matrix by a matrix of S rows;
//! * as **decode**: S forward passes of one token each, which multiply every
//!   weight matrix by one row, S times.
//!
//! The oracle: both ways must leave the same keys and values in the cache and
//! give the same logits for the last token, bit for bit. That is possible
//! because every capstone kernel computes each output element in the same
//! order whatever else is in the batch (batch invariance, Chapter 38). Then
//! both are timed for S = 1, 2, 4, ... up to `--max`.
//!
//!     cargo run --release -p ch04-prefill-decode -- [--threads 4] [--max 512]

use std::time::Instant;

use capstone::kv::{BlockId, KvPool};
use capstone::model::SeqChunk;
use capstone::{Model, ModelConfig};

const BLOCK: usize = 16;

/// A deterministic prompt of `n` token ids.
pub fn prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % vocab) as u32).collect()
}

/// One forward pass over `tokens` at positions `start..`, returning the last
/// token's logits.
fn pass(
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

/// Prefill: all tokens in one pass.
pub fn prefill(model: &Model, pool: &mut KvPool, table: &[BlockId], tokens: &[u32]) -> Vec<f32> {
    pass(model, pool, table, tokens, 0)
}

/// Decode: one pass per token, each reading the cache the earlier ones wrote.
pub fn one_by_one(model: &Model, pool: &mut KvPool, table: &[BlockId], tokens: &[u32]) -> Vec<f32> {
    let mut logits = Vec::new();
    for (t, tok) in tokens.iter().enumerate() {
        logits = pass(model, pool, table, std::slice::from_ref(tok), t);
    }
    logits
}

/// The oracle: same logits and same cache contents, bit for bit.
pub fn check(model: &Model, tokens: &[u32]) -> Result<(), String> {
    let cfg = model.cfg;
    let blocks = tokens.len().div_ceil(BLOCK);
    let mut a = KvPool::new(blocks, BLOCK, cfg.n_layers, cfg.kv_width());
    let mut b = KvPool::new(blocks, BLOCK, cfg.n_layers, cfg.kv_width());
    let (ta, tb) = (a.alloc(blocks).unwrap(), b.alloc(blocks).unwrap());
    let la = prefill(model, &mut a, &ta, tokens);
    let lb = one_by_one(model, &mut b, &tb, tokens);
    if let Some(i) = la
        .iter()
        .zip(&lb)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err(format!(
            "logit {i}: prefill {} vs one by one {}",
            la[i], lb[i]
        ));
    }
    for layer in 0..cfg.n_layers {
        for pos in 0..tokens.len() {
            let (ka, va) = a.read(&ta, layer, pos).unwrap();
            let (kb, vb) = b.read(&tb, layer, pos).unwrap();
            let same =
                |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
            if !same(ka, kb) || !same(va, vb) {
                return Err(format!("cache differs at layer {layer}, position {pos}"));
            }
        }
    }
    Ok(())
}

fn best_of(runs: usize, mut f: impl FnMut()) -> f64 {
    (0..runs)
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
    let threads = arg("--threads", 4);
    let max = arg("--max", 512);
    let cfg = ModelConfig::bench();
    let model = Model::new(cfg, 42).with_threads(threads);
    println!(
        "model: {} parameters, {:.0} MB of weights read per decode step, {} threads",
        cfg.parameters(),
        cfg.weight_bytes_per_step() as f64 / 1e6,
        threads
    );
    let tokens = prompt(max, cfg.vocab);
    if let Err(e) = check(&model, &tokens[..max.min(64)]) {
        eprintln!("oracle failed: {e}");
        std::process::exit(1);
    }
    println!(
        "oracle: prefill and one-by-one decode agree bit for bit (logits and cache, 64 tokens)"
    );

    let blocks = max.div_ceil(BLOCK);
    let mut pool = KvPool::new(blocks, BLOCK, cfg.n_layers, cfg.kv_width());
    let table = pool.alloc(blocks).unwrap();
    prefill(&model, &mut pool, &table, &tokens[..max.min(64)]); // warm the weights
    println!(
        "{:>5} {:>12} {:>13} {:>15} {:>14} {:>7}",
        "S", "prefill ms", "prefill tok/s", "one-by-one ms", "decode tok/s", "ratio"
    );
    let mut s = 1;
    while s <= max {
        let toks = &tokens[..s];
        let t_pre = best_of(3, || {
            prefill(&model, &mut pool, &table, toks);
        });
        let t_dec = best_of(if s <= 64 { 3 } else { 1 }, || {
            one_by_one(&model, &mut pool, &table, toks);
        });
        println!(
            "{s:>5} {:>12.1} {:>13.1} {:>15.1} {:>14.1} {:>6.1}x",
            t_pre * 1e3,
            s as f64 / t_pre,
            t_dec * 1e3,
            s as f64 / t_dec,
            t_dec / t_pre
        );
        s *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_and_decode_agree_bit_for_bit_on_a_tiny_model() {
        let model = Model::new(ModelConfig::tiny(), 7);
        check(&model, &prompt(37, 256)).unwrap();
        check(&model.with_threads(3), &prompt(19, 256)).unwrap();
    }
}
