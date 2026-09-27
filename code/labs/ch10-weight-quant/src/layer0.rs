//! Layer 0 of Llama 3.2 3B as the Chapter 10 lab uses it: the value projection
//! and exactly the inputs layer 0 sees for a text, RMSNorm(embedding) times the
//! layer's normalization weights, as distinct tokens with their counts.

use crate::gguf::Gguf;
use std::collections::BTreeMap;

/// Token IDs of the book's AUTHORING.md (see tokens.txt for provenance).
pub const TOKENS: &str = include_str!("../tokens.txt");

pub fn token_ids() -> Vec<u32> {
    TOKENS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(|v| v.parse().expect("token id")))
        .collect()
}

/// Distinct tokens with their counts, and their layer-0 inputs.
pub fn inputs(
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

/// Layer 0's value projection and its calibration and evaluation inputs.
pub struct Layer0 {
    pub w: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub cal: (Vec<f32>, Vec<f64>),
    pub eval: (Vec<f32>, Vec<f64>),
    pub cal_tokens: usize,
    pub eval_tokens: usize,
    pub eps: f32,
}

pub fn load(g: &mut Gguf) -> std::io::Result<Layer0> {
    let ids = token_ids();
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
    let cal = inputs(g, cal_ids, emb, &gamma, eps)?;
    let eval = inputs(g, eval_ids, emb, &gamma, eps)?;
    Ok(Layer0 {
        w,
        rows,
        cols,
        cal,
        eval,
        cal_tokens: cal_ids.len(),
        eval_tokens: eval_ids.len(),
        eps,
    })
}
