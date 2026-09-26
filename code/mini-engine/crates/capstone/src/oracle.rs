//! The oracle: recompute everything from scratch for every token.
//!
//! No cache survives between calls, keys and values sit in plain contiguous
//! arrays, and there is no paging, no chunking and no batching. It shares only
//! the kernels with the engine, and because those fix their summation order
//! the engine must reproduce its logits bit for bit. It is quadratic in the
//! sequence length per token, which is why no engine works this way and why it
//! is the right thing to test one against.

use crate::kernels::{argmax, dot, matmul_rows_simple, rms_norm, rope, silu, softmax};
use crate::model::Model;

/// Logits for the last position of `tokens`, computed from nothing.
pub fn oracle_logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let c = &model.cfg;
    let (d, qw, kvw, dh, n) = (c.d_model, c.q_width(), c.kv_width(), c.d_head, tokens.len());
    assert!(n > 0, "the oracle needs at least one token");
    let mut x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| {
            model.w.embed[t as usize * d..(t as usize + 1) * d]
                .iter()
                .copied()
        })
        .collect();
    let scale = 1.0 / (dh as f32).sqrt();
    for lw in &model.w.layers {
        let mut h = vec![0.0; n * d];
        for i in 0..n {
            rms_norm(
                &x[i * d..(i + 1) * d],
                &lw.attn_norm,
                c.norm_eps,
                &mut h[i * d..(i + 1) * d],
            );
        }
        let (mut q, mut k, mut v) = (vec![0.0; n * qw], vec![0.0; n * kvw], vec![0.0; n * kvw]);
        matmul_rows_simple(&lw.wq, qw, d, &h, n, &mut q);
        matmul_rows_simple(&lw.wk, kvw, d, &h, n, &mut k);
        matmul_rows_simple(&lw.wv, kvw, d, &h, n, &mut v);
        for i in 0..n {
            q[i * qw..(i + 1) * qw]
                .chunks_exact_mut(dh)
                .for_each(|head| rope(head, i, c.rope_base));
            k[i * kvw..(i + 1) * kvw]
                .chunks_exact_mut(dh)
                .for_each(|head| rope(head, i, c.rope_base));
        }
        let mut attn = vec![0.0; n * qw];
        for i in 0..n {
            for head in 0..c.n_heads {
                let g = head / c.group();
                let qh = &q[i * qw + head * dh..i * qw + (head + 1) * dh];
                let mut scores: Vec<f32> = (0..=i)
                    .map(|t| dot(qh, &k[t * kvw + g * dh..t * kvw + (g + 1) * dh]) * scale)
                    .collect();
                softmax(&mut scores);
                let out = &mut attn[i * qw + head * dh..i * qw + (head + 1) * dh];
                for (t, &p) in scores.iter().enumerate() {
                    for (oj, &vj) in out
                        .iter_mut()
                        .zip(&v[t * kvw + g * dh..t * kvw + (g + 1) * dh])
                    {
                        *oj += p * vj;
                    }
                }
            }
        }
        let mut o = vec![0.0; n * d];
        matmul_rows_simple(&lw.wo, d, qw, &attn, n, &mut o);
        x.iter_mut().zip(&o).for_each(|(xi, oi)| *xi += oi);
        for i in 0..n {
            rms_norm(
                &x[i * d..(i + 1) * d],
                &lw.ffn_norm,
                c.norm_eps,
                &mut h[i * d..(i + 1) * d],
            );
        }
        let (mut gate, mut up) = (vec![0.0; n * c.d_ff], vec![0.0; n * c.d_ff]);
        matmul_rows_simple(&lw.w_gate, c.d_ff, d, &h, n, &mut gate);
        matmul_rows_simple(&lw.w_up, c.d_ff, d, &h, n, &mut up);
        gate.iter_mut()
            .zip(&up)
            .for_each(|(g, u)| *g = silu(*g) * u);
        matmul_rows_simple(&lw.w_down, d, c.d_ff, &gate, n, &mut o);
        x.iter_mut().zip(&o).for_each(|(xi, oi)| *xi += oi);
    }
    let mut normed = vec![0.0; d];
    rms_norm(
        &x[(n - 1) * d..n * d],
        &model.w.final_norm,
        c.norm_eps,
        &mut normed,
    );
    let mut logits = vec![0.0; c.vocab];
    matmul_rows_simple(&model.w.output, c.vocab, d, &normed, 1, &mut logits);
    logits
}

/// Greedy generation by recomputation: the slow reference every engine
/// configuration must reproduce token for token.
pub fn oracle_generate(
    model: &Model,
    prompt: &[u32],
    max_new_tokens: usize,
    stop: Option<u32>,
) -> Vec<u32> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < max_new_tokens {
        let next = argmax(&oracle_logits(model, &seq)) as u32;
        out.push(next);
        if Some(next) == stop {
            break;
        }
        seq.push(next);
    }
    out
}
