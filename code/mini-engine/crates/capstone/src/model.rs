//! One forward pass over a scheduler step: any mix of prefill chunks and
//! single decode tokens from different sequences, with their keys and values
//! read from and written to the paged cache.

use crate::config::ModelConfig;
use crate::kernels::{dot, matmul_rows_threads, rms_norm, rope, silu, softmax};
use crate::kv::{BlockId, KvError, KvPool};
use crate::weights::Weights;

/// One sequence's share of a step: `tokens` occupy positions
/// `start..start + tokens.len()`, and everything before `start` is already in
/// the cache under `table`.
#[derive(Debug, Clone)]
pub struct SeqChunk<'a> {
    pub tokens: &'a [u32],
    pub start: usize,
    pub table: &'a [BlockId],
    /// Return logits for the chunk's last token (a decode token, or the end of
    /// a prompt); intermediate prefill chunks need none.
    pub want_logits: bool,
}

pub struct Model {
    pub cfg: ModelConfig,
    pub w: Weights,
    /// Threads for the matrix products. Results do not depend on it, bit for bit.
    pub threads: usize,
}

impl Model {
    pub fn new(cfg: ModelConfig, seed: u64) -> Self {
        cfg.validate().expect("invalid model configuration");
        let w = Weights::random(&cfg, seed);
        Self { cfg, w, threads: 1 }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Runs one step. Returns, for each chunk, its last token's logits if it
    /// asked for them.
    pub fn forward(
        &self,
        chunks: &[SeqChunk<'_>],
        pool: &mut KvPool,
    ) -> Result<Vec<Option<Vec<f32>>>, KvError> {
        let c = &self.cfg;
        let (d, qw, kvw, dh) = (c.d_model, c.q_width(), c.kv_width(), c.d_head);
        // Flatten the step: token i belongs to chunk owner[i] at position pos[i].
        let mut owner = Vec::new();
        let mut pos = Vec::new();
        let mut x = Vec::new();
        for (s, ch) in chunks.iter().enumerate() {
            for (j, &tok) in ch.tokens.iter().enumerate() {
                owner.push(s);
                pos.push(ch.start + j);
                x.extend_from_slice(&self.w.embed[tok as usize * d..(tok as usize + 1) * d]);
            }
        }
        let n = owner.len();
        let mut h = vec![0.0; n * d];
        let mut q = vec![0.0; n * qw];
        let mut k = vec![0.0; n * kvw];
        let mut v = vec![0.0; n * kvw];
        let mut attn = vec![0.0; n * qw];
        let mut o = vec![0.0; n * d];
        let mut gate = vec![0.0; n * c.d_ff];
        let mut up = vec![0.0; n * c.d_ff];
        let scale = 1.0 / (dh as f32).sqrt();

        for (layer, lw) in self.w.layers.iter().enumerate() {
            // 1. attention input: normalize each token's residual stream
            for i in 0..n {
                rms_norm(
                    &x[i * d..(i + 1) * d],
                    &lw.attn_norm,
                    c.norm_eps,
                    &mut h[i * d..(i + 1) * d],
                );
            }
            // 2. project every token of the step at once; each weight row is read once
            matmul_rows_threads(&lw.wq, qw, d, &h, n, &mut q, self.threads);
            matmul_rows_threads(&lw.wk, kvw, d, &h, n, &mut k, self.threads);
            matmul_rows_threads(&lw.wv, kvw, d, &h, n, &mut v, self.threads);
            // 3. rotate queries and keys by position, then append keys and values to the cache
            for i in 0..n {
                for head in q[i * qw..(i + 1) * qw].chunks_exact_mut(dh) {
                    rope(head, pos[i], c.rope_base);
                }
                for head in k[i * kvw..(i + 1) * kvw].chunks_exact_mut(dh) {
                    rope(head, pos[i], c.rope_base);
                }
                let ch = &chunks[owner[i]];
                pool.write(
                    ch.table,
                    layer,
                    pos[i],
                    &k[i * kvw..(i + 1) * kvw],
                    &v[i * kvw..(i + 1) * kvw],
                )?;
            }
            // 4. causal attention: each token reads its own sequence's cache up to its position
            for i in 0..n {
                let ch = &chunks[owner[i]];
                let mut scores = vec![0.0f32; pos[i] + 1];
                for head in 0..c.n_heads {
                    let kv_head = head / c.group();
                    let qh = &q[i * qw + head * dh..i * qw + (head + 1) * dh];
                    for (t, score) in scores.iter_mut().enumerate() {
                        let (kt, _) = pool.read(ch.table, layer, t)?;
                        *score = dot(qh, &kt[kv_head * dh..(kv_head + 1) * dh]) * scale;
                    }
                    softmax(&mut scores);
                    let out = &mut attn[i * qw + head * dh..i * qw + (head + 1) * dh];
                    out.fill(0.0);
                    for (t, &p) in scores.iter().enumerate() {
                        let (_, vt) = pool.read(ch.table, layer, t)?;
                        for (oj, &vj) in out.iter_mut().zip(&vt[kv_head * dh..(kv_head + 1) * dh]) {
                            *oj += p * vj;
                        }
                    }
                }
            }
            // 5. output projection and residual
            matmul_rows_threads(&lw.wo, d, qw, &attn, n, &mut o, self.threads);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            // 6. SwiGLU feed-forward and residual
            for i in 0..n {
                rms_norm(
                    &x[i * d..(i + 1) * d],
                    &lw.ffn_norm,
                    c.norm_eps,
                    &mut h[i * d..(i + 1) * d],
                );
            }
            matmul_rows_threads(&lw.w_gate, c.d_ff, d, &h, n, &mut gate, self.threads);
            matmul_rows_threads(&lw.w_up, c.d_ff, d, &h, n, &mut up, self.threads);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = silu(*g) * u;
            }
            matmul_rows_threads(&lw.w_down, d, c.d_ff, &gate, n, &mut o, self.threads);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
        }

        // 7. logits for the last token of every chunk that wants them
        let mut result = Vec::with_capacity(chunks.len());
        let mut end = 0;
        for ch in chunks {
            end += ch.tokens.len();
            if ch.want_logits && !ch.tokens.is_empty() {
                let last = end - 1;
                let mut normed = vec![0.0; d];
                rms_norm(
                    &x[last * d..(last + 1) * d],
                    &self.w.final_norm,
                    c.norm_eps,
                    &mut normed,
                );
                let mut logits = vec![0.0; c.vocab];
                matmul_rows_threads(
                    &self.w.output,
                    c.vocab,
                    d,
                    &normed,
                    1,
                    &mut logits,
                    self.threads,
                );
                result.push(Some(logits));
            } else {
                result.push(None);
            }
        }
        Ok(result)
    }
}
