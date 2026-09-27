//! Chapter 12 lab: quantize a real KV cache and watch attention drift.
//!
//! Layer 0 of Llama 3.2 3B, computed from its GGUF file for the 3,217 tokens of
//! the book's AUTHORING.md: queries, keys and values (24 query heads sharing 8
//! key/value heads of width 128), with RoPE as llama.cpp applies it for this
//! model (consecutive pairs of channels, base 500,000, the file's frequency
//! factors). The last `--queries` tokens attend causally to every earlier
//! token. Keys and values are stored in each cache format, and each format's
//! attention is compared with full precision: the error in the attention
//! output, how often the most-attended token changes, and the KL divergence
//! between the attention distributions.
//!
//!     cargo run --release -p ch12-kv-quant -- --gguf <llama-3.2-3b.gguf> [--queries 256]

use ch10_weight_quant::gguf::{
    decode, f16, quantize_q4_0, quantize_q8_0, to_f16, Gguf, Q4_0, Q8_0,
};
use ch10_weight_quant::layer0;
use ch10_weight_quant::quant::par_rows;
use ch11_block_formats::formats::E4M3;
use std::collections::HashMap;
use std::sync::Mutex;

pub const HD: usize = 128;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Rotates consecutive channel pairs of every head of `v` for position `pos`,
/// as ggml's normal RoPE does: theta_j = pos * base^(-2j/d) / factor_j.
pub fn rope(v: &mut [f32], pos: usize, base: f64, factors: &[f32]) {
    for head in v.chunks_mut(HD) {
        for j in 0..HD / 2 {
            let theta = pos as f64 * base.powf(-2.0 * j as f64 / HD as f64) / factors[j] as f64;
            let (s, c) = theta.sin_cos();
            let (x0, x1) = (head[2 * j] as f64, head[2 * j + 1] as f64);
            head[2 * j] = (x0 * c - x1 * s) as f32;
            head[2 * j + 1] = (x0 * s + x1 * c) as f32;
        }
    }
}

/// The normalized Walsh-Hadamard transform of each block of `n` values, in
/// place; it is its own inverse. llama.cpp rotates quantized keys in blocks of
/// the head width (128 here) and values in blocks of 64.
pub fn hadamard(v: &mut [f32], n: usize) {
    for block in v.chunks_mut(n) {
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (block[j], block[j + h]);
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
            }
            h *= 2;
        }
        let s = 1.0 / (n as f32).sqrt();
        block.iter_mut().for_each(|x| *x *= s);
    }
}

/// Asymmetric min-max quantization of a group to `bits` bits.
fn minmax(g: &mut [f32], bits: u32) {
    let (lo, hi) = g
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    let levels = ((1u32 << bits) - 1) as f32;
    let scale = (hi - lo) / levels;
    for x in g.iter_mut() {
        *x = if scale > 0.0 {
            lo + ((*x - lo) / scale).round() * scale
        } else {
            lo
        };
    }
}

/// Per-token asymmetric quantization in groups of 32 channels.
fn per_token(c: &[f32], bits: u32) -> Vec<f32> {
    let mut out = c.to_vec();
    for g in out.chunks_mut(32) {
        minmax(g, bits);
    }
    out
}

/// Per-channel asymmetric quantization over groups of `group` tokens (KIVI);
/// the tokens of the last, incomplete group stay in 16 bits, as KIVI keeps a
/// full-precision residual.
fn per_channel(c: &[f32], width: usize, bits: u32, group: usize) -> Vec<f32> {
    let n = c.len() / width;
    let mut out = c.to_vec();
    let full = n / group * group;
    let mut col = vec![0.0f32; group];
    for t0 in (0..full).step_by(group) {
        for ch in 0..width {
            for (i, v) in col.iter_mut().enumerate() {
                *v = c[(t0 + i) * width + ch];
            }
            minmax(&mut col, bits);
            for (i, v) in col.iter().enumerate() {
                out[(t0 + i) * width + ch] = *v;
            }
        }
    }
    for v in out[full * width..].iter_mut() {
        *v = f16(to_f16(*v));
    }
    out
}

pub struct Attention {
    pub out_err: f64,
    pub top_changed: f64,
    pub kl: f64,
}

/// Attention of the last `nq` positions over all earlier ones, in float64,
/// compared between an exact cache (k, v) and a quantized one (kq, vq).
#[allow(clippy::too_many_arguments)]
pub fn compare(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kq: &[f32],
    vq: &[f32],
    n: usize,
    nq: usize,
    q_heads: usize,
    kv_heads: usize,
    threads: usize,
) -> Attention {
    let (qd, kd) = (q_heads * HD, kv_heads * HD);
    let g = q_heads / kv_heads;
    let scale = 1.0 / (HD as f64).sqrt();
    let acc = Mutex::new((0.0f64, 0.0f64, 0usize, 0.0f64));
    par_rows(nq, threads, |r| {
        let i = n - nq + r;
        let (mut e, mut norm, mut changed, mut kl) = (0.0f64, 0.0f64, 0usize, 0.0f64);
        let mut s = vec![0.0f64; i + 1];
        let mut sq = vec![0.0f64; i + 1];
        for h in 0..q_heads {
            let qh = &q[i * qd + h * HD..i * qd + (h + 1) * HD];
            let kh = h / g;
            for j in 0..=i {
                let (a, b) = (
                    &k[j * kd + kh * HD..j * kd + (kh + 1) * HD],
                    &kq[j * kd + kh * HD..j * kd + (kh + 1) * HD],
                );
                let (mut x, mut y) = (0.0f64, 0.0f64);
                for ((qq, aa), bb) in qh.iter().zip(a).zip(b) {
                    x += *qq as f64 * *aa as f64;
                    y += *qq as f64 * *bb as f64;
                }
                s[j] = x * scale;
                sq[j] = y * scale;
            }
            let arg = |v: &[f64]| {
                v.iter().enumerate().fold(
                    (0, f64::MIN),
                    |(bi, bv), (j, &x)| if x > bv { (j, x) } else { (bi, bv) },
                )
            };
            let ((ai, m), (bi, mq)) = (arg(&s), arg(&sq));
            if ai != bi {
                changed += 1;
            }
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            let zq: f64 = sq.iter().map(|x| (x - mq).exp()).sum();
            let mut o = [0.0f64; HD];
            let mut oq = [0.0f64; HD];
            for j in 0..=i {
                let (p, pq) = ((s[j] - m).exp() / z, (sq[j] - mq).exp() / zq);
                kl += p * (p / pq).ln();
                let (a, b) = (
                    &v[j * kd + kh * HD..j * kd + (kh + 1) * HD],
                    &vq[j * kd + kh * HD..j * kd + (kh + 1) * HD],
                );
                for c in 0..HD {
                    o[c] += p * a[c] as f64;
                    oq[c] += pq * b[c] as f64;
                }
            }
            for c in 0..HD {
                e += (o[c] - oq[c]).powi(2);
                norm += o[c] * o[c];
            }
        }
        let mut t = acc.lock().unwrap();
        t.0 += e;
        t.1 += norm;
        t.2 += changed;
        t.3 += kl;
    });
    let (e, norm, changed, kl) = acc.into_inner().unwrap();
    let pairs = (nq * q_heads) as f64;
    Attention {
        out_err: (e / norm).sqrt(),
        top_changed: changed as f64 / pairs,
        kl: kl / pairs,
    }
}

/// Ratio of the loudest channel's mean |x| to the median channel's, and the
/// three loudest channels of the first head, over all tokens.
fn channel_stats(c: &[f32], width: usize) -> (f64, Vec<usize>) {
    let n = c.len() / width;
    let mut mean = vec![0.0f64; width];
    for t in c.chunks(width) {
        for (m, x) in mean.iter_mut().zip(t) {
            *m += (*x as f64).abs() / n as f64;
        }
    }
    let mut sorted = mean.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[width / 2];
    let mut first: Vec<usize> = (0..HD).collect();
    first.sort_by(|a, b| mean[*b].partial_cmp(&mean[*a]).unwrap());
    (sorted[width - 1] / median, first[..3].to_vec())
}

fn main() -> std::io::Result<()> {
    let path = arg("--gguf").expect("--gguf <Llama 3.2 3B file>");
    let nq: usize = arg("--queries").and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: usize = arg("--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let mut g = Gguf::open(&path)?;
    let idx = |g: &Gguf, n: &str| g.index(n).unwrap_or_else(|| panic!("tensor {n}"));
    let (emb, norm) = (
        idx(&g, "token_embd.weight"),
        idx(&g, "blk.0.attn_norm.weight"),
    );
    let (iq, ik, iv, irf) = (
        idx(&g, "blk.0.attn_q.weight"),
        idx(&g, "blk.0.attn_k.weight"),
        idx(&g, "blk.0.attn_v.weight"),
        idx(&g, "rope_freqs.weight"),
    );
    let eps = g
        .numbers
        .get("llama.attention.layer_norm_rms_epsilon")
        .copied()
        .unwrap_or(1e-5) as f32;
    let base = g
        .numbers
        .get("llama.rope.freq_base")
        .copied()
        .unwrap_or(500000.0);
    let gamma = g.rows(norm, 0, 1)?;
    let factors = g.rows(irf, 0, 1)?;
    let cols = gamma.len();
    let (qrows, krows) = (
        g.tensors[iq].dims[1] as usize,
        g.tensors[ik].dims[1] as usize,
    );
    let (q_heads, kv_heads) = (qrows / HD, krows / HD);
    let (wq, wk, wv) = (
        g.rows(iq, 0, qrows)?,
        g.rows(ik, 0, krows)?,
        g.rows(iv, 0, krows)?,
    );
    let ids = layer0::token_ids();
    let n = ids.len();
    // Layer-0 inputs per distinct token, then projections per distinct token.
    let mut distinct: Vec<u32> = ids.clone();
    distinct.sort();
    distinct.dedup();
    let (xd, _) = layer0::inputs(&mut g, &distinct, emb, &gamma, eps)?;
    let slot: HashMap<u32, usize> = distinct.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let proj = |w: &[f32], rows: usize| -> Vec<f32> {
        let out = Mutex::new(vec![0.0f32; distinct.len() * rows]);
        par_rows(distinct.len(), threads, |d| {
            let x = &xd[d * cols..(d + 1) * cols];
            let y: Vec<f32> = w
                .chunks(cols)
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum::<f64>() as f32
                })
                .collect();
            out.lock().unwrap()[d * rows..(d + 1) * rows].copy_from_slice(&y);
        });
        out.into_inner().unwrap()
    };
    let (qd, kd, vd) = (proj(&wq, qrows), proj(&wk, krows), proj(&wv, krows));
    let expand = |src: &[f32], rows: usize| -> Vec<f32> {
        ids.iter()
            .flat_map(|t| src[slot[t] * rows..(slot[t] + 1) * rows].to_vec())
            .collect()
    };
    let mut q = expand(&qd, qrows);
    let k_pre = expand(&kd, krows);
    let v = expand(&vd, krows);
    let mut k = k_pre.clone();
    for pos in 0..n {
        rope(&mut q[pos * qrows..(pos + 1) * qrows], pos, base, &factors);
        rope(&mut k[pos * krows..(pos + 1) * krows], pos, base, &factors);
    }
    println!(
        "layer 0 of Llama 3.2 3B over {n} tokens of AUTHORING.md; {q_heads} query heads, \
         {kv_heads} key/value heads of {HD}; RoPE base {base}; the last {nq} tokens attend \
         to all earlier ones"
    );
    let (rk_pre, top_pre) = channel_stats(&k_pre, krows);
    let (rk, top) = channel_stats(&k, krows);
    let (rv, _) = channel_stats(&v, krows);
    println!(
        "loudest channel's mean |x| over the median channel's: keys before RoPE {rk_pre:.1} \
         (head 0's loudest channels {top_pre:?}), keys after RoPE {rk:.1} ({top:?}), values {rv:.1}"
    );
    let kmax = k.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let vmax = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    println!("largest |key| {kmax:.3}, largest |value| {vmax:.3}");
    println!(
        "\n{:<46} {:>6} {:>10} {:>14} {:>12}",
        "cache format", "bits", "output err", "top-1 changed", "KL (nats)"
    );
    let q8 = |c: &[f32]| decode(Q8_0, &quantize_q8_0(c));
    let q4 = |c: &[f32]| decode(Q4_0, &quantize_q4_0(c));
    let rot = |c: &[f32], n: usize, f: &dyn Fn(&[f32]) -> Vec<f32>| {
        let mut r = c.to_vec();
        hadamard(&mut r, n);
        let mut d = f(&r);
        hadamard(&mut d, n);
        d
    };
    let fp8 = |c: &[f32], s: f32| {
        c.iter()
            .map(|x| E4M3.round(x / s) * s)
            .collect::<Vec<f32>>()
    };
    let pre_rope_channel = |bits: u32| {
        let mut kq = per_channel(&k_pre, krows, bits, 32);
        for pos in 0..n {
            rope(&mut kq[pos * krows..(pos + 1) * krows], pos, base, &factors);
        }
        kq
    };
    let formats: Vec<(&str, f64, Vec<f32>, Vec<f32>)> = vec![
        (
            "F16",
            16.0,
            k.iter().map(|x| f16(to_f16(*x))).collect(),
            v.iter().map(|x| f16(to_f16(*x))).collect(),
        ),
        (
            "FP8 E4M3, scale 1 (uncalibrated)",
            8.0,
            fp8(&k, 1.0),
            fp8(&v, 1.0),
        ),
        (
            "FP8 E4M3, a calibrated scale per tensor",
            8.0,
            fp8(&k, kmax / 448.0),
            fp8(&v, vmax / 448.0),
        ),
        ("Q8_0, per token (llama.cpp q8_0)", 8.5, q8(&k), q8(&v)),
        ("Q4_0, per token (llama.cpp q4_0)", 4.5, q4(&k), q4(&v)),
        (
            "Q4_0 per token, rotated as llama.cpp does",
            4.5,
            rot(&k, HD, &q4),
            rot(&v, 64, &q4),
        ),
        (
            "4-bit keys per channel, values per token",
            5.0,
            per_channel(&k, krows, 4, 32),
            per_token(&v, 4),
        ),
        (
            "4-bit keys per channel before RoPE, values per token",
            5.0,
            pre_rope_channel(4),
            per_token(&v, 4),
        ),
        (
            "4-bit keys and values per token",
            5.0,
            per_token(&k, 4),
            per_token(&v, 4),
        ),
        (
            "2-bit keys per channel, values per token (KIVI)",
            3.0,
            per_channel(&k, krows, 2, 32),
            per_token(&v, 2),
        ),
        (
            "2-bit keys per channel before RoPE, values per token",
            3.0,
            pre_rope_channel(2),
            per_token(&v, 2),
        ),
        (
            "2-bit keys and values per token",
            3.0,
            per_token(&k, 2),
            per_token(&v, 2),
        ),
    ];
    for (name, bits, kq, vq) in &formats {
        let a = compare(&q, &k, &v, kq, vq, n, nq, q_heads, kv_heads, threads);
        println!(
            "{:<46} {:>6.1} {:>10.4} {:>13.1}% {:>12.2e}",
            name,
            bits,
            a.out_err,
            100.0 * a.top_changed,
            a.kl
        );
    }
    // Which half is fragile: quantize only the keys, or only the values.
    println!("\nkeys against values, per token, the other half exact");
    for bits in [4u32, 2] {
        for (name, kq, vq) in [
            ("keys only", per_token(&k, bits), v.clone()),
            ("values only", k.clone(), per_token(&v, bits)),
        ] {
            let a = compare(&q, &k, &v, &kq, &vq, n, nq, q_heads, kv_heads, threads);
            println!(
                "{:<46} {:>6} {:>10.4} {:>13.1}% {:>12.2e}",
                format!("{bits}-bit, {name}"),
                "",
                a.out_err,
                100.0 * a.top_changed,
                a.kl
            );
        }
    }
    // Damage against context: the last 64 tokens of the first `m` tokens, with each
    // cache built from those m tokens only.
    println!("\ncontext sweep: the last 64 of the first m tokens; top token changed, KL");
    let sweep: [usize; 5] = [256, 512, 1024, 2048, n];
    print!("{:<46}", "cache format");
    for m in sweep {
        print!(" {:>17}", format!("m = {m}"));
    }
    println!();
    type Build<'a> = Box<dyn Fn(usize) -> (Vec<f32>, Vec<f32>) + 'a>;
    let rows: Vec<(&str, Build)> = vec![
        (
            "Q8_0, per token",
            Box::new(|m| (q8(&k[..m * krows]), q8(&v[..m * krows]))),
        ),
        (
            "Q4_0 per token, rotated as llama.cpp does",
            Box::new(|m| (rot(&k[..m * krows], HD, &q4), rot(&v[..m * krows], 64, &q4))),
        ),
        (
            "4-bit keys per channel before RoPE",
            Box::new(|m| {
                let mut kq = per_channel(&k_pre[..m * krows], krows, 4, 32);
                for pos in 0..m {
                    rope(&mut kq[pos * krows..(pos + 1) * krows], pos, base, &factors);
                }
                (kq, per_token(&v[..m * krows], 4))
            }),
        ),
        (
            "2-bit keys per channel (KIVI)",
            Box::new(|m| {
                (
                    per_channel(&k[..m * krows], krows, 2, 32),
                    per_token(&v[..m * krows], 2),
                )
            }),
        ),
    ];
    for (name, build) in &rows {
        print!("{name:<46}");
        for m in sweep {
            let (kq, vq) = build(m);
            let a = compare(
                &q[..m * qrows],
                &k[..m * krows],
                &v[..m * krows],
                &kq,
                &vq,
                m,
                64,
                q_heads,
                kv_heads,
                threads,
            );
            print!(" {:>7.1}% {:>8.1e}", 100.0 * a.top_changed, a.kl);
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_preserves_norms_and_relative_positions() {
        let factors = vec![1.0f32; HD / 2];
        let mut a: Vec<f32> = (0..HD).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut b: Vec<f32> = (0..HD).map(|i| (i as f32 * 0.11).cos()).collect();
        let dot = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (*p as f64) * (*q as f64))
                .sum::<f64>()
        };
        let n0 = dot(&a, &a);
        let (mut a2, mut b2) = (a.clone(), b.clone());
        rope(&mut a, 7, 500000.0, &factors);
        rope(&mut b, 3, 500000.0, &factors);
        rope(&mut a2, 107, 500000.0, &factors);
        rope(&mut b2, 103, 500000.0, &factors);
        assert!((dot(&a, &a) - n0).abs() < 1e-4);
        // the score depends only on the distance between positions (7-3 = 107-103)
        assert!((dot(&a, &b) - dot(&a2, &b2)).abs() < 1e-3);
    }

    #[test]
    fn the_hadamard_transform_is_orthogonal_and_its_own_inverse() {
        let x: Vec<f32> = (0..2 * HD).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut y = x.clone();
        hadamard(&mut y, HD);
        let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
        assert!((n(&x) - n(&y)).abs() < 1e-4);
        hadamard(&mut y, HD);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn per_channel_groups_keep_a_residual_in_sixteen_bits() {
        // 40 tokens of width 4: the first 32 are quantized per channel, the last 8 kept.
        let c: Vec<f32> = (0..160).map(|i| (i as f32 * 0.013).sin()).collect();
        let q = per_channel(&c, 4, 2, 32);
        for (a, b) in c[128..].iter().zip(&q[128..]) {
            assert!((a - b).abs() < 1e-3);
        }
        // 2 bits: four levels per channel group
        let mut levels: Vec<i64> = (0..32).map(|t| (q[t * 4] * 1e6) as i64).collect();
        levels.sort();
        levels.dedup();
        assert!(levels.len() <= 4);
    }

    #[test]
    fn identical_caches_agree_exactly() {
        let (n, nq, qh, kh) = (20, 5, 6, 2);
        let q: Vec<f32> = (0..n * qh * HD).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n * kh * HD).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n * kh * HD).map(|i| (i as f32 * 0.03).sin()).collect();
        let a = compare(&q, &k, &v, &k, &v, n, nq, qh, kh, 2);
        assert_eq!(a.out_err, 0.0);
        assert_eq!(a.top_changed, 0.0);
        assert!(a.kl.abs() < 1e-12);
    }
}
