//! Weight quantizers for a matrix W (rows x cols, row-major; a row is one
//! output feature): symmetric round-to-nearest with one scale per tensor, per
//! row or per group of columns; an AWQ-style activation-aware scaling; and
//! GPTQ. Each returns the dequantized weights, so that errors are measured in
//! the layer's output.

/// Runs `f(row)` for every row, rows split over `threads` threads.
pub fn par_rows<F: Fn(usize) + Sync>(rows: usize, threads: usize, f: F) {
    let t = threads.max(1).min(rows.max(1));
    let per = rows.div_ceil(t);
    std::thread::scope(|s| {
        for k in 0..t {
            let f = &f;
            s.spawn(move || {
                for r in k * per..((k + 1) * per).min(rows) {
                    f(r);
                }
            });
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grouping {
    Tensor,
    Row,
    Group(usize),
}

/// Rounds `x` to one of 2^(bits-1) - 1 levels each side of zero with scale `s`.
#[inline]
fn round_to(x: f32, s: f32, qmax: f32) -> f32 {
    if s == 0.0 {
        0.0
    } else {
        (x / s).round().clamp(-qmax, qmax) * s
    }
}

/// Symmetric round-to-nearest. Returns the dequantized weights and the bits per
/// weight, counting each scale as 16 bits (scales are kept in 32 bits here).
pub fn rtn(w: &[f32], rows: usize, cols: usize, bits: u32, g: Grouping) -> (Vec<f32>, f64) {
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let amax = |v: &[f32]| v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let mut out = w.to_vec();
    let scales = match g {
        Grouping::Tensor => {
            let s = amax(w) / qmax;
            out.iter_mut().for_each(|x| *x = round_to(*x, s, qmax));
            1
        }
        Grouping::Row => {
            for row in out.chunks_mut(cols) {
                let s = amax(row) / qmax;
                row.iter_mut().for_each(|x| *x = round_to(*x, s, qmax));
            }
            rows
        }
        Grouping::Group(n) => {
            assert!(cols.is_multiple_of(n));
            for grp in out.chunks_mut(n) {
                let s = amax(grp) / qmax;
                grp.iter_mut().for_each(|x| *x = round_to(*x, s, qmax));
            }
            rows * cols / n
        }
    };
    let bpw = bits as f64 + 16.0 * scales as f64 / (rows * cols) as f64;
    (out, bpw)
}

/// Squared output error summed over tokens: sum_t c_t |(W - Q) x_t|^2, and the
/// squared output norm sum_t c_t |W x_t|^2. `x` holds n tokens of `cols` values
/// and `c` their counts.
pub fn output_error(
    w: &[f32],
    q: &[f32],
    cols: usize,
    x: &[f32],
    c: &[f64],
    threads: usize,
) -> (f64, f64) {
    let rows = w.len() / cols;
    let parts = std::sync::Mutex::new((0.0f64, 0.0f64));
    par_rows(rows, threads, |r| {
        let (wr, qr) = (&w[r * cols..(r + 1) * cols], &q[r * cols..(r + 1) * cols]);
        let (mut e, mut n) = (0.0f64, 0.0f64);
        for (xt, &ct) in x.chunks_exact(cols).zip(c) {
            let (mut y, mut yq) = (0.0f64, 0.0f64);
            for ((a, b), v) in wr.iter().zip(qr).zip(xt) {
                y += (*a as f64) * (*v as f64);
                yq += (*b as f64) * (*v as f64);
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

/// AWQ-style: scale input channel j of W by s_j = a_j^alpha (a_j the mean |x_j|
/// over calibration tokens), quantize, and undo the scale; alpha is searched on
/// the calibration tokens. Returns the weights, the chosen alpha and the bpw.
#[allow(clippy::too_many_arguments)]
pub fn awq(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
    group: usize,
    x: &[f32],
    c: &[f64],
    threads: usize,
) -> (Vec<f32>, f32, f64) {
    let total: f64 = c.iter().sum();
    let mut a = vec![0.0f64; cols];
    for (xt, &ct) in x.chunks_exact(cols).zip(c) {
        for (aj, v) in a.iter_mut().zip(xt) {
            *aj += ct * (*v as f64).abs() / total;
        }
    }
    let mut best: Option<(f64, f32, Vec<f32>, f64)> = None;
    for step in 0..=10 {
        let alpha = step as f32 / 10.0;
        let mut s: Vec<f32> = a
            .iter()
            .map(|&v| (v.max(1e-8) as f32).powf(alpha).max(1e-4))
            .collect();
        let (smax, smin) = s
            .iter()
            .fold((0.0f32, f32::MAX), |(hi, lo), &v| (hi.max(v), lo.min(v)));
        let norm = (smax * smin).sqrt();
        s.iter_mut().for_each(|v| *v /= norm);
        let scaled: Vec<f32> = w
            .chunks_exact(cols)
            .flat_map(|row| row.iter().zip(&s).map(|(x, sj)| x * sj).collect::<Vec<_>>())
            .collect();
        let (qs, bpw) = rtn(&scaled, rows, cols, bits, Grouping::Group(group));
        let q: Vec<f32> = qs
            .chunks_exact(cols)
            .flat_map(|row| row.iter().zip(&s).map(|(x, sj)| x / sj).collect::<Vec<_>>())
            .collect();
        let (e, _) = output_error(w, &q, cols, x, c, threads);
        if best.as_ref().is_none_or(|b| e < b.0) {
            best = Some((e, alpha, q, bpw));
        }
    }
    let (_, alpha, q, bpw) = best.expect("at least one alpha");
    (q, alpha, bpw)
}

/// Cholesky factor L (lower, row-major) of a symmetric positive definite matrix.
pub fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            let s = a[i * n + j]
                - l[i * n..i * n + j]
                    .iter()
                    .zip(&l[j * n..j * n + j])
                    .map(|(x, y)| x * y)
                    .sum::<f64>();
            l[i * n + j] = if i == j {
                assert!(s > 0.0, "matrix not positive definite");
                s.sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    l
}

/// Inverse of a lower-triangular matrix, by forward substitution per column.
pub fn invert_lower(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0f64; n * n];
    let mut col = vec![0.0f64; n];
    for j in 0..n {
        col.iter_mut().for_each(|v| *v = 0.0);
        for i in j..n {
            let s = if i == j { 1.0 } else { 0.0 }
                - l[i * n + j..i * n + i]
                    .iter()
                    .zip(&col[j..i])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            col[i] = s / l[i * n + i];
        }
        for i in j..n {
            inv[i * n + j] = col[i];
        }
    }
    inv
}

/// The upper-triangular U with H^-1 = U^T U, which GPTQ needs, without forming
/// H^-1: with J the index reversal, if J H J = L L^T then U = J L^-1 J.
pub fn gptq_factor(h: &[f64], n: usize) -> Vec<f64> {
    let mut rev = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            rev[i * n + j] = h[(n - 1 - i) * n + (n - 1 - j)];
        }
    }
    let li = invert_lower(&cholesky(&rev, n), n);
    let mut u = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            u[i * n + j] = li[(n - 1 - i) * n + (n - 1 - j)];
        }
    }
    u
}

/// GPTQ (Frantar et al.): quantize W column by column with a scale per group of
/// `group` columns, and after each column spread its rounding error over the
/// columns not yet quantized, weighted by the inverse Hessian of the layer's
/// squared output error, H = 2 X^T X. Rows are independent and run in parallel.
#[allow(clippy::too_many_arguments)]
pub fn gptq(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
    group: usize,
    x: &[f32],
    c: &[f64],
    threads: usize,
) -> (Vec<f32>, f64) {
    // H = 2/N sum_t c_t x_t x_t^T, damped by 1% of its mean diagonal.
    let total: f64 = c.iter().sum();
    let hm = std::sync::Mutex::new(vec![0.0f64; cols * cols]);
    par_rows(cols, threads, |i| {
        let mut row = vec![0.0f64; cols];
        for (xt, &ct) in x.chunks_exact(cols).zip(c) {
            let xi = xt[i] as f64 * ct;
            for (r, v) in row.iter_mut().zip(xt) {
                *r += xi * *v as f64;
            }
        }
        let mut h = hm.lock().unwrap();
        for (d, r) in h[i * cols..(i + 1) * cols].iter_mut().zip(&row) {
            *d = 2.0 * r / total;
        }
    });
    let mut h = hm.into_inner().unwrap();
    let damp = 0.01 * (0..cols).map(|i| h[i * cols + i]).sum::<f64>() / cols as f64;
    for i in 0..cols {
        h[i * cols + i] += damp;
    }
    let u = gptq_factor(&h, cols);
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let out = std::sync::Mutex::new(vec![0.0f32; rows * cols]);
    par_rows(rows, threads, |r| {
        let mut wr: Vec<f64> = w[r * cols..(r + 1) * cols]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mut qr = vec![0.0f32; cols];
        let mut s = 0.0f32;
        for i in 0..cols {
            if i % group == 0 {
                // the group's scale, from the weights as updated so far
                let amax = wr[i..i + group].iter().fold(0.0f64, |a, v| a.max(v.abs()));
                s = (amax as f32) / qmax;
            }
            let q = round_to(wr[i] as f32, s, qmax);
            qr[i] = q;
            let err = (wr[i] - q as f64) / u[i * cols + i];
            for (wj, uj) in wr[i + 1..]
                .iter_mut()
                .zip(&u[i * cols + i + 1..(i + 1) * cols])
            {
                *wj -= err * uj;
            }
        }
        out.lock().unwrap()[r * cols..(r + 1) * cols].copy_from_slice(&qr);
    });
    let bpw = bits as f64 + 16.0 / group as f64;
    (out.into_inner().unwrap(), bpw)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn the_worked_rounding_example() {
        // Eight weights, 4 bits, one scale: amax 0.9 / 7 levels.
        let w = [0.9f32, -0.35, 0.12, 0.05, -0.61, 0.28, -0.02, 0.44];
        let (q, bpw) = rtn(&w, 1, 8, 4, Grouping::Tensor);
        let s = 0.9f32 / 7.0;
        assert!((q[1] - (-3.0 * s)).abs() < 1e-6);
        assert!((q[3] - 0.0).abs() < 1e-6);
        assert!((bpw - 6.0).abs() < 1e-12); // 4 bits + 16 bits of scale over 8 weights
    }

    #[test]
    fn the_gptq_factor_inverts_the_hessian() {
        let n = 24;
        let a = fill(n * 40, 7);
        let mut h = vec![0.0f64; n * n];
        for t in a.chunks_exact(n) {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += (t[i] * t[j]) as f64;
                }
            }
        }
        for i in 0..n {
            h[i * n + i] += 0.1;
        }
        let u = gptq_factor(&h, n);
        // H (U^T U) = I, and U is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0);
            }
            for j in 0..n {
                let mut v = 0.0;
                for k in 0..n {
                    let hinv_kj: f64 = (0..n).map(|m| u[m * n + k] * u[m * n + j]).sum();
                    v += h[i * n + k] * hinv_kj;
                }
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn activation_aware_methods_beat_plain_rounding_on_outlier_channels() {
        let (rows, cols, n) = (32, 128, 256);
        let w = fill(rows * cols, 1);
        let mut x = fill(n * cols, 2);
        for t in x.chunks_exact_mut(cols) {
            for j in [3, 40, 97] {
                t[j] *= 30.0; // a few loud input channels
            }
        }
        let c = vec![1.0f64; n];
        let (r, _) = rtn(&w, rows, cols, 4, Grouping::Group(32));
        let (a, _, _) = awq(&w, rows, cols, 4, 32, &x, &c, 2);
        let (g, _) = gptq(&w, rows, cols, 4, 32, &x, &c, 2);
        let e = |q: &[f32]| output_error(&w, q, cols, &x, &c, 2).0;
        assert!(e(&a) < e(&r));
        assert!(e(&g) < e(&r));
    }
}
