//! The engine's arithmetic. Every kernel fixes its summation order, so an
//! output element is the same number whatever batch it was computed in.

/// Four lanes: the width of one 128-bit vector register (NEON, SSE).
type Quad = [f32; 4];

#[inline(always)]
fn add(a: Quad, b: Quad) -> Quad {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline(always)]
fn mul(a: Quad, b: Quad) -> Quad {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

#[inline(always)]
fn quad(c: &[f32; 16], k: usize) -> Quad {
    [c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]]
}

/// The one fixed way sixteen running sums become one number: the four quads
/// are added lane by lane, ((s0 + s1) + (s2 + s3)), then the four lanes as
/// (v0 + v2) + (v1 + v3).
#[inline(always)]
fn combine(s: [Quad; 4]) -> f32 {
    let v = add(add(s[0], s[1]), add(s[2], s[3]));
    (v[0] + v[2]) + (v[1] + v[3])
}

/// Dot product in a fixed order: sixteen running sums, kept as four quads so
/// the compiler holds each in one vector register, combined by [`combine`],
/// then the tail in index order. The order is part of the definition: another
/// order gives slightly different bits, and a combine order the compiler
/// cannot map onto whole registers runs at half the speed (Chapter 42).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut s = [[0.0f32; 4]; 4];
    for (x, y) in ca.zip(cb) {
        let x: &[f32; 16] = x.try_into().unwrap();
        let y: &[f32; 16] = y.try_into().unwrap();
        for (k, acc) in s.iter_mut().enumerate() {
            *acc = add(*acc, mul(quad(x, k), quad(y, k)));
        }
    }
    let mut sum = combine(s);
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// Four dot products that share their first operand. The arithmetic of each
/// is exactly that of [`dot`] --- the same sixteen running sums, combined the
/// same way, then the same tail --- so `dot4(a, b)[j]` equals `dot(a, b[j])`
/// bit for bit; the difference is that each element of `a` is loaded once for
/// all four, and sixty-four independent sums keep the arithmetic units busy.
pub fn dot4(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = a.len() / 16 * 16;
    let mut s = [[[0.0f32; 4]; 4]; 4];
    let chunks = a[..n]
        .chunks_exact(16)
        .zip(b[0][..n].chunks_exact(16))
        .zip(b[1][..n].chunks_exact(16))
        .zip(b[2][..n].chunks_exact(16))
        .zip(b[3][..n].chunks_exact(16));
    for ((((w, x0), x1), x2), x3) in chunks {
        let w: &[f32; 16] = w.try_into().unwrap();
        let xs: [&[f32; 16]; 4] = [
            x0.try_into().unwrap(),
            x1.try_into().unwrap(),
            x2.try_into().unwrap(),
            x3.try_into().unwrap(),
        ];
        let wq = [quad(w, 0), quad(w, 1), quad(w, 2), quad(w, 3)];
        for (acc, x) in s.iter_mut().zip(xs) {
            for (k, (a, wk)) in acc.iter_mut().zip(wq).enumerate() {
                *a = add(*a, mul(wk, quad(x, k)));
            }
        }
    }
    let mut out = [0.0f32; 4];
    for ((o, acc), x) in out.iter_mut().zip(s).zip(b) {
        let mut sum = combine(acc);
        for (p, q) in a[n..].iter().zip(&x[n..]) {
            sum += p * q;
        }
        *o = sum;
    }
    out
}

/// `y[t][r] = dot(W[r], x[t])` computed one output at a time: the reference
/// form the oracle uses. W is row-major `[rows, cols]`, `x` is `[n, cols]`,
/// `y` is `[n, rows]`.
pub fn matmul_rows_simple(w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), n * cols);
    debug_assert_eq!(y.len(), n * rows);
    for (r, w_row) in w.chunks_exact(cols).enumerate() {
        for (t, x_row) in x.chunks_exact(cols).enumerate() {
            y[t * rows + r] = dot(w_row, x_row);
        }
    }
}

/// The same products, tiled across tokens: the weight row is the outer loop,
/// so each row is read from memory once per step, and four tokens at a time
/// share every load of it. Bit-identical to [`matmul_rows_simple`], because
/// tiling across tokens never splits a reduction.
pub fn matmul_rows(w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), n * cols);
    debug_assert_eq!(y.len(), n * rows);
    let token = |t: usize| &x[t * cols..(t + 1) * cols];
    for (r, w_row) in w.chunks_exact(cols).enumerate() {
        let mut t = 0;
        while t + 4 <= n {
            let out = dot4(w_row, [token(t), token(t + 1), token(t + 2), token(t + 3)]);
            for (j, value) in out.into_iter().enumerate() {
                y[(t + j) * rows + r] = value;
            }
            t += 4;
        }
        while t < n {
            y[t * rows + r] = dot(w_row, token(t));
            t += 1;
        }
    }
}

/// The signature shared by the matrix kernels: `(w, rows, cols, x, n, y)`.
pub type MatmulKernel = fn(&[f32], usize, usize, &[f32], usize, &mut [f32]);

/// [`matmul_rows`] with the output rows split among `threads` scoped threads.
/// Each output is still computed by one thread with the same arithmetic, so
/// the result does not depend on the thread count, bit for bit.
pub fn matmul_rows_threads(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
    y: &mut [f32],
    threads: usize,
) {
    split_rows(matmul_rows, w, rows, cols, x, n, y, threads);
}

/// Run `kernel` with its output rows split among `threads` scoped threads,
/// each writing a private `[n, rows/threads]` block that is copied into `y`.
/// Threads are spawned per call, which is simple and costs tens of
/// microseconds; a pool of waiting threads would remove that.
#[allow(clippy::too_many_arguments)]
pub fn split_rows(
    kernel: MatmulKernel,
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
    y: &mut [f32],
    threads: usize,
) {
    if threads <= 1 || rows < 8 * threads {
        return kernel(w, rows, cols, x, n, y);
    }
    let per = rows.div_ceil(threads);
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .filter(|i| i * per < rows)
            .map(|i| {
                let (r0, r1) = (i * per, ((i + 1) * per).min(rows));
                scope.spawn(move || {
                    let mut local = vec![0.0f32; n * (r1 - r0)];
                    kernel(&w[r0 * cols..r1 * cols], r1 - r0, cols, x, n, &mut local);
                    (r0, local)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("matmul worker panicked"))
            .collect()
    });
    for (r0, local) in parts {
        let nr = local.len() / n;
        for t in 0..n {
            y[t * rows + r0..t * rows + r0 + nr].copy_from_slice(&local[t * nr..(t + 1) * nr]);
        }
    }
}

/// RMSNorm: `out = x / sqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = dot(x, x) / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
        *o = xi * scale * wi;
    }
}

/// Rotary position embedding on one head, pairing element `i` with element
/// `i + d/2` (the "rotate half" layout). Angle: `pos * base^(-2i/d)`.
pub fn rope(head: &mut [f32], pos: usize, base: f32) {
    let d = head.len();
    let half = d / 2;
    for i in 0..half {
        let freq = base.powf(-2.0 * i as f32 / d as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = a * sin + b * cos;
    }
}

/// Numerically stable softmax in place: subtract the maximum, exponentiate,
/// then divide by the sum accumulated in index order.
pub fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Index of the largest value; the lowest index wins ties.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_a_long_hand_sum_on_small_integers() {
        let a: Vec<f32> = (1..=19).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=19).map(|i| (20 - i) as f32).collect();
        let expected: f32 = (1..=19).map(|i| (i * (20 - i)) as f32).sum();
        assert_eq!(dot(&a, &b), expected); // exact: every partial sum is an integer below 2^24
    }

    #[test]
    fn matmul_rows_is_batch_invariant_bit_for_bit() {
        let (rows, cols) = (7, 13);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let x: Vec<f32> = (0..5 * cols)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 11.0)
            .collect();
        let mut batched = vec![0.0; 5 * rows];
        matmul_rows(&w, rows, cols, &x, 5, &mut batched);
        for t in 0..5 {
            let mut alone = vec![0.0; rows];
            matmul_rows(&w, rows, cols, &x[t * cols..(t + 1) * cols], 1, &mut alone);
            for r in 0..rows {
                assert_eq!(alone[r].to_bits(), batched[t * rows + r].to_bits());
            }
        }
    }

    #[test]
    fn dot4_tiling_and_threads_change_no_bits() {
        let (rows, cols) = (37, 29); // odd sizes exercise every tail
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        for n in 1..=9 {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i * 53 % 97) as f32 - 48.0) / 11.0)
                .collect();
            let mut reference = vec![0.0; n * rows];
            matmul_rows_simple(&w, rows, cols, &x, n, &mut reference);
            for threads in 1..=5 {
                let mut y = vec![0.0; n * rows];
                matmul_rows_threads(&w, rows, cols, &x, n, &mut y, threads);
                assert!(
                    y.iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "n={n} threads={threads}"
                );
            }
        }
    }

    #[test]
    fn rope_is_a_rotation_and_position_zero_is_identity() {
        let original: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let mut head = original.clone();
        rope(&mut head, 0, 10_000.0);
        assert_eq!(head, original);
        rope(&mut head, 5, 10_000.0);
        let norm = |v: &[f32]| dot(v, v).sqrt();
        assert!((norm(&head) - norm(&original)).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let mut v = vec![1000.0, 1001.0, 999.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&v), 1);
    }
}
