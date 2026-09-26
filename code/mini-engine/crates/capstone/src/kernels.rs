//! The engine's arithmetic. Every kernel fixes its summation order, so an
//! output element is the same number whatever batch it was computed in.

/// Dot product in a fixed order: eight running sums over the length, combined
/// as ((0+4)+(1+5))+((2+6)+(3+7)), then the tail in index order. Eight
/// independent sums let the compiler vectorize the loop; fixing how they are
/// combined keeps the result independent of everything but the two inputs.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for lane in 0..8 {
            acc[lane] += x[lane] * y[lane];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// `y[t][r] = dot(W[r], x[t])` for `n` tokens: W is row-major `[rows, cols]`,
/// `x` is `[n, cols]`, `y` is `[n, rows]`. The weight row is the outer loop,
/// so each row is read from memory once per step and reused for every token
/// in the batch while it is still in cache: the source of batching's gain.
pub fn matmul_rows(w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), n * cols);
    debug_assert_eq!(y.len(), n * rows);
    for (r, w_row) in w.chunks_exact(cols).enumerate() {
        for (t, x_row) in x.chunks_exact(cols).enumerate() {
            y[t * rows + r] = dot(w_row, x_row);
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
