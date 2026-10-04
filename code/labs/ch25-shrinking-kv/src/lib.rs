//! Chapter 42 structural/numerical lab.
//! The lab separates two claims: GQA changes head ownership, while MLA projection
//! absorption is an algebraic rewrite of a fixed linear map.

pub fn kv_head(q_head: usize, hq: usize, hkv: usize) -> usize {
    assert!(hq > 0 && hkv > 0 && hq.is_multiple_of(hkv) && q_head < hq);
    q_head / (hq / hkv)
}
pub fn gqa_bytes(layers: usize, hkv: usize, dh: usize, bytes: usize) -> usize {
    2 * layers * hkv * dh * bytes
}
pub fn mla_bytes(layers: usize, dc: usize, dr: usize, bytes: usize) -> usize {
    layers * (dc + dr) * bytes
}
pub fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
/// Explicit path: project every historical latent, then take weighted sum.
pub fn explicit_value(w: &[Vec<f64>], cs: &[Vec<f64>], alpha: &[f64]) -> Vec<f64> {
    let ys: Vec<_> = cs.iter().map(|c| matvec(w, c)).collect();
    (0..w.len())
        .map(|j| ys.iter().zip(alpha).map(|(y, a)| y[j] * a).sum())
        .collect()
}
/// Absorbed path: reduce latents first, apply the same fixed linear map once.
pub fn absorbed_value(w: &[Vec<f64>], cs: &[Vec<f64>], alpha: &[f64]) -> Vec<f64> {
    let d = cs[0].len();
    let mut z = vec![0.0; d];
    for (c, a) in cs.iter().zip(alpha) {
        for j in 0..d {
            z[j] += a * c[j];
        }
    }
    matvec(w, &z)
}
pub fn close(a: &[f64], b: &[f64], eps: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gqa_mapping() {
        assert_eq!(
            (0..8).map(|i| kv_head(i, 8, 2)).collect::<Vec<_>>(),
            vec![0, 0, 0, 0, 1, 1, 1, 1]
        );
    }
    #[test]
    fn measured_llama_bytes() {
        assert_eq!(gqa_bytes(28, 8, 128, 2), 112 * 1024);
    }
    #[test]
    fn measured_qwen_bytes() {
        assert_eq!(gqa_bytes(36, 2, 128, 2), 36 * 1024);
    }
    #[test]
    fn mla_example_bytes() {
        assert_eq!(mla_bytes(61, 512, 64, 2), 70272);
    }
    #[test]
    fn absorption_matches_independent_order() {
        let w = vec![vec![2., -1., 0.5], vec![0.25, 3., -2.]];
        let cs = vec![vec![1., 2., 3.], vec![-2., 1., 4.], vec![0.5, -1., 2.]];
        let a = vec![0.2, 0.3, 0.5];
        assert!(close(
            &explicit_value(&w, &cs, &a),
            &absorbed_value(&w, &cs, &a),
            1e-12
        ));
    }
    #[test]
    fn wrong_group_is_detectable() {
        assert_ne!(kv_head(3, 8, 2), kv_head(4, 8, 2));
    }
}
