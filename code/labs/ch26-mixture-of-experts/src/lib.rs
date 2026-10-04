//! Chapter 43 MoE lab.
//!
//! This lab deliberately makes routing metadata explicit. The reference implementation
//! executes assignments in logical token order. The grouped implementation reorders
//! work by expert and must use token IDs to reconstruct exactly the same outputs.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assignment {
    pub token: usize,
    pub expert: usize,
    pub gate: f64,
}

/// Tiny affine "expert". Real experts are gated FFNs; scalar affine maps make dispatch
/// and combine errors visible without hiding them behind matrix code.
pub fn expert(expert_id: usize, x: f64) -> f64 {
    let scale = (expert_id + 1) as f64;
    scale * x + 0.1 * scale
}

/// Independent reference path: preserve logical assignment order and accumulate
/// directly into the owning token.
pub fn reference(tokens: &[f64], assignments: &[Assignment]) -> Vec<f64> {
    let mut out = vec![0.0; tokens.len()];
    for a in assignments {
        out[a.token] += a.gate * expert(a.expert, tokens[a.token]);
    }
    out
}

/// Grouped execution path. Sorting changes physical row order only. The token field is
/// the inverse map used by combine, so execution order cannot change sequence identity.
pub fn grouped(tokens: &[f64], assignments: &[Assignment]) -> Vec<f64> {
    let mut packed = assignments.to_vec();
    packed.sort_by_key(|a| (a.expert, a.token));
    let mut out = vec![0.0; tokens.len()];
    for a in packed {
        let y = expert(a.expert, tokens[a.token]);
        out[a.token] += a.gate * y;
    }
    out
}

/// Intentionally broken combine: it forgets token identity and assigns grouped results
/// round-robin. Tests demonstrate why an inverse row map is a correctness requirement.
pub fn broken_without_inverse_map(tokens: &[f64], assignments: &[Assignment]) -> Vec<f64> {
    let mut packed = assignments.to_vec();
    packed.sort_by_key(|a| (a.expert, a.token));
    let mut out = vec![0.0; tokens.len()];
    for (row, a) in packed.into_iter().enumerate() {
        let wrong_token = row % tokens.len();
        out[wrong_token] += a.gate * expert(a.expert, tokens[a.token]);
    }
    out
}

/// Expected distinct-expert union under the chapter's independent/uniform oracle.
pub fn expected_union(experts: usize, top_k: usize, batch: usize) -> f64 {
    assert!(experts > 0 && top_k > 0 && top_k <= experts);
    let miss_one = 1.0 - top_k as f64 / experts as f64;
    experts as f64 * (1.0 - miss_one.powi(batch as i32))
}

/// Counts assignment rows per expert. Zero-row experts remain visible in the returned
/// vector, which is important for capacity and imbalance diagnostics.
pub fn occupancy(experts: usize, assignments: &[Assignment]) -> Vec<usize> {
    let mut n = vec![0usize; experts];
    for a in assignments {
        assert!(a.expert < experts);
        n[a.expert] += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f64>, Vec<Assignment>) {
        let t = vec![1.0, 2.0, -1.0];
        let a = vec![
            Assignment {
                token: 0,
                expert: 2,
                gate: 0.7,
            },
            Assignment {
                token: 0,
                expert: 0,
                gate: 0.3,
            },
            Assignment {
                token: 1,
                expert: 1,
                gate: 0.6,
            },
            Assignment {
                token: 1,
                expert: 2,
                gate: 0.4,
            },
            Assignment {
                token: 2,
                expert: 2,
                gate: 0.8,
            },
            Assignment {
                token: 2,
                expert: 3,
                gate: 0.2,
            },
        ];
        (t, a)
    }

    #[test]
    fn grouped_matches_reference_despite_reordering() {
        let (t, a) = fixture();
        assert_eq!(reference(&t, &a), grouped(&t, &a));
    }

    #[test]
    fn missing_inverse_map_is_detected() {
        let (t, a) = fixture();
        assert_ne!(reference(&t, &a), broken_without_inverse_map(&t, &a));
    }

    #[test]
    fn occupancy_exposes_hot_expert() {
        let (_, a) = fixture();
        assert_eq!(occupancy(4, &a), vec![1, 1, 3, 1]);
    }

    #[test]
    fn qwen_shape_union_matches_derived_values() {
        let u1 = expected_union(128, 8, 1);
        let u32 = expected_union(128, 8, 32);
        let u64 = expected_union(128, 8, 64);
        assert!((u1 - 8.0).abs() < 1e-12);
        assert!((u32 - 111.8).abs() < 0.2);
        assert!((u64 - 125.9).abs() < 0.2);
    }

    #[test]
    fn gate_expert_pairing_matters() {
        let (t, mut a) = fixture();
        let good = reference(&t, &a);
        // Swap gates between two different experts for the same logical token.
        let g0 = a[0].gate;
        a[0].gate = a[1].gate;
        a[1].gate = g0;
        assert_ne!(good, reference(&t, &a));
    }
}
