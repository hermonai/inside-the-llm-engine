use engine0::rope::{
    rope_in_place, rope_reference, Pairing, RopeConfig, RopeError, MAX_EXACT_POSITION,
};
use engine0::tensor::{OwnedTensor, TensorView};

fn tensor(shape: &[usize], data: &[f32]) -> OwnedTensor {
    OwnedTensor::from_vec(shape.to_vec(), data.to_vec()).unwrap()
}
fn config(d: usize, r: usize, pairing: Pairing) -> RopeConfig {
    RopeConfig::try_new(d, r, 100.0, pairing).unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 2e-6 + 2e-6 * b.abs(), "{a} != {b}");
    }
}
fn norm(x: &[f32]) -> f64 {
    x.iter().map(|&x| f64::from(x).powi(2)).sum()
}
fn dot(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}
fn bits(x: &OwnedTensor) -> Vec<u32> {
    x.as_slice().iter().map(|x| x.to_bits()).collect()
}

#[test]
fn hand_computed_positive_rotation() {
    let x = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let y = rope_reference(&x.view(), 1, config(4, 4, Pairing::Adjacent)).unwrap();
    close(
        y.as_slice(),
        &[
            (1.0_f64.cos() - 2.0 * 1.0_f64.sin()) as f32,
            (1.0_f64.sin() + 2.0 * 1.0_f64.cos()) as f32,
            (3.0 * 0.1_f64.cos() - 4.0 * 0.1_f64.sin()) as f32,
            (3.0 * 0.1_f64.sin() + 4.0 * 0.1_f64.cos()) as f32,
        ],
    );
}
#[test]
fn zero_preserves_signed_zero_bits_in_both_paths() {
    let mut x = tensor(&[1, 4], &[-0., 0., 1., -2.]);
    let old = bits(&x);
    assert_eq!(
        bits(&rope_reference(&x.view(), 0, config(4, 4, Pairing::Adjacent)).unwrap()),
        old
    );
    rope_in_place(&mut x, 0, config(4, 4, Pairing::SplitHalf)).unwrap();
    assert_eq!(bits(&x), old);
}
#[test]
fn both_pairings_preserve_norms_and_inverse_with_f32_tolerance() {
    for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
        for p in [1, 7, 2048, 32768, -7] {
            let x = tensor(&[2, 5], &[1., 2., 3., 4., -0., -2., 1., 0.5, -3., 9.]);
            let y = rope_reference(&x.view(), p, config(5, 4, pairing)).unwrap();
            for (x, y) in x.as_slice().chunks(5).zip(y.as_slice().chunks(5)) {
                assert!((norm(x) - norm(y)).abs() <= 3e-6 * norm(x));
            }
            close(
                rope_reference(&y.view(), -p, config(5, 4, pairing))
                    .unwrap()
                    .as_slice(),
                x.as_slice(),
            );
        }
    }
}
#[test]
fn group_law_uses_fixed_frequency_contract() {
    let x = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let c = config(4, 4, Pairing::Adjacent);
    let y = rope_reference(&x.view(), 3, c).unwrap();
    close(
        rope_reference(&y.view(), 7, c).unwrap().as_slice(),
        rope_reference(&x.view(), 10, c).unwrap().as_slice(),
    );
}
#[test]
fn relative_inner_product_has_key_minus_query_sign() {
    let q = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let k = tensor(&[1, 4], &[2., -1., 1., 3.]);
    for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
        let c = config(4, 4, pairing);
        let qp = rope_reference(&q.view(), 2, c).unwrap();
        let kn = rope_reference(&k.view(), 5, c).unwrap();
        let delta = rope_reference(&k.view(), 3, c).unwrap();
        let wrong = rope_reference(&k.view(), -3, c).unwrap();
        assert!(
            (dot(qp.as_slice(), kn.as_slice()) - dot(q.as_slice(), delta.as_slice())).abs() < 4e-6
        );
        assert!(
            (dot(qp.as_slice(), kn.as_slice()) - dot(q.as_slice(), wrong.as_slice())).abs() > 1.0
        );
    }
}
#[test]
fn common_position_shift_preserves_fixed_vector_score() {
    let q = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let k = tensor(&[1, 4], &[2., -1., 1., 3.]);
    let c = config(4, 4, Pairing::Adjacent);
    let score = |p, n| {
        dot(
            rope_reference(&q.view(), p, c).unwrap().as_slice(),
            rope_reference(&k.view(), n, c).unwrap().as_slice(),
        )
    };
    assert!((score(2, 5) - score(102, 105)).abs() < 4e-6);
}
#[test]
fn partial_prefix_preserves_tail_bits_and_allows_odd_head_width() {
    for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
        let mut x = tensor(&[2, 5], &[1., 2., 3., 4., -0., 5., 6., 7., 8., 9.]);
        let old = bits(&x);
        rope_in_place(&mut x, 7, config(5, 4, pairing)).unwrap();
        assert_eq!(bits(&x)[4], old[4]);
        assert_eq!(bits(&x)[9], old[9]);
    }
}
#[test]
fn split_half_is_permutation_conjugate_not_same_layout() {
    let a = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let s = tensor(&[1, 4], &[1., 3., 2., 4.]);
    let ya = rope_reference(&a.view(), 7, config(4, 4, Pairing::Adjacent)).unwrap();
    let ys = rope_reference(&s.view(), 7, config(4, 4, Pairing::SplitHalf)).unwrap();
    close(
        ys.as_slice(),
        &[
            ya.as_slice()[0],
            ya.as_slice()[2],
            ya.as_slice()[1],
            ya.as_slice()[3],
        ],
    );
    let wrong = rope_reference(&a.view(), 7, config(4, 4, Pairing::SplitHalf)).unwrap();
    assert!(ya
        .as_slice()
        .iter()
        .zip(wrong.as_slice())
        .any(|(a, b)| (a - b).abs() > 0.5));
    assert!((norm(ya.as_slice()) - norm(wrong.as_slice())).abs() < 1e-5);
}
#[test]
fn borrowed_strided_offset_input_matches_canonical() {
    let storage = [
        99., 1., 99., 2., 99., 3., 99., 4., 99., -2., 99., 1., 99., 0.5, 99., -3.,
    ];
    let v = TensorView::try_from_parts(&storage, vec![2, 4], vec![8, 2], 1).unwrap();
    let x = tensor(&[2, 4], &[1., 2., 3., 4., -2., 1., 0.5, -3.]);
    let c = config(4, 4, Pairing::Adjacent);
    let result = rope_reference(&v, 7, c).unwrap();
    close(
        result.as_slice(),
        rope_reference(&x.view(), 7, c).unwrap().as_slice(),
    );
    assert_eq!(result.strides(), &[4, 1]);
    assert_eq!(storage[0], 99.);
}
#[test]
fn fresh_output_does_not_alias_input() {
    let x = tensor(&[1, 2], &[1., 2.]);
    let y = rope_reference(&x.view(), 0, config(2, 2, Pairing::Adjacent)).unwrap();
    assert_ne!(x.as_slice().as_ptr(), y.as_slice().as_ptr());
    assert_eq!(x.as_slice(), &[1., 2.]);
}
#[test]
fn broadcast_input_is_valid_for_immutable_reference() {
    let values = [1., 2., 3., 4.];
    let view = TensorView::try_from_parts(&values, vec![3, 4], vec![0, 1], 0).unwrap();
    let out = rope_reference(&view, 7, config(4, 4, Pairing::Adjacent)).unwrap();
    assert_eq!(&out.as_slice()[..4], &out.as_slice()[4..8]);
    assert_eq!(&out.as_slice()[..4], &out.as_slice()[8..]);
}
#[test]
fn f32_position_conversion_loses_adjacent_integers() {
    let p = 1_i64 << 24;
    assert_eq!(p as f32, (p + 1) as f32);
    assert_ne!(p as f64, (p + 1) as f64);
    let x = tensor(&[1, 2], &[1., 0.]);
    let c = config(2, 2, Pairing::Adjacent);
    assert_ne!(
        rope_reference(&x.view(), p, c).unwrap(),
        rope_reference(&x.view(), p + 1, c).unwrap()
    );
}
#[test]
fn same_position_preserves_cross_vector_dot_product() {
    let q = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let k = tensor(&[1, 4], &[2., -1., 1., 3.]);
    let c = config(4, 4, Pairing::Adjacent);
    assert!(
        (dot(q.as_slice(), k.as_slice())
            - dot(
                rope_reference(&q.view(), 7, c).unwrap().as_slice(),
                rope_reference(&k.view(), 7, c).unwrap().as_slice()
            ))
        .abs()
            < 4e-6
    );
}
#[test]
fn in_place_retains_payload_allocation_and_matches_reference() {
    for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
        let mut x = tensor(&[2, 4], &[1., 2., 3., 4., -2., 1., 0.5, -3.]);
        let ptr = x.as_slice().as_ptr();
        let expected = rope_reference(&x.view(), 7, config(4, 4, pairing)).unwrap();
        rope_in_place(&mut x, 7, config(4, 4, pairing)).unwrap();
        assert_eq!(ptr, x.as_slice().as_ptr());
        assert_eq!(x, expected);
    }
}
#[test]
fn invalid_dimensions_are_rejected() {
    for (d, r) in [(0, 0), (4, 0), (4, 3), (4, 6), (3, 3)] {
        assert!(matches!(
            RopeConfig::try_new(d, r, 100., Pairing::Adjacent),
            Err(RopeError::InvalidDimensions { .. })
        ));
    }
}
#[test]
fn base_domain_and_config_getters() {
    for base in [0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            RopeConfig::try_new(4, 4, base, Pairing::Adjacent),
            Err(RopeError::InvalidBase)
        );
    }
    for base in [0.5, 1., 10000.] {
        let c = RopeConfig::try_new(5, 4, base, Pairing::SplitHalf).unwrap();
        assert_eq!(
            (c.head_dim(), c.rotary_dim(), c.base(), c.pairing()),
            (5, 4, base, Pairing::SplitHalf)
        );
    }
}
#[test]
fn shape_failures_are_atomic() {
    for shape in [vec![4], vec![1, 1, 4], vec![0, 4], vec![2, 3]] {
        let mut x = OwnedTensor::zeros(shape).unwrap();
        let old = bits(&x);
        assert!(matches!(
            rope_in_place(&mut x, 1, config(4, 4, Pairing::Adjacent)),
            Err(RopeError::ShapeMismatch { .. })
        ));
        assert_eq!(bits(&x), old);
    }
}
#[test]
fn invalid_positions_are_atomic_including_integer_extremes() {
    for p in [
        MAX_EXACT_POSITION + 1,
        -MAX_EXACT_POSITION - 1,
        i64::MAX,
        i64::MIN,
    ] {
        let mut x = tensor(&[1, 2], &[1., 2.]);
        let old = bits(&x);
        assert_eq!(
            rope_in_place(&mut x, p, config(2, 2, Pairing::Adjacent)),
            Err(RopeError::PositionOutOfRange(p))
        );
        assert_eq!(bits(&x), old);
    }
}
#[test]
fn accepted_conversion_boundary_is_finite_not_accuracy_guarantee() {
    let x = tensor(&[1, 2], &[1., 2.]);
    for p in [-MAX_EXACT_POSITION, MAX_EXACT_POSITION] {
        assert!(
            rope_reference(&x.view(), p, config(2, 2, Pairing::Adjacent))
                .unwrap()
                .as_slice()
                .iter()
                .all(|x| x.is_finite())
        );
    }
}
#[test]
fn nonfinite_even_in_tail_or_zero_position_is_atomic() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for p in [0, 1] {
            let mut x = tensor(&[2, 3], &[1., 2., 3., 4., 5., value]);
            let old = bits(&x);
            assert_eq!(
                rope_in_place(&mut x, p, config(3, 2, Pairing::Adjacent)),
                Err(RopeError::NonFiniteInput {
                    head: 1,
                    coordinate: 2
                })
            );
            assert_eq!(bits(&x), old);
        }
    }
}
#[test]
fn late_output_overflow_cannot_partially_modify_earlier_heads() {
    let mut x = tensor(&[2, 4], &[1., 2., 3., 4., 1., 2., f32::MAX, f32::MAX]);
    let old = bits(&x);
    assert_eq!(
        rope_in_place(&mut x, 7, config(4, 4, Pairing::Adjacent)),
        Err(RopeError::NonFiniteOutput { head: 1, pair: 1 })
    );
    assert_eq!(bits(&x), old);
}
#[test]
fn phase_overflow_is_atomic() {
    let mut x = tensor(&[1, 128], &[1.; 128]);
    let old = bits(&x);
    let c = RopeConfig::try_new(128, 128, f64::from_bits(1), Pairing::Adjacent).unwrap();
    assert!(matches!(
        rope_in_place(&mut x, MAX_EXACT_POSITION, c),
        Err(RopeError::UnrepresentablePhase { .. })
    ));
    assert_eq!(bits(&x), old);
}
#[test]
fn width_two_has_no_base_sensitivity() {
    let x = tensor(&[1, 2], &[1., 2.]);
    let a = RopeConfig::try_new(2, 2, 10., Pairing::Adjacent).unwrap();
    let b = RopeConfig::try_new(2, 2, 1e6, Pairing::Adjacent).unwrap();
    assert_eq!(
        rope_reference(&x.view(), 7, a).unwrap(),
        rope_reference(&x.view(), 7, b).unwrap()
    );
}
#[test]
fn width_four_exposes_base_sensitivity_only_after_first_pair() {
    let x = tensor(&[1, 4], &[1., 2., 3., 4.]);
    let a = rope_reference(&x.view(), 7, config(4, 4, Pairing::Adjacent)).unwrap();
    let b = rope_reference(
        &x.view(),
        7,
        RopeConfig::try_new(4, 4, 10000., Pairing::Adjacent).unwrap(),
    )
    .unwrap();
    assert_eq!(&a.as_slice()[..2], &b.as_slice()[..2]);
    assert_ne!(&a.as_slice()[2..], &b.as_slice()[2..]);
}
#[test]
fn repeated_updates_accumulate_position_and_are_not_idempotent() {
    let mut x = tensor(&[1, 2], &[1., 2.]);
    let original = x.clone();
    let c = config(2, 2, Pairing::Adjacent);
    rope_in_place(&mut x, 7, c).unwrap();
    let once = x.clone();
    rope_in_place(&mut x, 7, c).unwrap();
    assert_ne!(x, once);
    close(
        x.as_slice(),
        rope_reference(&original.view(), 14, c).unwrap().as_slice(),
    );
}
