#[path = "../visual_support/chapter08.rs"]
mod fixture;
use engine0::qkv::{project_qkv_reference, QkvConfig, QkvError};
use engine0::tensor::{OwnedTensor, TensorView};

fn tensor(shape: &[usize], data: &[f32]) -> OwnedTensor {
    OwnedTensor::from_vec(shape.to_vec(), data.to_vec()).unwrap()
}

#[test]
fn hand_computed_nonidentity_projection() {
    let (c, x, [q, k, v]) = fixture::inputs();
    let p = project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()).unwrap();
    assert_eq!(p.query().as_slice(), &[7.0, 2.0, -2.0, -3.0]);
    assert_eq!(p.key().as_slice(), &[1.0, -3.0, 3.0, 3.5]);
    assert_eq!(p.value().as_slice(), &[-3.0, -1.0, 0.5, -2.0]);
}

#[test]
fn geometry_covers_mha_mqa_gqa_and_non_power_widths() {
    for (d, hq, hkv, dh) in [
        (1, 1, 1, 1),
        (6, 3, 1, 2),
        (8, 4, 2, 2),
        (8, 4, 4, 2),
        (3, 3, 1, 1),
    ] {
        let c = QkvConfig::try_new(d, hq, hkv, dh).unwrap();
        assert_eq!(c.model_dim(), d);
        assert_eq!(c.query_heads(), hq);
        assert_eq!(c.kv_heads(), hkv);
        assert_eq!(c.head_dim(), dh);
        assert_eq!(c.group_size(), hq / hkv);
        let x = tensor(&[d], &vec![1.0; d]);
        let q = tensor(&[hq * dh, d], &vec![2.0; hq * dh * d]);
        let kv = tensor(&[hkv * dh, d], &vec![-1.0; hkv * dh * d]);
        let p = project_qkv_reference(c, &x.view(), &q.view(), &kv.view(), &kv.view()).unwrap();
        assert_eq!(p.query().shape(), &[hq, dh]);
        assert_eq!(p.key().shape(), &[hkv, dh]);
        assert_eq!(p.value().shape(), &[hkv, dh]);
        assert!(p.query().as_slice().iter().all(|n| *n == 2.0 * d as f32));
        assert!(p.key().as_slice().iter().all(|n| *n == -(d as f32)));
    }
}

#[test]
fn each_zero_dimension_is_rejected() {
    for (args, name) in [
        ([0, 2, 2, 2], "model_dim"),
        ([4, 0, 2, 2], "query_heads"),
        ([4, 2, 0, 2], "kv_heads"),
        ([4, 2, 2, 0], "head_dim"),
    ] {
        assert_eq!(
            QkvConfig::try_new(args[0], args[1], args[2], args[3]),
            Err(QkvError::ZeroDimension(name))
        );
    }
}

#[test]
fn both_width_products_are_checked() {
    assert_eq!(
        QkvConfig::try_new(4, usize::MAX, 1, 2),
        Err(QkvError::WidthOverflow("query"))
    );
    assert_eq!(
        QkvConfig::try_new(4, 2, usize::MAX, 2),
        Err(QkvError::WidthOverflow("key/value"))
    );
}

#[test]
fn wrong_model_width_and_nondivisible_groups_are_rejected() {
    assert!(matches!(
        QkvConfig::try_new(5, 2, 1, 2),
        Err(QkvError::ModelWidthMismatch { .. })
    ));
    for (d, hq, hkv, dh) in [(6, 3, 2, 2), (4, 2, 4, 2)] {
        assert!(matches!(
            QkvConfig::try_new(d, hq, hkv, dh),
            Err(QkvError::NonDivisibleHeads { .. })
        ));
    }
}

#[test]
fn wrong_input_rank_or_width_is_rejected() {
    let (c, _, [q, k, v]) = fixture::inputs();
    for shape in [vec![1, 4], vec![3], vec![0]] {
        let x = OwnedTensor::zeros(shape.clone()).unwrap();
        assert_eq!(
            project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()),
            Err(QkvError::ShapeMismatch {
                operand: "input",
                expected: vec![4],
                actual: shape
            })
        );
    }
}

#[test]
fn every_weight_has_independent_rank_and_width_validation() {
    let (c, x, w) = fixture::inputs();
    for (index, name) in ["query_weight", "key_weight", "value_weight"]
        .iter()
        .enumerate()
    {
        for shape in [vec![16], vec![1, 4, 4], vec![3, 4], vec![4, 3], vec![0, 4]] {
            let mut changed = w.clone();
            changed[index] = OwnedTensor::zeros(shape.clone()).unwrap();
            assert_eq!(
                project_qkv_reference(
                    c,
                    &x.view(),
                    &changed[0].view(),
                    &changed[1].view(),
                    &changed[2].view()
                ),
                Err(QkvError::ShapeMismatch {
                    operand: name,
                    expected: vec![4, 4],
                    actual: shape
                })
            );
        }
    }
}

#[test]
fn gqa_rejects_accidentally_full_width_kv() {
    let (old, x, [q, k, v]) = fixture::inputs();
    let c = QkvConfig::try_new(old.model_dim(), 2, 1, 2).unwrap();
    assert!(matches!(
        project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()),
        Err(QkvError::ShapeMismatch {
            operand: "key_weight",
            ..
        })
    ));
}

#[test]
fn strided_input_and_independently_strided_weights_match() {
    let (c, x, w) = fixture::inputs();
    let input_storage = [99.0, 1.0, 99.0, -2.0, 99.0, 3.0, 99.0, -4.0];
    let xv = TensorView::try_from_parts(&input_storage, vec![4], vec![2], 1).unwrap();
    let storage: Vec<Vec<f32>> = w
        .iter()
        .enumerate()
        .map(|(i, weight)| {
            let step = i + 2;
            let mut data = vec![99.0; 1 + 16 * step];
            for (j, n) in weight.as_slice().iter().enumerate() {
                data[1 + j * step] = *n;
            }
            data
        })
        .collect();
    let views: Vec<_> = storage
        .iter()
        .enumerate()
        .map(|(i, data)| {
            TensorView::try_from_parts(data, vec![4, 4], vec![4 * (i + 2), i + 2], 1).unwrap()
        })
        .collect();
    let actual = project_qkv_reference(c, &xv, &views[0], &views[1], &views[2]).unwrap();
    let expected =
        project_qkv_reference(c, &x.view(), &w[0].view(), &w[1].view(), &w[2].view()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn immutable_broadcast_views_are_valid() {
    let c = QkvConfig::try_new(4, 2, 1, 2).unwrap();
    let x = TensorView::try_from_parts(&[2.0], vec![4], vec![0], 0).unwrap();
    let q = TensorView::try_from_parts(&[1.0], vec![4, 4], vec![0, 0], 0).unwrap();
    let kv = TensorView::try_from_parts(&[-1.0], vec![2, 4], vec![0, 0], 0).unwrap();
    let p = project_qkv_reference(c, &x, &q, &kv, &kv).unwrap();
    assert_eq!(p.query().as_slice(), &[8.0; 4]);
    assert_eq!(p.key().as_slice(), &[-8.0; 2]);
}

#[test]
fn all_head_coordinates_and_borrow_addresses_agree() {
    let c = QkvConfig::try_new(6, 3, 3, 2).unwrap();
    let x = tensor(&[6], &[1.0; 6]);
    let w = tensor(&[6, 6], &(0..36).map(|n| n as f32).collect::<Vec<_>>());
    let p = project_qkv_reference(c, &x.view(), &w.view(), &w.view(), &w.view()).unwrap();
    for (owner, head) in [
        (p.query(), 0),
        (p.query(), 1),
        (p.query(), 2),
        (p.key(), 2),
        (p.value(), 1),
    ] {
        let view = if std::ptr::eq(owner, p.query()) {
            p.query_head(head)
        } else if std::ptr::eq(owner, p.key()) {
            p.key_head(head)
        } else {
            p.value_head(head)
        }
        .unwrap();
        assert_eq!(view.shape(), &[2]);
        assert_eq!(view.base_offset(), head * 2);
        for j in 0..2 {
            assert!(std::ptr::eq(
                view.get(&[j]).unwrap(),
                &owner.as_slice()[head * 2 + j]
            ));
        }
    }
}

#[test]
fn head_bounds_reject_equal_count_and_maximum_index() {
    let (c, x, [q, k, v]) = fixture::inputs();
    let p = project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()).unwrap();
    for h in [2, usize::MAX] {
        assert!(matches!(
            p.query_head(h),
            Err(QkvError::HeadOutOfRange {
                operand: "query",
                ..
            })
        ));
        assert!(matches!(
            p.key_head(h),
            Err(QkvError::HeadOutOfRange { operand: "key", .. })
        ));
        assert!(matches!(
            p.value_head(h),
            Err(QkvError::HeadOutOfRange {
                operand: "value",
                ..
            })
        ));
    }
}

#[test]
fn output_ownership_is_disjoint_and_transfer_does_not_copy() {
    let (c, x, [q, k, v]) = fixture::inputs();
    let p = project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()).unwrap();
    let pointer = p.query().as_slice().as_ptr();
    let (mut out, kout, vout) = p.into_parts();
    assert_eq!(pointer, out.as_slice().as_ptr());
    *out.view_mut().get_mut(&[0, 0]).unwrap() = 99.0;
    assert_eq!(x.as_slice(), &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(q.as_slice()[0], 1.0);
    assert_eq!(kout.as_slice()[0], 1.0);
    assert_eq!(vout.as_slice()[0], -3.0);
}

#[test]
fn raw_projection_has_no_position_argument_or_hidden_state() {
    let (c, x, [q, k, v]) = fixture::inputs();
    let at_zero = project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()).unwrap();
    let at_seven = project_qkv_reference(c, &x.view(), &q.view(), &k.view(), &v.view()).unwrap();
    assert_eq!(at_zero, at_seven);
    assert_ne!(
        at_zero.query().as_slice().as_ptr(),
        at_seven.query().as_slice().as_ptr()
    );
}

#[test]
fn zero_weights_produce_zero_and_nonfinite_values_follow_gemv() {
    let c = QkvConfig::try_new(1, 1, 1, 1).unwrap();
    let zero = tensor(&[1, 1], &[0.0]);
    let x = tensor(&[1], &[3.0]);
    assert_eq!(
        project_qkv_reference(c, &x.view(), &zero.view(), &zero.view(), &zero.view())
            .unwrap()
            .query()
            .as_slice(),
        &[0.0]
    );
    let huge = tensor(&[1, 1], &[f32::MAX]);
    let p = project_qkv_reference(c, &x.view(), &huge.view(), &zero.view(), &zero.view()).unwrap();
    assert!(p.query().as_slice()[0].is_infinite());
    let nan = tensor(&[1], &[f32::NAN]);
    assert!(
        project_qkv_reference(c, &nan.view(), &zero.view(), &zero.view(), &zero.view())
            .unwrap()
            .query()
            .as_slice()[0]
            .is_nan()
    );
}
