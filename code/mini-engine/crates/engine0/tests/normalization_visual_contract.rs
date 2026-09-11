#[path = "../visual_support/chapter07.rs"]
mod fixture;
use engine0::embedding::{embedding_lookup_reference, embedding_sequence_reference};
use engine0::normalization::{rms_norm_reference, NormalizationError};
use engine0::tensor::OwnedTensor;
use engine0::tokenizer::TokenId;

#[test]
fn selected_row_is_the_existing_hand_vector() {
    let (table, weight, epsilon) = fixture::inputs();
    let x = embedding_lookup_reference(&table.view(), TokenId(1)).unwrap();
    assert_eq!(x.as_slice(), &[1., -2., 3., -4.]);
    let y = rms_norm_reference(&x.view(), &weight.view(), epsilon).unwrap();
    for (a, b) in y
        .as_slice()
        .iter()
        .zip([0.36514813, -0.36514813, 2.190_889, 1.4605925])
    {
        assert!((a - b).abs() < 2e-6);
    }
}

#[test]
fn repeated_tokens_have_independent_activation_cells() {
    let (table, _, _) = fixture::inputs();
    let mut output =
        embedding_sequence_reference(&table.view(), &[TokenId(1), TokenId(0), TokenId(1)]).unwrap();
    *output.view_mut().get_mut(&[0, 0]).unwrap() = 99.;
    assert_eq!(*output.get2(2, 0).unwrap(), 1.);
    assert_eq!(*table.get2(1, 0).unwrap(), 1.);
}

#[test]
fn zero_row_stays_zero_with_nonunit_gain() {
    let (table, weight, epsilon) = fixture::inputs();
    let x = embedding_lookup_reference(&table.view(), TokenId(0)).unwrap();
    assert_eq!(
        rms_norm_reference(&x.view(), &weight.view(), epsilon)
            .unwrap()
            .as_slice(),
        &[0.; 4]
    );
}

#[test]
fn output_is_not_required_to_have_unit_rms() {
    let (table, weight, epsilon) = fixture::inputs();
    let x = embedding_lookup_reference(&table.view(), TokenId(1)).unwrap();
    let y = rms_norm_reference(&x.view(), &weight.view(), epsilon).unwrap();
    let rms = (y.as_slice().iter().map(|n| n * n).sum::<f32>() / 4.).sqrt();
    assert!(rms > 1.3);
}

#[test]
fn finite_squares_can_overflow_the_reduction() {
    let (_, weight, epsilon) = fixture::inputs();
    let x = OwnedTensor::from_vec(vec![4], vec![1e19; 4]).unwrap();
    assert!(matches!(
        rms_norm_reference(&x.view(), &weight.view(), epsilon),
        Err(NormalizationError::NonFiniteReduction { .. })
    ));
}
