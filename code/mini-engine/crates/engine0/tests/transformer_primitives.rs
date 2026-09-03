use engine0::embedding::{
    embedding_lookup_reference, embedding_sequence_reference, EmbeddingError,
};
use engine0::normalization::{rms_norm_reference, NormalizationError};
use engine0::tensor::{OwnedTensor, TensorView};
use engine0::tokenizer::TokenId;

fn tensor(shape: &[usize], values: &[f32]) -> OwnedTensor {
    OwnedTensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-5_f32 + 1.0e-5_f32 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "element {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn embedding_selects_first_middle_and_last_rows() {
    let table = tensor(
        &[4, 3],
        &[
            0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0,
        ],
    );
    for (token, expected) in [
        (TokenId(0), &[0.0, 1.0, 2.0][..]),
        (TokenId(2), &[20.0, 21.0, 22.0][..]),
        (TokenId(3), &[30.0, 31.0, 32.0][..]),
    ] {
        assert_eq!(
            embedding_lookup_reference(&table.view(), token)
                .unwrap()
                .as_slice(),
            expected
        );
    }
}

#[test]
fn embedding_supports_model_dimension_one() {
    let table = tensor(&[3, 1], &[2.0, 4.0, 8.0]);
    let output = embedding_lookup_reference(&table.view(), TokenId(1)).unwrap();
    assert_eq!(output.shape(), &[1]);
    assert_eq!(output.as_slice(), &[4.0]);
}

#[test]
fn embedding_rejects_wrong_rank() {
    let vector = tensor(&[3], &[1.0, 2.0, 3.0]);
    assert_eq!(
        embedding_lookup_reference(&vector.view(), TokenId(0)),
        Err(EmbeddingError::RankMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn embedding_rejects_empty_vocabulary() {
    let table = tensor(&[0, 3], &[]);
    assert_eq!(
        embedding_lookup_reference(&table.view(), TokenId(0)),
        Err(EmbeddingError::EmptyVocabulary)
    );
}

#[test]
fn embedding_rejects_empty_model_dimension() {
    let table = tensor(&[3, 0], &[]);
    assert_eq!(
        embedding_lookup_reference(&table.view(), TokenId(0)),
        Err(EmbeddingError::EmptyModelDimension)
    );
}

#[test]
fn embedding_rejects_token_equal_to_or_above_vocabulary() {
    let table = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    for token in [TokenId(2), TokenId(u32::MAX)] {
        assert_eq!(
            embedding_lookup_reference(&table.view(), token),
            Err(EmbeddingError::TokenOutOfRange {
                token,
                vocab_size: 2,
            })
        );
    }
}

#[test]
fn embedding_reads_a_strided_table_logically() {
    let storage = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0, 5.0, 99.0, 6.0];
    let table = TensorView::try_from_parts(&storage, vec![3, 2], vec![4, 2], 0).unwrap();
    assert_eq!(
        embedding_lookup_reference(&table, TokenId(1))
            .unwrap()
            .as_slice(),
        &[3.0, 4.0]
    );
}

#[test]
fn embedding_defines_zero_stride_table_semantics() {
    let storage = [7.0];
    let table = TensorView::try_from_parts(&storage, vec![4, 3], vec![0, 0], 0).unwrap();
    assert_eq!(
        embedding_lookup_reference(&table, TokenId(3))
            .unwrap()
            .as_slice(),
        &[7.0, 7.0, 7.0]
    );
}

#[test]
fn embedding_result_owns_storage_independent_of_parameters() {
    let table = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut activation = embedding_lookup_reference(&table.view(), TokenId(1)).unwrap();
    *activation.view_mut().get_mut(&[0]).unwrap() = -100.0;
    assert_eq!(activation.as_slice(), &[-100.0, 4.0]);
    assert_eq!(table.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn sequence_embedding_preserves_token_order_and_repetition() {
    let table = tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let output =
        embedding_sequence_reference(&table.view(), &[TokenId(2), TokenId(0), TokenId(2)]).unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.strides(), &[2, 1]);
    assert_eq!(output.as_slice(), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn empty_token_sequence_produces_an_empty_owned_matrix() {
    let table = tensor(&[2, 3], &[1.0; 6]);
    let output = embedding_sequence_reference(&table.view(), &[]).unwrap();
    assert_eq!(output.shape(), &[0, 3]);
    assert!(output.is_empty());
}

#[test]
fn sequence_embedding_stops_on_an_invalid_token() {
    let table = tensor(&[2, 2], &[1.0; 4]);
    assert_eq!(
        embedding_sequence_reference(&table.view(), &[TokenId(0), TokenId(2)]),
        Err(EmbeddingError::TokenOutOfRange {
            token: TokenId(2),
            vocab_size: 2,
        })
    );
}

#[test]
fn rmsnorm_dimension_one_matches_its_formula() {
    let input = tensor(&[1], &[-3.0]);
    let weight = tensor(&[1], &[2.0]);
    let expected = -3.0 / (9.0_f32 + 1.0e-5).sqrt() * 2.0;
    assert_close(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5)
            .unwrap()
            .as_slice(),
        &[expected],
    );
}

#[test]
fn rmsnorm_matches_the_hand_calculated_mixed_sign_example() {
    let input = tensor(&[4], &[1.0, -2.0, 3.0, -4.0]);
    let weight = tensor(&[4], &[1.0, 0.5, 2.0, -1.0]);
    let inverse_rms = 1.0_f32 / (7.5_f32 + 1.0e-5).sqrt();
    let expected = [
        1.0 * inverse_rms,
        -inverse_rms,
        6.0 * inverse_rms,
        4.0 * inverse_rms,
    ];
    assert_close(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5)
            .unwrap()
            .as_slice(),
        &expected,
    );
}

#[test]
fn rmsnorm_zero_vector_is_finite_and_zero() {
    let input = tensor(&[4], &[0.0; 4]);
    let weight = tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5)
            .unwrap()
            .as_slice(),
        &[0.0; 4]
    );
}

#[test]
fn rmsnorm_uniform_vector_has_unit_rms_when_epsilon_is_negligible() {
    let input = tensor(&[8], &[4.0; 8]);
    let weight = tensor(&[8], &[1.0; 8]);
    let output = rms_norm_reference(&input.view(), &weight.view(), 1.0e-8).unwrap();
    assert_close(output.as_slice(), &[1.0; 8]);
}

#[test]
fn rmsnorm_applies_non_unit_learned_weights_elementwise() {
    let input = tensor(&[3], &[1.0, 1.0, 1.0]);
    let weight = tensor(&[3], &[0.5, 1.0, 2.0]);
    let output = rms_norm_reference(&input.view(), &weight.view(), 1.0e-8).unwrap();
    assert_close(output.as_slice(), &[0.5, 1.0, 2.0]);
}

#[test]
fn rmsnorm_accepts_strided_input_and_weight() {
    let input_storage = [1.0, 99.0, -2.0, 99.0, 3.0];
    let weight_storage = [2.0, 99.0, 0.5, 99.0, -1.0];
    let input = TensorView::try_from_parts(&input_storage, vec![3], vec![2], 0).unwrap();
    let weight = TensorView::try_from_parts(&weight_storage, vec![3], vec![2], 0).unwrap();
    let scale = 1.0_f32 / ((14.0_f32 / 3.0) + 1.0e-5).sqrt();
    assert_close(
        rms_norm_reference(&input, &weight, 1.0e-5)
            .unwrap()
            .as_slice(),
        &[2.0 * scale, -scale, -3.0 * scale],
    );
}

#[test]
fn rmsnorm_defines_zero_stride_input_and_weight() {
    let input = TensorView::try_from_parts(&[2.0], vec![4], vec![0], 0).unwrap();
    let weight = TensorView::try_from_parts(&[3.0], vec![4], vec![0], 0).unwrap();
    assert_close(
        rms_norm_reference(&input, &weight, 1.0e-8)
            .unwrap()
            .as_slice(),
        &[3.0; 4],
    );
}

#[test]
fn rmsnorm_rejects_wrong_input_and_weight_ranks() {
    let vector = tensor(&[2], &[1.0, 2.0]);
    let matrix = tensor(&[1, 2], &[1.0, 2.0]);
    assert!(matches!(
        rms_norm_reference(&matrix.view(), &vector.view(), 1.0e-5),
        Err(NormalizationError::RankMismatch {
            operand: "input",
            ..
        })
    ));
    assert!(matches!(
        rms_norm_reference(&vector.view(), &matrix.view(), 1.0e-5),
        Err(NormalizationError::RankMismatch {
            operand: "weight",
            ..
        })
    ));
}

#[test]
fn rmsnorm_rejects_length_mismatch() {
    let input = tensor(&[2], &[1.0, 2.0]);
    let weight = tensor(&[3], &[1.0; 3]);
    assert_eq!(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5),
        Err(NormalizationError::LengthMismatch {
            input: 2,
            weight: 3,
        })
    );
}

#[test]
fn rmsnorm_rejects_empty_dimension() {
    let empty = tensor(&[0], &[]);
    assert_eq!(
        rms_norm_reference(&empty.view(), &empty.view(), 1.0e-5),
        Err(NormalizationError::EmptyDimension)
    );
}

#[test]
fn rmsnorm_rejects_every_nonpositive_or_nonfinite_epsilon() {
    let input = tensor(&[1], &[1.0]);
    let weight = tensor(&[1], &[1.0]);
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            rms_norm_reference(&input.view(), &weight.view(), epsilon),
            Err(NormalizationError::InvalidEpsilon { .. })
        ));
    }
}

#[test]
fn rmsnorm_rejects_nonfinite_input_and_weight_values() {
    let finite = tensor(&[2], &[1.0, 2.0]);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = tensor(&[2], &[1.0, bad]);
        assert!(matches!(
            rms_norm_reference(&input.view(), &finite.view(), 1.0e-5),
            Err(NormalizationError::NonFiniteValue {
                operand: "input",
                index: 1,
                ..
            })
        ));

        let weight = tensor(&[2], &[1.0, bad]);
        assert!(matches!(
            rms_norm_reference(&finite.view(), &weight.view(), 1.0e-5),
            Err(NormalizationError::NonFiniteValue {
                operand: "weight",
                index: 1,
                ..
            })
        ));
    }
}

#[test]
fn rmsnorm_handles_large_values_whose_squares_remain_finite() {
    let input = tensor(&[2], &[1.0e10, -1.0e10]);
    let weight = tensor(&[2], &[1.0; 2]);
    assert_close(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5)
            .unwrap()
            .as_slice(),
        &[1.0, -1.0],
    );
}

#[test]
fn rmsnorm_reports_f32_square_overflow() {
    let input = tensor(&[1], &[1.0e20]);
    let weight = tensor(&[1], &[1.0]);
    assert_eq!(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5),
        Err(NormalizationError::NonFiniteSquare {
            index: 0,
            value: 1.0e20,
        })
    );
}

#[test]
fn rmsnorm_reports_f32_reduction_overflow() {
    let input = tensor(&[4], &[1.0e19; 4]);
    let weight = tensor(&[4], &[1.0; 4]);
    assert!(matches!(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5),
        Err(NormalizationError::NonFiniteReduction { .. })
    ));
}

#[test]
fn rmsnorm_exposes_small_square_underflow_and_epsilon_dominance() {
    let input = tensor(&[2], &[1.0e-30, -1.0e-30]);
    let weight = tensor(&[2], &[1.0; 2]);
    let output = rms_norm_reference(&input.view(), &weight.view(), 1.0e-6).unwrap();
    assert_close(output.as_slice(), &[1.0e-27, -1.0e-27]);
}

#[test]
fn rmsnorm_reports_nonfinite_output_from_extreme_finite_weight() {
    let input = tensor(&[2], &[1.0, 0.0]);
    let weight = tensor(&[2], &[f32::MAX, 1.0]);
    assert!(matches!(
        rms_norm_reference(&input.view(), &weight.view(), 1.0e-5),
        Err(NormalizationError::NonFiniteOutput { index: 0, .. })
    ));
}

#[test]
fn rmsnorm_is_approximately_scale_invariant_when_epsilon_is_negligible() {
    let weight = tensor(&[4], &[1.0, 0.5, 2.0, -1.0]);
    let base = [1.0_f32, -2.0, 3.0, -4.0];
    let baseline = rms_norm_reference(&tensor(&[4], &base).view(), &weight.view(), 1.0e-6).unwrap();
    for factor in [0.1_f32, 10.0, 100.0] {
        let scaled: Vec<f32> = base.iter().map(|value| value * factor).collect();
        let output =
            rms_norm_reference(&tensor(&[4], &scaled).view(), &weight.view(), 1.0e-6).unwrap();
        assert_close(output.as_slice(), baseline.as_slice());
    }
}
