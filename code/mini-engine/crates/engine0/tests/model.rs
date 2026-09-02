use engine0::model::{Model, ModelError, TinyLanguageModel};
use engine0::tokenizer::{TokenId, TINY_LM_EOS, TINY_LM_I, TINY_LM_LIKE, TINY_LM_RUST};

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs()
}

fn assert_close_vector(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close(*actual, *expected),
            "logit {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn construction_accepts_valid_shapes_and_counts_bytes() {
    let model = TinyLanguageModel::chapter3_fixture();
    assert_eq!(model.vocabulary_size(), 4);
    assert_eq!(model.hidden_dim(), 3);
    assert_eq!(model.parameter_count(), 28);
    assert_eq!(model.parameter_bytes(), 112);
}

#[test]
fn zero_vocabulary_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(0, 2, vec![], vec![], vec![]),
        Err(ModelError::InvalidDimensions { .. })
    ));
}

#[test]
fn zero_hidden_dimension_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(2, 0, vec![], vec![], vec![0.0; 2]),
        Err(ModelError::InvalidDimensions { .. })
    ));
}

#[test]
fn dimension_multiplication_overflow_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(usize::MAX, 2, vec![], vec![], vec![]),
        Err(ModelError::DimensionOverflow { .. })
    ));
}

#[test]
fn embedding_shape_mismatch_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(2, 2, vec![0.0; 3], vec![0.0; 4], vec![0.0; 2]),
        Err(ModelError::InvalidParameterCount {
            name: "embedding",
            ..
        })
    ));
}

#[test]
fn projection_shape_mismatch_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(2, 2, vec![0.0; 4], vec![0.0; 3], vec![0.0; 2]),
        Err(ModelError::InvalidParameterCount {
            name: "output projection",
            ..
        })
    ));
}

#[test]
fn bias_shape_mismatch_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(2, 2, vec![0.0; 4], vec![0.0; 4], vec![0.0; 1]),
        Err(ModelError::InvalidParameterCount {
            name: "output bias",
            ..
        })
    ));
}

#[test]
fn non_finite_parameter_is_rejected() {
    assert!(matches!(
        TinyLanguageModel::try_new(1, 1, vec![f32::NAN], vec![0.0], vec![0.0]),
        Err(ModelError::NonFiniteParameter { .. })
    ));
}

#[test]
fn finite_parameters_that_overflow_produce_a_typed_logit_error() {
    let model = TinyLanguageModel::try_new(1, 1, vec![2.0], vec![f32::MAX], vec![0.0]).unwrap();
    assert!(matches!(
        model.forward(&[TokenId(0)]),
        Err(ModelError::NonFiniteLogit { index: 0, .. })
    ));
}

#[test]
fn empty_sequence_is_rejected() {
    assert_eq!(
        TinyLanguageModel::chapter3_fixture().forward(&[]),
        Err(ModelError::EmptyInput)
    );
}

#[test]
fn token_out_of_range_is_rejected() {
    assert_eq!(
        TinyLanguageModel::chapter3_fixture().forward(&[TokenId(4)]),
        Err(ModelError::TokenOutOfRange {
            token: TokenId(4),
            vocab_size: 4
        })
    );
}

#[test]
fn embedding_lookup_selects_exact_row() {
    let pass = TinyLanguageModel::chapter3_fixture()
        .forward(&[TINY_LM_LIKE])
        .unwrap();
    assert_eq!(pass.input_token, TINY_LM_LIKE);
    assert_eq!(pass.hidden, vec![1.0, -0.5, 2.0]);
}

#[test]
fn full_logits_match_independent_hand_oracle() {
    let pass = TinyLanguageModel::chapter3_fixture()
        .forward(&[TINY_LM_LIKE])
        .unwrap();
    assert_close_vector(pass.logits.as_slice(), &[-0.7, 0.1, 0.4, 2.2]);
}

#[test]
fn bias_is_applied_to_zero_embedding() {
    let pass = TinyLanguageModel::chapter3_fixture()
        .forward(&[TINY_LM_EOS])
        .unwrap();
    assert_close_vector(pass.logits.as_slice(), &[-0.2, 0.0, 0.0, 0.5]);
}

#[test]
fn negative_and_zero_values_survive_forward_arithmetic() {
    let model = TinyLanguageModel::try_new(
        2,
        2,
        vec![-2.0, 0.0, 0.0, 3.0],
        vec![1.0, -1.0, 0.0, 0.0],
        vec![0.0, -0.5],
    )
    .unwrap();
    let pass = model.forward(&[TokenId(0)]).unwrap();
    assert_close_vector(pass.logits.as_slice(), &[-2.0, -0.5]);
}

#[test]
fn repeatability_covers_the_entire_forward_result() {
    let model = TinyLanguageModel::chapter3_fixture();
    assert_eq!(
        model.forward(&[TINY_LM_LIKE]),
        model.forward(&[TINY_LM_LIKE])
    );
}

#[test]
fn same_last_token_produces_same_logits_despite_different_history() {
    let model = TinyLanguageModel::chapter3_fixture();
    let a = model.forward(&[TINY_LM_I, TINY_LM_LIKE]).unwrap();
    let b = model
        .forward(&[TINY_LM_RUST, TINY_LM_EOS, TINY_LM_LIKE])
        .unwrap();
    assert_eq!(a.logits, b.logits);
    assert_eq!(a.hidden, b.hidden);
}

#[test]
fn different_embedding_rows_produce_expected_different_logits() {
    let model = TinyLanguageModel::chapter3_fixture();
    let like = model.forward(&[TINY_LM_LIKE]).unwrap();
    let rust = model.forward(&[TINY_LM_RUST]).unwrap();
    assert_ne!(like.hidden, rust.hidden);
    assert_close_vector(rust.logits.as_slice(), &[0.3, -0.2, -0.3, -0.5]);
}

#[test]
fn eos_has_a_regular_logit_position() {
    let pass = TinyLanguageModel::chapter3_fixture()
        .forward(&[TINY_LM_RUST])
        .unwrap();
    assert_eq!(pass.logits.len(), 4);
    assert!(close(pass.logits.as_slice()[TINY_LM_EOS.0 as usize], 0.3));
}

#[test]
fn changing_one_projection_weight_changes_only_its_output_row() {
    let base = TinyLanguageModel::chapter3_fixture()
        .forward(&[TINY_LM_LIKE])
        .unwrap();
    let changed = TinyLanguageModel::try_new(
        4,
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -0.5, 2.0, -1.0, 0.0, 0.0],
        vec![
            -0.5, 0.4, 0.1, 0.2, 0.2, 0.0, 0.3, 0.2, 0.1, 1.5, -0.4, 0.25,
        ],
        vec![-0.2, 0.0, 0.0, 0.5],
    )
    .unwrap()
    .forward(&[TINY_LM_LIKE])
    .unwrap();

    assert_close_vector(
        &changed.logits.as_slice()[..3],
        &base.logits.as_slice()[..3],
    );
    assert!(close(changed.logits.as_slice()[3], 2.7));
}
