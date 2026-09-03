use engine0::linear::{
    dot_reference, gemv_reference, matmul_blocked, matmul_reference, BlockSize, KernelError,
};
use engine0::tensor::{OwnedTensor, TensorView};

fn tensor(shape: &[usize], values: &[f32]) -> OwnedTensor {
    OwnedTensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-5_f32 + 1.0e-5_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn dot_matches_a_hand_computed_example() {
    let left = tensor(&[3], &[1.0, 2.0, 3.0]);
    let right = tensor(&[3], &[4.0, 5.0, 6.0]);
    assert_eq!(dot_reference(&left.view(), &right.view()).unwrap(), 32.0);
}

#[test]
fn dot_accepts_strided_and_zero_stride_views() {
    let storage = [2.0, 99.0, 3.0, 99.0, 4.0];
    let strided = TensorView::try_from_parts(&storage, vec![3], vec![2], 0).unwrap();
    let broadcast = TensorView::try_from_parts(&[2.0], vec![3], vec![0], 0).unwrap();
    assert_eq!(dot_reference(&strided, &broadcast).unwrap(), 18.0);
}

#[test]
fn dot_of_empty_vectors_is_the_additive_identity() {
    let empty = tensor(&[0], &[]);
    assert_eq!(dot_reference(&empty.view(), &empty.view()).unwrap(), 0.0);
}

#[test]
fn dot_rejects_rank_and_length_mismatches() {
    let vector = tensor(&[2], &[1.0, 2.0]);
    let matrix = tensor(&[1, 2], &[1.0, 2.0]);
    assert!(matches!(
        dot_reference(&matrix.view(), &vector.view()),
        Err(KernelError::RankMismatch {
            operand: "left",
            ..
        })
    ));
    let longer = tensor(&[3], &[1.0, 2.0, 3.0]);
    assert_eq!(
        dot_reference(&vector.view(), &longer.view()),
        Err(KernelError::LengthMismatch {
            operation: "dot",
            left: 2,
            right: 3,
        })
    );
}

#[test]
fn gemv_matches_a_hand_computed_example() {
    let matrix = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let vector = tensor(&[3], &[2.0, -1.0, 0.5]);
    let output = gemv_reference(&matrix.view(), &vector.view()).unwrap();
    assert_eq!(output.shape(), &[2]);
    assert_close(output.as_slice(), &[1.5, 6.0]);
}

#[test]
fn gemv_reads_strided_matrix_and_vector_logically() {
    let matrix_storage = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0, 5.0, 99.0, 6.0];
    let vector_storage = [1.0, 99.0, 2.0, 99.0, 3.0];
    let matrix = TensorView::try_from_parts(&matrix_storage, vec![2, 3], vec![6, 2], 0).unwrap();
    let vector = TensorView::try_from_parts(&vector_storage, vec![3], vec![2], 0).unwrap();
    assert_eq!(
        gemv_reference(&matrix, &vector).unwrap().as_slice(),
        &[14.0, 32.0]
    );
}

#[test]
fn gemv_defines_zero_inner_and_zero_row_shapes() {
    let matrix = tensor(&[3, 0], &[]);
    let vector = tensor(&[0], &[]);
    assert_eq!(
        gemv_reference(&matrix.view(), &vector.view())
            .unwrap()
            .as_slice(),
        &[0.0, 0.0, 0.0]
    );

    let no_rows = tensor(&[0, 4], &[]);
    let vector = tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let output = gemv_reference(&no_rows.view(), &vector.view()).unwrap();
    assert_eq!(output.shape(), &[0]);
    assert!(output.is_empty());
}

#[test]
fn gemv_rejects_bad_ranks_and_inner_dimensions() {
    let matrix = tensor(&[2, 2], &[1.0; 4]);
    let vector = tensor(&[3], &[1.0; 3]);
    assert!(matches!(
        gemv_reference(&matrix.view(), &matrix.view()),
        Err(KernelError::RankMismatch {
            operand: "vector",
            ..
        })
    ));
    assert_eq!(
        gemv_reference(&matrix.view(), &vector.view()),
        Err(KernelError::InnerDimensionMismatch {
            operation: "gemv",
            left: 2,
            right: 3,
        })
    );
}

#[test]
fn reference_matmul_matches_a_hand_computed_rectangle() {
    let left = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let right = tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let output = matmul_reference(&left.view(), &right.view()).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_close(output.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn reference_matmul_handles_fractional_asymmetric_values() {
    let left = tensor(&[1, 3], &[0.25, -1.5, 2.0]);
    let right = tensor(&[3, 2], &[4.0, -2.0, 0.5, 3.0, -1.0, 0.25]);
    let output = matmul_reference(&left.view(), &right.view()).unwrap();
    assert_close(output.as_slice(), &[-1.75, -4.5]);
}

#[test]
fn reference_matmul_accepts_a_transposed_left_view() {
    let storage = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transposed = storage.view().transpose().unwrap();
    let right = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let output = matmul_reference(&transposed, &right.view()).unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    assert_close(output.as_slice(), &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
}

#[test]
fn reference_matmul_accepts_zero_stride_broadcast_views() {
    let left = TensorView::try_from_parts(&[2.0], vec![2, 3], vec![0, 0], 0).unwrap();
    let right = tensor(&[3, 1], &[1.0, 2.0, 3.0]);
    assert_eq!(
        matmul_reference(&left, &right.view()).unwrap().as_slice(),
        &[12.0, 12.0]
    );
}

#[test]
fn reference_matmul_defines_all_zero_dimension_cases() {
    let a = tensor(&[2, 0], &[]);
    let b = tensor(&[0, 3], &[]);
    let output = matmul_reference(&a.view(), &b.view()).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.as_slice(), &[0.0; 6]);

    let a = tensor(&[0, 4], &[]);
    let b = tensor(&[4, 3], &[1.0; 12]);
    assert_eq!(
        matmul_reference(&a.view(), &b.view()).unwrap().shape(),
        &[0, 3]
    );

    let a = tensor(&[2, 4], &[1.0; 8]);
    let b = tensor(&[4, 0], &[]);
    assert_eq!(
        matmul_reference(&a.view(), &b.view()).unwrap().shape(),
        &[2, 0]
    );
}

#[test]
fn reference_matmul_rejects_bad_ranks_and_inner_dimensions() {
    let vector = tensor(&[2], &[1.0, 2.0]);
    let matrix = tensor(&[2, 2], &[1.0; 4]);
    assert!(matches!(
        matmul_reference(&vector.view(), &matrix.view()),
        Err(KernelError::RankMismatch {
            operand: "left",
            ..
        })
    ));
    let wide = tensor(&[3, 1], &[1.0; 3]);
    assert_eq!(
        matmul_reference(&matrix.view(), &wide.view()),
        Err(KernelError::InnerDimensionMismatch {
            operation: "matmul_reference",
            left: 2,
            right: 3,
        })
    );
}

#[test]
fn output_shape_overflow_is_reported_before_allocation() {
    let empty: [f32; 0] = [];
    let left = TensorView::try_from_parts(&empty, vec![usize::MAX, 0], vec![0, 1], 0).unwrap();
    let right =
        TensorView::try_from_parts(&empty, vec![0, usize::MAX], vec![usize::MAX, 1], 0).unwrap();
    assert_eq!(
        matmul_reference(&left, &right),
        Err(KernelError::OutputShapeOverflow {
            rows: usize::MAX,
            columns: usize::MAX,
        })
    );
}

#[test]
fn blocked_matmul_matches_the_hand_computed_rectangle() {
    let left = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let right = tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let output = matmul_blocked(
        &left.view(),
        &right.view(),
        BlockSize::try_new(2, 2, 2).unwrap(),
    )
    .unwrap();
    assert_close(output.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn blocked_matmul_covers_three_axis_tile_tails() {
    let left_values: Vec<f32> = (0..35).map(|x| (x as f32 - 11.0) / 7.0).collect();
    let right_values: Vec<f32> = (0..21).map(|x| (13.0 - x as f32) / 5.0).collect();
    let left = tensor(&[5, 7], &left_values);
    let right = tensor(&[7, 3], &right_values);
    let reference = matmul_reference(&left.view(), &right.view()).unwrap();
    let blocked = matmul_blocked(
        &left.view(),
        &right.view(),
        BlockSize::try_new(4, 4, 2).unwrap(),
    )
    .unwrap();
    assert_close(blocked.as_slice(), reference.as_slice());
}

#[test]
fn a_tile_larger_than_the_problem_is_valid() {
    let left = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let right = tensor(&[2, 1], &[5.0, 6.0]);
    let output = matmul_blocked(
        &left.view(),
        &right.view(),
        BlockSize::try_new(64, 64, 64).unwrap(),
    )
    .unwrap();
    assert_eq!(output.as_slice(), &[17.0, 39.0]);
}

#[test]
fn blocked_matmul_rejects_noncanonical_left_and_right_without_copying() {
    let packed = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let transposed = packed.view().transpose().unwrap();
    assert_eq!(
        matmul_blocked(&transposed, &packed.view(), BlockSize::DEFAULT),
        Err(KernelError::UnsupportedLayout {
            operation: "matmul_blocked",
            operand: "left",
        })
    );
    assert_eq!(
        matmul_blocked(&packed.view(), &transposed, BlockSize::DEFAULT),
        Err(KernelError::UnsupportedLayout {
            operation: "matmul_blocked",
            operand: "right",
        })
    );
}

#[test]
fn block_dimensions_must_all_be_positive() {
    for dimensions in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert!(matches!(
            BlockSize::try_new(dimensions.0, dimensions.1, dimensions.2),
            Err(KernelError::InvalidBlockSize { .. })
        ));
    }
    let invalid = BlockSize { m: 8, k: 0, n: 8 };
    let empty = tensor(&[0, 0], &[]);
    assert!(matches!(
        matmul_blocked(&empty.view(), &empty.view(), invalid),
        Err(KernelError::InvalidBlockSize { .. })
    ));
}

#[test]
fn blocked_matmul_preserves_zero_dimension_semantics() {
    let left = tensor(&[2, 0], &[]);
    let right = tensor(&[0, 3], &[]);
    let output = matmul_blocked(&left.view(), &right.view(), BlockSize::DEFAULT).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.as_slice(), &[0.0; 6]);
}

#[test]
fn blocked_and_reference_agree_over_a_deterministic_shape_grid() {
    let blocks = [
        BlockSize::try_new(1, 1, 1).unwrap(),
        BlockSize::try_new(2, 3, 4).unwrap(),
        BlockSize::try_new(5, 4, 3).unwrap(),
    ];
    for rows in 0..=6 {
        for inner in 0..=6 {
            for columns in 0..=6 {
                let left_values: Vec<f32> = (0..rows * inner)
                    .map(|index| ((index * 17 + rows * 3 + 1) % 29) as f32 / 11.0 - 1.0)
                    .collect();
                let right_values: Vec<f32> = (0..inner * columns)
                    .map(|index| ((index * 13 + columns * 5 + 2) % 31) as f32 / 9.0 - 1.5)
                    .collect();
                let left = tensor(&[rows, inner], &left_values);
                let right = tensor(&[inner, columns], &right_values);
                let reference = matmul_reference(&left.view(), &right.view()).unwrap();
                for block in blocks {
                    let actual = matmul_blocked(&left.view(), &right.view(), block).unwrap();
                    assert_close(actual.as_slice(), reference.as_slice());
                }
            }
        }
    }
}

#[test]
fn kernel_outputs_are_new_owners_independent_of_inputs() {
    let output = {
        let left = tensor(&[1, 2], &[2.0, 3.0]);
        let right = tensor(&[2, 1], &[4.0, 5.0]);
        matmul_blocked(&left.view(), &right.view(), BlockSize::DEFAULT).unwrap()
    };
    assert_eq!(output.shape(), &[1, 1]);
    assert_eq!(output.as_slice(), &[23.0]);
}

#[test]
fn blocked_accumulation_order_is_deterministic_for_a_fixed_tile() {
    let left = tensor(
        &[3, 5],
        &(0..15).map(|x| x as f32 / 7.0).collect::<Vec<_>>(),
    );
    let right = tensor(
        &[5, 4],
        &(0..20).map(|x| x as f32 / 9.0).collect::<Vec<_>>(),
    );
    let block = BlockSize::try_new(2, 3, 3).unwrap();
    let first = matmul_blocked(&left.view(), &right.view(), block).unwrap();
    let second = matmul_blocked(&left.view(), &right.view(), block).unwrap();
    assert_eq!(first.as_slice(), second.as_slice());
}
