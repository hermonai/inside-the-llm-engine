use engine0::tensor::{
    canonical_row_major_strides, checked_byte_count, checked_element_count, checked_offset, DType,
    OwnedTensor, TensorError, TensorView,
};

fn matrix() -> OwnedTensor {
    OwnedTensor::from_vec(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()
}

#[test]
fn scalar_has_rank_zero_one_element_and_empty_metadata() {
    let scalar = OwnedTensor::from_vec(vec![], vec![42.0]).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.strides(), &[]);
    assert_eq!(scalar.get(&[]), Ok(&42.0));
    assert_eq!(scalar.dtype(), DType::F32);
}

#[test]
fn owned_tensor_reports_shape_rank_strides_and_storage() {
    let tensor = matrix();
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.strides(), &[3, 1]);
    assert_eq!(tensor.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn canonical_strides_cover_rank_one_through_four() {
    assert_eq!(canonical_row_major_strides(&[4]).unwrap(), vec![1]);
    assert_eq!(canonical_row_major_strides(&[2, 3]).unwrap(), vec![3, 1]);
    assert_eq!(
        canonical_row_major_strides(&[2, 3, 4]).unwrap(),
        vec![12, 4, 1]
    );
    assert_eq!(
        canonical_row_major_strides(&[2, 3, 4, 5]).unwrap(),
        vec![60, 20, 5, 1]
    );
    assert_eq!(
        canonical_row_major_strides(&[1, 3, 1]).unwrap(),
        vec![3, 1, 1]
    );
}

#[test]
fn element_and_byte_counts_are_checked() {
    assert_eq!(checked_element_count(&[]), Ok(1));
    assert_eq!(checked_element_count(&[2, 3, 4]), Ok(24));
    assert_eq!(checked_byte_count(&[2, 3, 4], DType::F32), Ok(96));
    assert_eq!(
        checked_element_count(&[usize::MAX, 2]),
        Err(TensorError::ShapeOverflow)
    );
    assert_eq!(
        checked_byte_count(&[usize::MAX / 2 + 1], DType::F32),
        Err(TensorError::ByteCountOverflow)
    );
}

#[test]
fn canonical_stride_overflow_is_typed_without_allocation() {
    assert_eq!(
        canonical_row_major_strides(&[2, usize::MAX]),
        Err(TensorError::ShapeOverflow)
    );
}

#[test]
fn malformed_storage_length_is_rejected() {
    assert_eq!(
        OwnedTensor::from_vec(vec![2, 3], vec![0.0; 5]),
        Err(TensorError::StorageLengthMismatch {
            expected: 6,
            actual: 5
        })
    );
}

#[test]
fn checked_get_and_get2_map_logical_indices() {
    let tensor = matrix();
    assert_eq!(tensor.get(&[0, 0]), Ok(&0.0));
    assert_eq!(tensor.get2(0, 2), Ok(&2.0));
    assert_eq!(tensor.get2(1, 0), Ok(&3.0));
    assert_eq!(tensor.get(&[1, 2]), Ok(&5.0));
    assert_eq!(
        checked_offset(&[2, 3, 4], &[12, 4, 1], 0, 24, &[1, 2, 3]),
        Ok(23)
    );
}

#[test]
fn indexing_rejects_rank_and_axis_errors() {
    let tensor = matrix();
    assert_eq!(
        tensor.get(&[1]),
        Err(TensorError::RankMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        tensor.get(&[2, 0]),
        Err(TensorError::IndexOutOfBounds {
            axis: 0,
            index: 2,
            dimension: 2
        })
    );
}

#[test]
fn view_constructor_rejects_rank_mismatch_and_bad_extent() {
    let storage = [0.0; 4];
    assert!(matches!(
        TensorView::try_from_parts(&storage, vec![2, 2], vec![2], 0),
        Err(TensorError::ShapeStrideRankMismatch { .. })
    ));
    assert_eq!(
        TensorView::try_from_parts(&storage, vec![2, 2], vec![100, 1], 0).unwrap_err(),
        TensorError::InvalidViewExtent {
            required: 102,
            storage_len: 4
        }
    );
}

#[test]
fn view_extent_and_index_arithmetic_overflow_are_typed() {
    let storage = [0.0; 1];
    assert_eq!(
        TensorView::try_from_parts(&storage, vec![2], vec![usize::MAX], 1).unwrap_err(),
        TensorError::OffsetOverflow
    );
    assert_eq!(
        checked_offset(&[1], &[1], usize::MAX, usize::MAX, &[0]),
        Err(TensorError::OffsetOverflow)
    );
}

#[test]
fn base_offset_and_slice_select_without_copying() {
    let tensor = OwnedTensor::from_vec(vec![4, 3], (0..12).map(|x| x as f32).collect()).unwrap();
    let slice = tensor.view().slice_axis(0, 1, 3).unwrap();
    assert_eq!(slice.shape(), &[2, 3]);
    assert_eq!(slice.strides(), &[3, 1]);
    assert_eq!(slice.base_offset(), 3);
    assert_eq!(slice.get2(0, 0), Ok(&3.0));
    assert_eq!(slice.get2(1, 2), Ok(&8.0));
}

#[test]
fn slicing_rejects_bad_axis_and_ranges() {
    let tensor = matrix();
    assert!(matches!(
        tensor.view().slice_axis(2, 0, 1),
        Err(TensorError::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        tensor.view().slice_axis(0, 2, 1),
        Err(TensorError::InvalidSlice { .. })
    ));
    assert!(matches!(
        tensor.view().slice_axis(1, 0, 4),
        Err(TensorError::InvalidSlice { .. })
    ));
}

#[test]
fn transpose_swaps_shape_strides_and_logical_values() {
    let tensor = matrix();
    let transposed = tensor.view().transpose().unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    assert!(!transposed.is_contiguous_row_major());
    assert_eq!(transposed.get2(0, 0), Ok(&0.0));
    assert_eq!(transposed.get2(0, 1), Ok(&3.0));
    assert_eq!(transposed.get2(2, 1), Ok(&5.0));
}

#[test]
fn contiguous_slice_borrows_exact_logical_range_without_copying() {
    let tensor = OwnedTensor::from_vec(vec![4, 2], (0..8).map(|x| x as f32).collect()).unwrap();
    let slice = tensor.view().slice_axis(0, 1, 3).unwrap();
    let packed = slice.as_contiguous_slice().unwrap();
    assert_eq!(packed, &[2.0, 3.0, 4.0, 5.0]);
    assert!(std::ptr::eq(
        packed.as_ptr(),
        tensor.as_slice()[2..].as_ptr()
    ));

    assert_eq!(
        tensor.view().transpose().unwrap().as_contiguous_slice(),
        Err(TensorError::NonContiguous)
    );
}

#[test]
fn transpose_rejects_non_matrix_rank() {
    let vector = OwnedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(
        vector.view().transpose().unwrap_err(),
        TensorError::ExpectedMatrix { rank: 1 }
    );
}

#[test]
fn canonical_reshape_borrows_without_copying() {
    let tensor = matrix();
    let reshaped = tensor.view().reshape_view(vec![3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.strides(), &[2, 1]);
    assert_eq!(reshaped.get2(2, 1), Ok(&5.0));
    assert!(std::ptr::eq(
        tensor.get2(1, 2).unwrap(),
        reshaped.get2(2, 1).unwrap()
    ));
}

#[test]
fn reshape_supports_vector_and_size_one_axes() {
    let tensor = matrix();
    assert_eq!(tensor.view().reshape_view(vec![6]).unwrap().shape(), &[6]);
    assert_eq!(
        tensor.view().reshape_view(vec![1, 6]).unwrap().strides(),
        &[6, 1]
    );
}

#[test]
fn reshape_rejects_element_mismatch_and_non_contiguous_source() {
    let tensor = matrix();
    assert_eq!(
        tensor.view().reshape_view(vec![4, 2]).unwrap_err(),
        TensorError::ReshapeElementMismatch {
            current: 6,
            requested: 8
        }
    );
    assert_eq!(
        tensor
            .view()
            .transpose()
            .unwrap()
            .reshape_view(vec![6])
            .unwrap_err(),
        TensorError::NonContiguous
    );
}

#[test]
fn to_contiguous_materializes_transposed_logical_order() {
    let tensor = matrix();
    let copied = tensor.view().transpose().unwrap().to_contiguous().unwrap();
    assert_eq!(copied.shape(), &[3, 2]);
    assert_eq!(copied.strides(), &[2, 1]);
    assert_eq!(copied.as_slice(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn immutable_view_aliases_owner_and_owned_copy_is_independent() {
    let mut tensor = matrix();
    {
        let view = tensor.view();
        assert!(std::ptr::eq(
            tensor.get2(1, 1).unwrap(),
            view.get2(1, 1).unwrap()
        ));
    }
    let mut copied = tensor.view().to_contiguous().unwrap();
    *copied.view_mut().get_mut(&[1, 1]).unwrap() = 99.0;
    assert_eq!(tensor.get2(1, 1), Ok(&4.0));
    assert_eq!(copied.get2(1, 1), Ok(&99.0));

    *tensor.view_mut().get_mut(&[0, 2]).unwrap() = 22.0;
    assert_eq!(tensor.view().get2(0, 2), Ok(&22.0));
}

#[test]
fn zero_sized_dimensions_are_valid_but_have_no_indices() {
    let tensor = OwnedTensor::from_vec(vec![0, 4], vec![]).unwrap();
    assert!(tensor.is_empty());
    assert_eq!(tensor.shape(), &[0, 4]);
    assert_eq!(tensor.strides(), &[4, 1]);
    assert_eq!(tensor.view().element_count(), Ok(0));
    assert!(matches!(
        tensor.get(&[0, 0]),
        Err(TensorError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn empty_view_may_begin_at_storage_end() {
    let storage = [1.0, 2.0, 3.0];
    let empty = TensorView::try_from_parts(&storage, vec![0], vec![1], 3).unwrap();
    assert_eq!(empty.element_count(), Ok(0));
    assert!(empty.to_contiguous().unwrap().is_empty());
    assert!(matches!(
        TensorView::try_from_parts(&storage, vec![0], vec![1], 4),
        Err(TensorError::InvalidViewExtent { .. })
    ));
}

#[test]
fn zero_stride_immutable_view_can_alias_one_element() {
    let storage = [7.0];
    let repeated = TensorView::try_from_parts(&storage, vec![3], vec![0], 0).unwrap();
    assert_eq!(repeated.get(&[0]), Ok(&7.0));
    assert_eq!(repeated.get(&[2]), Ok(&7.0));
    assert!(!repeated.is_contiguous_row_major());
    assert_eq!(
        repeated.to_contiguous().unwrap().as_slice(),
        &[7.0, 7.0, 7.0]
    );
}

#[test]
fn debug_output_reports_metadata_without_values() {
    let tensor = matrix();
    let debug = format!("{tensor:?}");
    assert!(debug.contains("shape: [2, 3]"));
    assert!(debug.contains("strides: [3, 1]"));
    assert!(!debug.contains("0.0"));
}

#[test]
fn generated_small_shapes_map_every_index_inside_storage() {
    for rank in 1..=4 {
        let shape = vec![3; rank];
        let strides = canonical_row_major_strides(&shape).unwrap();
        let count = checked_element_count(&shape).unwrap();
        let storage = vec![0.0; count];
        for flat in 0..count {
            let mut remaining = flat;
            let mut indices = vec![0; rank];
            for axis in (0..rank).rev() {
                indices[axis] = remaining % 3;
                remaining /= 3;
            }
            assert_eq!(
                checked_offset(&shape, &strides, 0, storage.len(), &indices),
                Ok(flat)
            );
        }
        assert_eq!(
            checked_offset(&shape, &strides, 0, storage.len(), &vec![2; rank]),
            Ok(count - 1)
        );
    }
}

#[test]
fn into_vec_transfers_owned_physical_storage() {
    assert_eq!(matrix().into_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}
