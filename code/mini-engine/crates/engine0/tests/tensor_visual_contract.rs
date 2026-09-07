use engine0::tensor::{OwnedTensor, TensorError, TensorView};

fn fixture() -> OwnedTensor {
    OwnedTensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap()
}

#[test]
fn tensor_visual_transpose_and_reshape_have_same_shape_but_different_values() {
    let owner = fixture();
    let transpose = owner.view().transpose().unwrap();
    let reshape = owner.view().reshape_view(vec![3, 2]).unwrap();
    assert_eq!(transpose.shape(), reshape.shape());
    assert_eq!(*transpose.get2(0, 1).unwrap(), 4.);
    assert_eq!(*reshape.get2(0, 1).unwrap(), 2.);
    assert!(std::ptr::eq(
        transpose.get2(0, 1).unwrap(),
        owner.get2(1, 0).unwrap()
    ));
    assert!(matches!(
        transpose.reshape_view(vec![6]),
        Err(TensorError::NonContiguous)
    ));
}

#[test]
fn tensor_visual_column_slice_requires_six_backing_slots_for_four_values() {
    let owner = fixture();
    let sliced = owner.view().slice_axis(1, 1, 3).unwrap();
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.strides(), &[3, 1]);
    assert_eq!(sliced.base_offset(), 1);
    assert_eq!(*sliced.get2(1, 1).unwrap(), 6.);
    assert!(matches!(
        TensorView::try_from_parts(&owner.as_slice()[..5], vec![2, 2], vec![3, 1], 1),
        Err(TensorError::InvalidViewExtent { .. })
    ));
}

#[test]
fn tensor_visual_copy_and_clone_are_independent_owners() {
    let owner = fixture();
    let mut copied = owner.view().transpose().unwrap().to_contiguous().unwrap();
    let mut cloned = owner.clone();
    assert_eq!(copied.as_slice(), &[1., 4., 2., 5., 3., 6.]);
    *copied.view_mut().get_mut(&[0, 1]).unwrap() = 40.;
    *cloned.view_mut().get_mut(&[0, 0]).unwrap() = 99.;
    assert_eq!(owner.as_slice(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn tensor_visual_shared_last_use_then_exclusive_write() {
    let mut owner = fixture();
    let shared = owner.view();
    assert_eq!(*shared.get2(1, 2).unwrap(), 6.); // Last use ends the shared borrow.
    *owner.view_mut().get_mut(&[1, 2]).unwrap() = 60.;
    assert_eq!(*owner.view().get2(1, 2).unwrap(), 60.);
}
