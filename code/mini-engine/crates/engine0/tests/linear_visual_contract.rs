#[path = "../visual_support/chapter06.rs"]
mod fixture;
use engine0::linear::{gemv_reference, matmul_blocked, matmul_reference, BlockSize, KernelError};
use engine0::tensor::{checked_offset, OwnedTensor, TensorView};

#[test]
fn visual_gemm_columns_are_repeated_gemv() {
    let (w, x, b) = fixture::inputs();
    let c = matmul_reference(&w.view(), &b.view()).unwrap();
    for j in 0..b.shape()[1] {
        let column =
            TensorView::try_from_parts(b.as_slice(), vec![b.shape()[0]], vec![b.shape()[1]], j)
                .unwrap();
        let y = gemv_reference(&w.view(), &column).unwrap();
        for i in 0..w.shape()[0] {
            assert_eq!(y.as_slice()[i], *c.get2(i, j).unwrap());
        }
    }
    assert_eq!(
        x.as_slice(),
        b.as_slice().iter().step_by(2).copied().collect::<Vec<_>>()
    );
}

#[test]
fn visual_outer_products_and_blocked_candidate_agree() {
    let (w, _, b) = fixture::inputs();
    let (m, k, n) = (w.shape()[0], w.shape()[1], b.shape()[1]);
    let mut outer = vec![0.0_f32; m * n];
    for p in 0..k {
        for i in 0..m {
            for j in 0..n {
                outer[i * n + j] += w.as_slice()[i * k + p] * b.as_slice()[p * n + j];
            }
        }
    }
    let block = fixture::numbers("block");
    let c = matmul_blocked(
        &w.view(),
        &b.view(),
        BlockSize::try_new(block[0] as usize, block[1] as usize, block[2] as usize).unwrap(),
    )
    .unwrap();
    assert_eq!(outer, c.as_slice());
}

#[test]
fn visual_offsets_map_each_load_to_the_correct_value() {
    let (w, _, b) = fixture::inputs();
    for i in 0..w.shape()[0] {
        for k in 0..w.shape()[1] {
            let offset = checked_offset(w.shape(), w.strides(), 0, w.len(), &[i, k]).unwrap();
            assert_eq!(offset, i * w.shape()[1] + k);
            assert_eq!(w.as_slice()[offset], *w.get2(i, k).unwrap());
            assert_eq!(4 * offset, 4 * i * w.shape()[1] + 4 * k);
        }
    }
    for k in 0..b.shape()[0] {
        assert_eq!(*b.get2(k, 1).unwrap(), b.as_slice()[k * 2 + 1]);
    }
}

#[test]
fn visual_tail_fixture_and_layout_rejection() {
    let dims = fixture::numbers("tail_shape");
    let tile = fixture::numbers("tail_block");
    let (m, k, n) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let a =
        OwnedTensor::from_vec(vec![m, k], (0..m * k).map(|i| i as f32 - 11.).collect()).unwrap();
    let b =
        OwnedTensor::from_vec(vec![k, n], (0..k * n).map(|i| 13. - i as f32).collect()).unwrap();
    let block = BlockSize::try_new(tile[0] as usize, tile[1] as usize, tile[2] as usize).unwrap();
    assert_eq!(
        matmul_reference(&a.view(), &b.view()).unwrap().as_slice(),
        matmul_blocked(&a.view(), &b.view(), block)
            .unwrap()
            .as_slice()
    );
    let (w, _, _) = fixture::inputs();
    assert!(matches!(
        matmul_blocked(&w.view().transpose().unwrap(), &w.view(), block),
        Err(KernelError::UnsupportedLayout { .. })
    ));
}

#[test]
fn visual_reassociation_changes_a_binary32_sum() {
    let a = std::hint::black_box(16_777_216.0_f32);
    let b = std::hint::black_box(1.0_f32);
    let c = std::hint::black_box(-16_777_216.0_f32);
    assert_eq!((a + b) + c, 0.0);
    assert_eq!(a + (b + c), 1.0);
}

#[test]
fn visual_fma_has_one_rounding_not_two() {
    let a = std::hint::black_box(1.0_f32 + 2.0_f32.powi(-12));
    let b = std::hint::black_box(1.0_f32 - 2.0_f32.powi(-12));
    let product = std::hint::black_box(a * b);
    assert_eq!(product - 1.0, -2.0_f32.powi(-24));
    // A smaller perturbation makes the separate product round to one.
    let a = std::hint::black_box(1.0_f32 + 2.0_f32.powi(-13));
    let b = std::hint::black_box(1.0_f32 - 2.0_f32.powi(-13));
    let product = std::hint::black_box(a * b);
    assert_eq!(product - 1.0, 0.0);
    assert_eq!(a.mul_add(b, -1.0), -2.0_f32.powi(-26));
}
