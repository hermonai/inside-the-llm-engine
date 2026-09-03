//! Explicit scalar linear-algebra kernels for ENGINE-2.
//!
//! The reference path accepts any valid read-only `TensorView`. The blocked
//! path is deliberately narrower: both matrices must already be canonical
//! row-major. Neither path performs an implicit layout conversion.

use std::fmt;

use crate::tensor::{checked_element_count, OwnedTensor, TensorError, TensorView};

/// One positive tile extent for each matrix-multiplication axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSize {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl BlockSize {
    pub const DEFAULT: Self = Self {
        m: 32,
        k: 32,
        n: 32,
    };

    pub fn try_new(m: usize, k: usize, n: usize) -> Result<Self, KernelError> {
        if m == 0 || k == 0 || n == 0 {
            return Err(KernelError::InvalidBlockSize { m, k, n });
        }
        Ok(Self { m, k, n })
    }
}

/// Scalar dot product with `f32` multiplication and accumulation.
pub fn dot_reference(left: &TensorView<'_>, right: &TensorView<'_>) -> Result<f32, KernelError> {
    require_rank("dot", "left", left, 1)?;
    require_rank("dot", "right", right, 1)?;
    let left_len = left.shape()[0];
    let right_len = right.shape()[0];
    if left_len != right_len {
        return Err(KernelError::LengthMismatch {
            operation: "dot",
            left: left_len,
            right: right_len,
        });
    }

    let mut sum = 0.0_f32;
    for index in 0..left_len {
        sum += *left.get(&[index])? * *right.get(&[index])?;
    }
    Ok(sum)
}

/// Reference matrix-vector product: `[M,K] × [K] -> [M]`.
pub fn gemv_reference(
    matrix: &TensorView<'_>,
    vector: &TensorView<'_>,
) -> Result<OwnedTensor, KernelError> {
    require_rank("gemv", "matrix", matrix, 2)?;
    require_rank("gemv", "vector", vector, 1)?;
    let (rows, inner) = (matrix.shape()[0], matrix.shape()[1]);
    require_inner("gemv", inner, vector.shape()[0])?;
    checked_output_count(rows, 1)?;

    let mut output = vec![0.0_f32; rows];
    for (row, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for k in 0..inner {
            sum += *matrix.get2(row, k)? * *vector.get(&[k])?;
        }
        *value = sum;
    }
    OwnedTensor::from_vec(vec![rows], output).map_err(KernelError::Tensor)
}

/// Reference matrix product: `[M,K] × [K,N] -> [M,N]`.
///
/// Logical indexing makes this path correct for valid non-contiguous and
/// zero-stride views. Its loop order is the equation-shaped `i, j, k` order.
pub fn matmul_reference(
    left: &TensorView<'_>,
    right: &TensorView<'_>,
) -> Result<OwnedTensor, KernelError> {
    require_rank("matmul_reference", "left", left, 2)?;
    require_rank("matmul_reference", "right", right, 2)?;
    let (rows, inner) = (left.shape()[0], left.shape()[1]);
    let (right_inner, columns) = (right.shape()[0], right.shape()[1]);
    require_inner("matmul_reference", inner, right_inner)?;
    let count = checked_output_count(rows, columns)?;

    let mut output = vec![0.0_f32; count];
    for i in 0..rows {
        for j in 0..columns {
            let mut sum = 0.0_f32;
            for k in 0..inner {
                sum += *left.get2(i, k)? * *right.get2(k, j)?;
            }
            output[i * columns + j] = sum;
        }
    }
    OwnedTensor::from_vec(vec![rows, columns], output).map_err(KernelError::Tensor)
}

/// Scalar cache-blocked matrix product for canonical row-major inputs.
///
/// The output is always a fresh canonical owner. A non-canonical input is a
/// typed error; callers must request `to_contiguous` themselves if copying is
/// the policy they want.
pub fn matmul_blocked(
    left: &TensorView<'_>,
    right: &TensorView<'_>,
    block: BlockSize,
) -> Result<OwnedTensor, KernelError> {
    require_rank("matmul_blocked", "left", left, 2)?;
    require_rank("matmul_blocked", "right", right, 2)?;
    if block.m == 0 || block.k == 0 || block.n == 0 {
        return Err(KernelError::InvalidBlockSize {
            m: block.m,
            k: block.k,
            n: block.n,
        });
    }

    let (rows, inner) = (left.shape()[0], left.shape()[1]);
    let (right_inner, columns) = (right.shape()[0], right.shape()[1]);
    require_inner("matmul_blocked", inner, right_inner)?;
    let count = checked_output_count(rows, columns)?;
    let left_data = left
        .as_contiguous_slice()
        .map_err(|error| layout_error("matmul_blocked", "left", error))?;
    let right_data = right
        .as_contiguous_slice()
        .map_err(|error| layout_error("matmul_blocked", "right", error))?;
    let mut output = vec![0.0_f32; count];

    // `ii, kk, jj` chooses tiles; `i, k, j` walks each B row and output row
    // contiguously. Shape validation proves every flat index is in range.
    for ii in (0..rows).step_by(block.m) {
        let i_end = ii.saturating_add(block.m).min(rows);
        for kk in (0..inner).step_by(block.k) {
            let k_end = kk.saturating_add(block.k).min(inner);
            for jj in (0..columns).step_by(block.n) {
                let j_end = jj.saturating_add(block.n).min(columns);
                for i in ii..i_end {
                    let output_row = i * columns;
                    let left_row = i * inner;
                    for k in kk..k_end {
                        let left_value = left_data[left_row + k];
                        let right_row = k * columns;
                        for j in jj..j_end {
                            output[output_row + j] += left_value * right_data[right_row + j];
                        }
                    }
                }
            }
        }
    }

    OwnedTensor::from_vec(vec![rows, columns], output).map_err(KernelError::Tensor)
}

fn require_rank(
    operation: &'static str,
    operand: &'static str,
    view: &TensorView<'_>,
    expected: usize,
) -> Result<(), KernelError> {
    if view.rank() == expected {
        Ok(())
    } else {
        Err(KernelError::RankMismatch {
            operation,
            operand,
            expected,
            actual: view.rank(),
        })
    }
}

fn require_inner(operation: &'static str, left: usize, right: usize) -> Result<(), KernelError> {
    if left == right {
        Ok(())
    } else {
        Err(KernelError::InnerDimensionMismatch {
            operation,
            left,
            right,
        })
    }
}

fn checked_output_count(rows: usize, columns: usize) -> Result<usize, KernelError> {
    checked_element_count(&[rows, columns])
        .map_err(|_| KernelError::OutputShapeOverflow { rows, columns })
}

fn layout_error(operation: &'static str, operand: &'static str, error: TensorError) -> KernelError {
    match error {
        TensorError::NonContiguous => KernelError::UnsupportedLayout { operation, operand },
        other => KernelError::Tensor(other),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    RankMismatch {
        operation: &'static str,
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        operation: &'static str,
        left: usize,
        right: usize,
    },
    InnerDimensionMismatch {
        operation: &'static str,
        left: usize,
        right: usize,
    },
    UnsupportedLayout {
        operation: &'static str,
        operand: &'static str,
    },
    InvalidBlockSize {
        m: usize,
        k: usize,
        n: usize,
    },
    OutputShapeOverflow {
        rows: usize,
        columns: usize,
    },
    Tensor(TensorError),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch {
                operation,
                operand,
                expected,
                actual,
            } => write!(
                f,
                "{operation} requires rank-{expected} {operand}, got rank {actual}"
            ),
            Self::LengthMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "{operation} requires equal vector lengths, got {left} and {right}"
            ),
            Self::InnerDimensionMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "{operation} inner dimensions differ: left K={left}, right K={right}"
            ),
            Self::UnsupportedLayout { operation, operand } => write!(
                f,
                "{operation} requires canonical row-major {operand} storage"
            ),
            Self::InvalidBlockSize { m, k, n } => {
                write!(f, "block dimensions must be positive, got ({m}, {k}, {n})")
            }
            Self::OutputShapeOverflow { rows, columns } => {
                write!(f, "output shape [{rows}, {columns}] overflows usize")
            }
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
        }
    }
}

impl std::error::Error for KernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for KernelError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}
