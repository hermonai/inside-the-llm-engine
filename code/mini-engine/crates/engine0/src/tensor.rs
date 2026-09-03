//! A small, checked `f32` tensor and view substrate.
//!
//! Tensor Substrate v1 separates owned contiguous storage from borrowed
//! strided views. It deliberately omits operators: a tensor describes data and
//! layout; later chapters will implement computations over those descriptions.

use std::fmt;
use std::mem::size_of;

/// The semantic storage type of a tensor element.
///
/// V1 implements only `F32`; keeping the name visible prevents byte width from
/// becoming an accidental substitute for dtype semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

impl DType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
        }
    }
}

/// Contiguous, row-major `f32` storage with checked shape metadata.
///
/// `from_vec` moves the input allocation into the tensor. Views borrow that
/// allocation; only `zeros` and an explicit view materialization allocate new
/// element storage inside this module.
#[derive(Clone, PartialEq)]
pub struct OwnedTensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl OwnedTensor {
    /// Move `data` into a canonical row-major tensor.
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let expected = checked_element_count(&shape)?;
        if data.len() != expected {
            return Err(TensorError::StorageLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        let strides = canonical_row_major_strides(&shape)?;
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// Allocate and zero-initialize a canonical row-major tensor.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let count = checked_element_count(&shape)?;
        Self::from_vec(shape, vec![0.0; count])
    }

    pub const fn dtype(&self) -> DType {
        DType::F32
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Canonical row-major strides measured in elements, not bytes.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Physical row-major storage. This does not create a logical copy.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            storage: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            base_offset: 0,
        }
    }

    /// Exclusively borrow the complete canonical tensor.
    ///
    /// V1 intentionally provides no arbitrary-stride mutable constructor. This
    /// keeps overlapping mutable layouts out of the safe API.
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            storage: &mut self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn get(&self, indices: &[usize]) -> Result<&f32, TensorError> {
        let offset = checked_offset(&self.shape, &self.strides, 0, self.data.len(), indices)?;
        self.data.get(offset).ok_or(TensorError::InvalidViewExtent {
            required: offset.saturating_add(1),
            storage_len: self.data.len(),
        })
    }

    pub fn get2(&self, row: usize, column: usize) -> Result<&f32, TensorError> {
        self.get(&[row, column])
    }
}

impl fmt::Debug for OwnedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedTensor")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("contiguous", &true)
            .field("elements", &self.data.len())
            .finish()
    }
}

/// Immutable metadata over borrowed `f32` storage.
///
/// Shapes and strides are owned by the view; element storage is borrowed for
/// `'a` and therefore cannot be used after its owner is dropped.
#[derive(Clone)]
pub struct TensorView<'a> {
    storage: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    base_offset: usize,
}

impl<'a> TensorView<'a> {
    /// Construct a general non-negative-stride read-only view.
    ///
    /// Strides and `base_offset` are measured in elements. The constructor
    /// proves that every reachable logical index is inside `storage`.
    pub fn try_from_parts(
        storage: &'a [f32],
        shape: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
        base_offset: usize,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let strides = strides.into();
        validate_view_extent(&shape, &strides, base_offset, storage.len())?;
        Ok(Self {
            storage,
            shape,
            strides,
            base_offset,
        })
    }

    pub const fn dtype(&self) -> DType {
        DType::F32
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element strides. Multiply by `DType::byte_width()` for byte distance.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    pub fn element_count(&self) -> Result<usize, TensorError> {
        checked_element_count(&self.shape)
    }

    /// V1 contiguity is deliberately strict: strides must exactly equal the
    /// canonical row-major strides, including axes of length one.
    pub fn is_contiguous_row_major(&self) -> bool {
        canonical_row_major_strides(&self.shape).is_ok_and(|canonical| canonical == self.strides)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&'a f32, TensorError> {
        let offset = checked_offset(
            &self.shape,
            &self.strides,
            self.base_offset,
            self.storage.len(),
            indices,
        )?;
        self.storage
            .get(offset)
            .ok_or(TensorError::InvalidViewExtent {
                required: offset.saturating_add(1),
                storage_len: self.storage.len(),
            })
    }

    pub fn get2(&self, row: usize, column: usize) -> Result<&'a f32, TensorError> {
        self.get(&[row, column])
    }

    /// Borrow the same storage with a new canonical shape.
    ///
    /// This never copies and rejects non-canonical source layouts. The new
    /// shape must describe exactly the same number of logical elements.
    pub fn reshape_view(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        if !self.is_contiguous_row_major() {
            return Err(TensorError::NonContiguous);
        }
        let new_shape = new_shape.into();
        let current = self.element_count()?;
        let requested = checked_element_count(&new_shape)?;
        if current != requested {
            return Err(TensorError::ReshapeElementMismatch { current, requested });
        }
        let new_strides = canonical_row_major_strides(&new_shape)?;
        Self::try_from_parts(self.storage, new_shape, new_strides, self.base_offset)
    }

    /// Return a rank-2 transpose by swapping shape and stride metadata.
    /// No element storage is allocated or moved.
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedMatrix { rank: self.rank() });
        }
        let shape = vec![self.shape[1], self.shape[0]];
        let strides = vec![self.strides[1], self.strides[0]];
        Self::try_from_parts(self.storage, shape, strides, self.base_offset)
    }

    /// Borrow a bounded half-open range on one axis without copying elements.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Result<Self, TensorError> {
        let dimension = *self.shape.get(axis).ok_or(TensorError::AxisOutOfBounds {
            axis,
            rank: self.rank(),
        })?;
        if start > end || end > dimension {
            return Err(TensorError::InvalidSlice {
                axis,
                start,
                end,
                dimension,
            });
        }
        let shift = start
            .checked_mul(self.strides[axis])
            .ok_or(TensorError::OffsetOverflow)?;
        let base_offset = self
            .base_offset
            .checked_add(shift)
            .ok_or(TensorError::OffsetOverflow)?;
        let mut shape = self.shape.clone();
        shape[axis] = end - start;
        Self::try_from_parts(self.storage, shape, self.strides.clone(), base_offset)
    }

    /// Materialize logical iteration order into a new canonical owner.
    ///
    /// This is the explicit element-copy boundary for non-contiguous views.
    pub fn to_contiguous(&self) -> Result<OwnedTensor, TensorError> {
        let count = self.element_count()?;
        let mut copied = Vec::with_capacity(count);
        for flat_index in 0..count {
            let indices = unravel_row_major(flat_index, &self.shape);
            copied.push(*self.get(&indices)?);
        }
        OwnedTensor::from_vec(self.shape.clone(), copied)
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("base_offset", &self.base_offset)
            .field("contiguous", &self.is_contiguous_row_major())
            .finish()
    }
}

/// Exclusive access to one complete canonical owner.
pub struct TensorViewMut<'a> {
    storage: &'a mut [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorViewMut<'_> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut f32, TensorError> {
        let offset = checked_offset(&self.shape, &self.strides, 0, self.storage.len(), indices)?;
        let storage_len = self.storage.len();
        self.storage
            .get_mut(offset)
            .ok_or(TensorError::InvalidViewExtent {
                required: offset.saturating_add(1),
                storage_len,
            })
    }
}

impl fmt::Debug for TensorViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorViewMut")
            .field("dtype", &DType::F32)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("contiguous", &true)
            .finish()
    }
}

/// Checked product of logical dimensions. The rank-0 empty product is one;
/// any zero-sized dimension makes the result zero without allocating.
pub fn checked_element_count(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(TensorError::ShapeOverflow)
    })
}

pub fn checked_byte_count(shape: &[usize], dtype: DType) -> Result<usize, TensorError> {
    checked_element_count(shape)?
        .checked_mul(dtype.byte_width())
        .ok_or(TensorError::ByteCountOverflow)
}

/// Derive canonical row-major element strides from right to left.
pub fn canonical_row_major_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    let mut strides = vec![0; shape.len()];
    let mut suffix = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = suffix;
        suffix = suffix
            .checked_mul(shape[axis])
            .ok_or(TensorError::ShapeOverflow)?;
    }
    Ok(strides)
}

/// Authoritative checked logical-index to physical-element offset mapping.
pub fn checked_offset(
    shape: &[usize],
    strides: &[usize],
    base_offset: usize,
    storage_len: usize,
    indices: &[usize],
) -> Result<usize, TensorError> {
    validate_view_extent(shape, strides, base_offset, storage_len)?;
    if indices.len() != shape.len() {
        return Err(TensorError::RankMismatch {
            expected: shape.len(),
            actual: indices.len(),
        });
    }

    let mut offset = base_offset;
    for (axis, ((&index, &dimension), &stride)) in
        indices.iter().zip(shape).zip(strides).enumerate()
    {
        if index >= dimension {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index,
                dimension,
            });
        }
        let step = index
            .checked_mul(stride)
            .ok_or(TensorError::OffsetOverflow)?;
        offset = offset
            .checked_add(step)
            .ok_or(TensorError::OffsetOverflow)?;
    }
    Ok(offset)
}

fn validate_view_extent(
    shape: &[usize],
    strides: &[usize],
    base_offset: usize,
    storage_len: usize,
) -> Result<(), TensorError> {
    if shape.len() != strides.len() {
        return Err(TensorError::ShapeStrideRankMismatch {
            shape_rank: shape.len(),
            stride_rank: strides.len(),
        });
    }
    let count = checked_element_count(shape)?;
    if count == 0 {
        if base_offset <= storage_len {
            return Ok(());
        }
        return Err(TensorError::InvalidViewExtent {
            required: base_offset,
            storage_len,
        });
    }

    let mut max_offset = base_offset;
    for (&dimension, &stride) in shape.iter().zip(strides) {
        let contribution = (dimension - 1)
            .checked_mul(stride)
            .ok_or(TensorError::OffsetOverflow)?;
        max_offset = max_offset
            .checked_add(contribution)
            .ok_or(TensorError::OffsetOverflow)?;
    }
    let required = max_offset
        .checked_add(1)
        .ok_or(TensorError::OffsetOverflow)?;
    if required > storage_len {
        return Err(TensorError::InvalidViewExtent {
            required,
            storage_len,
        });
    }
    Ok(())
}

fn unravel_row_major(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = flat_index % shape[axis];
        flat_index /= shape[axis];
    }
    indices
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeOverflow,
    ByteCountOverflow,
    StorageLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ShapeStrideRankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    AxisOutOfBounds {
        axis: usize,
        rank: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dimension: usize,
    },
    OffsetOverflow,
    InvalidViewExtent {
        required: usize,
        storage_len: usize,
    },
    NonContiguous,
    ReshapeElementMismatch {
        current: usize,
        requested: usize,
    },
    ExpectedMatrix {
        rank: usize,
    },
    InvalidSlice {
        axis: usize,
        start: usize,
        end: usize,
        dimension: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => {
                f.write_str("tensor element-count or stride arithmetic overflowed")
            }
            Self::ByteCountOverflow => f.write_str("tensor byte-count arithmetic overflowed"),
            Self::StorageLengthMismatch { expected, actual } => {
                write!(f, "shape needs {expected} elements, storage has {actual}")
            }
            Self::ShapeStrideRankMismatch {
                shape_rank,
                stride_rank,
            } => write!(
                f,
                "shape rank {shape_rank} does not match stride rank {stride_rank}"
            ),
            Self::RankMismatch { expected, actual } => {
                write!(
                    f,
                    "index rank {actual} does not match tensor rank {expected}"
                )
            }
            Self::AxisOutOfBounds { axis, rank } => {
                write!(f, "axis {axis} is outside tensor rank {rank}")
            }
            Self::IndexOutOfBounds {
                axis,
                index,
                dimension,
            } => write!(
                f,
                "index {index} is outside axis {axis} dimension 0..{dimension}"
            ),
            Self::OffsetOverflow => f.write_str("tensor physical-offset arithmetic overflowed"),
            Self::InvalidViewExtent {
                required,
                storage_len,
            } => write!(
                f,
                "tensor view requires storage length {required}, available {storage_len}"
            ),
            Self::NonContiguous => {
                f.write_str("operation requires canonical row-major contiguous layout")
            }
            Self::ReshapeElementMismatch { current, requested } => write!(
                f,
                "reshape changes element count from {current} to {requested}"
            ),
            Self::ExpectedMatrix { rank } => {
                write!(f, "rank-2 transpose requires a matrix, got rank {rank}")
            }
            Self::InvalidSlice {
                axis,
                start,
                end,
                dimension,
            } => write!(
                f,
                "slice {start}..{end} is invalid for axis {axis} dimension {dimension}"
            ),
        }
    }
}

impl std::error::Error for TensorError {}
