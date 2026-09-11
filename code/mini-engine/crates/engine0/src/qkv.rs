//! Raw, bias-free one-token Q/K/V projections. No position or attention state.
//!
//! Model parameters use semantic `[output, input]` orientation. Inputs may be
//! valid strided immutable views; results own canonical head-major F32 storage.
//! Arithmetic inherits ordered F32 GEMV, including IEEE non-finite propagation.

use std::fmt;

use crate::linear::{gemv_reference, KernelError};
use crate::tensor::{OwnedTensor, TensorError, TensorView};

/// Validated standard geometry. Private fields prevent bypassing validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvConfig {
    model_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_width: usize,
    kv_width: usize,
}

impl QkvConfig {
    pub fn try_new(
        model_dim: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, QkvError> {
        for (name, value) in [
            ("model_dim", model_dim),
            ("query_heads", query_heads),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
        ] {
            if value == 0 {
                return Err(QkvError::ZeroDimension(name));
            }
        }
        let query_width = query_heads
            .checked_mul(head_dim)
            .ok_or(QkvError::WidthOverflow("query"))?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or(QkvError::WidthOverflow("key/value"))?;
        if model_dim != query_width {
            return Err(QkvError::ModelWidthMismatch {
                model_dim,
                query_width,
            });
        }
        if !query_heads.is_multiple_of(kv_heads) {
            return Err(QkvError::NonDivisibleHeads {
                query_heads,
                kv_heads,
            });
        }
        Ok(Self {
            model_dim,
            query_heads,
            kv_heads,
            head_dim,
            query_width,
            kv_width,
        })
    }

    pub fn model_dim(self) -> usize {
        self.model_dim
    }
    pub fn query_heads(self) -> usize {
        self.query_heads
    }
    pub fn kv_heads(self) -> usize {
        self.kv_heads
    }
    pub fn head_dim(self) -> usize {
        self.head_dim
    }
    pub fn query_width(self) -> usize {
        self.query_width
    }
    pub fn kv_width(self) -> usize {
        self.kv_width
    }
    /// Geometry only; this chapter does not execute query-to-KV matching.
    pub fn group_size(self) -> usize {
        self.query_heads / self.kv_heads
    }
}

/// Three independent activation owners, not references into model parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct QkvProjection {
    query: OwnedTensor,
    key: OwnedTensor,
    value: OwnedTensor,
}

impl QkvProjection {
    pub fn query(&self) -> &OwnedTensor {
        &self.query
    }
    pub fn key(&self) -> &OwnedTensor {
        &self.key
    }
    pub fn value(&self) -> &OwnedTensor {
        &self.value
    }

    pub fn query_head(&self, head: usize) -> Result<TensorView<'_>, QkvError> {
        head_view(&self.query, "query", head)
    }
    pub fn key_head(&self, head: usize) -> Result<TensorView<'_>, QkvError> {
        head_view(&self.key, "key", head)
    }
    pub fn value_head(&self, head: usize) -> Result<TensorView<'_>, QkvError> {
        head_view(&self.value, "value", head)
    }

    /// Transfer ownership for later transformations without copying payloads.
    pub fn into_parts(self) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
        (self.query, self.key, self.value)
    }
}

fn head_view<'a>(
    tensor: &'a OwnedTensor,
    operand: &'static str,
    head: usize,
) -> Result<TensorView<'a>, QkvError> {
    let heads = tensor.shape()[0];
    let width = tensor.shape()[1];
    if head >= heads {
        return Err(QkvError::HeadOutOfRange {
            operand,
            head,
            heads,
        });
    }
    // Private construction guarantees heads*width fits and owns this payload.
    TensorView::try_from_parts(tensor.as_slice(), vec![width], vec![1], head * width)
        .map_err(QkvError::Tensor)
}

/// Validate every semantic shape before invoking any of the three GEMVs.
///
/// Allocation failure follows the existing Rust Vec/GEMV contract (not a typed
/// QkvError). There is no hidden materialization of strided input views.
pub fn project_qkv_reference(
    config: QkvConfig,
    input: &TensorView<'_>,
    query_weight: &TensorView<'_>,
    key_weight: &TensorView<'_>,
    value_weight: &TensorView<'_>,
) -> Result<QkvProjection, QkvError> {
    require_shape("input", input, &[config.model_dim])?;
    require_shape(
        "query_weight",
        query_weight,
        &[config.query_width, config.model_dim],
    )?;
    require_shape(
        "key_weight",
        key_weight,
        &[config.kv_width, config.model_dim],
    )?;
    require_shape(
        "value_weight",
        value_weight,
        &[config.kv_width, config.model_dim],
    )?;
    let project = |weight: &TensorView<'_>, heads| {
        let flat = gemv_reference(weight, input).map_err(QkvError::Kernel)?;
        // into_vec transfers the allocation; from_vec validates new metadata.
        OwnedTensor::from_vec(vec![heads, config.head_dim], flat.into_vec())
            .map_err(QkvError::Tensor)
    };
    Ok(QkvProjection {
        query: project(query_weight, config.query_heads)?,
        key: project(key_weight, config.kv_heads)?,
        value: project(value_weight, config.kv_heads)?,
    })
}

fn require_shape(
    operand: &'static str,
    view: &TensorView<'_>,
    expected: &[usize],
) -> Result<(), QkvError> {
    if view.shape() != expected {
        return Err(QkvError::ShapeMismatch {
            operand,
            expected: expected.to_vec(),
            actual: view.shape().to_vec(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QkvError {
    ZeroDimension(&'static str),
    WidthOverflow(&'static str),
    ModelWidthMismatch {
        model_dim: usize,
        query_width: usize,
    },
    NonDivisibleHeads {
        query_heads: usize,
        kv_heads: usize,
    },
    ShapeMismatch {
        operand: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    HeadOutOfRange {
        operand: &'static str,
        head: usize,
        heads: usize,
    },
    Kernel(KernelError),
    Tensor(TensorError),
}

impl fmt::Display for QkvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "QKV {name} must be positive"),
            Self::WidthOverflow(name) => write!(f, "QKV {name} width overflows usize"),
            Self::ModelWidthMismatch {
                model_dim,
                query_width,
            } => write!(
                f,
                "QKV model width {model_dim} differs from query width {query_width}"
            ),
            Self::NonDivisibleHeads {
                query_heads,
                kv_heads,
            } => write!(
                f,
                "QKV query heads {query_heads} must be divisible by KV heads {kv_heads}"
            ),
            Self::ShapeMismatch {
                operand,
                expected,
                actual,
            } => write!(f, "QKV {operand} expected {expected:?}, got {actual:?}"),
            Self::HeadOutOfRange {
                operand,
                head,
                heads,
            } => write!(f, "QKV {operand} head {head} outside 0..{heads}"),
            Self::Kernel(error) => write!(f, "QKV kernel: {error}"),
            Self::Tensor(error) => write!(f, "QKV tensor: {error}"),
        }
    }
}

impl std::error::Error for QkvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}
