//! Checked scalar RMSNorm for Transformer Primitives v1.
//!
//! This deliberately transparent reference uses `f32` storage, products,
//! accumulation, reciprocal square root, and output. It reports finite-range
//! failures rather than silently replacing them with a stabilized norm method.

use std::fmt;

use crate::tensor::{OwnedTensor, TensorError, TensorView};

/// Normalize one logical vector and apply one learned scale per element.
///
/// For `x,w:[D]` and positive finite `epsilon`, this computes
/// `y[i] = x[i] * (1 / sqrt(mean(x*x) + epsilon)) * w[i]`.
/// Valid strided and zero-stride rank-1 views are supported. The output is a
/// fresh canonical `[D]` tensor.
pub fn rms_norm_reference(
    input: &TensorView<'_>,
    weight: &TensorView<'_>,
    epsilon: f32,
) -> Result<OwnedTensor, NormalizationError> {
    require_vector("input", input)?;
    require_vector("weight", weight)?;
    let dimension = input.shape()[0];
    if dimension != weight.shape()[0] {
        return Err(NormalizationError::LengthMismatch {
            input: dimension,
            weight: weight.shape()[0],
        });
    }
    if dimension == 0 {
        return Err(NormalizationError::EmptyDimension);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NormalizationError::InvalidEpsilon { epsilon });
    }

    let mut sum_squares = 0.0_f32;
    for index in 0..dimension {
        let value = *input.get(&[index])?;
        require_finite("input", index, value)?;
        let square = value * value;
        if !square.is_finite() {
            return Err(NormalizationError::NonFiniteSquare { index, value });
        }
        sum_squares += square;
        if !sum_squares.is_finite() {
            return Err(NormalizationError::NonFiniteReduction {
                through_index: index,
            });
        }
    }

    let mean_square = sum_squares / dimension as f32;
    let inverse_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
    if !inverse_rms.is_finite() {
        return Err(NormalizationError::NonFiniteInverseRms {
            mean_square,
            epsilon,
        });
    }

    let mut output = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let value = *input.get(&[index])?;
        let scale = *weight.get(&[index])?;
        require_finite("weight", index, scale)?;
        let normalized = value * inverse_rms * scale;
        if !normalized.is_finite() {
            return Err(NormalizationError::NonFiniteOutput {
                index,
                value: normalized,
            });
        }
        output.push(normalized);
    }

    OwnedTensor::from_vec(vec![dimension], output).map_err(NormalizationError::Tensor)
}

fn require_vector(operand: &'static str, view: &TensorView<'_>) -> Result<(), NormalizationError> {
    if view.rank() == 1 {
        Ok(())
    } else {
        Err(NormalizationError::RankMismatch {
            operand,
            expected: 1,
            actual: view.rank(),
        })
    }
}

fn require_finite(
    operand: &'static str,
    index: usize,
    value: f32,
) -> Result<(), NormalizationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(NormalizationError::NonFiniteValue {
            operand,
            index,
            value,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationError {
    RankMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        input: usize,
        weight: usize,
    },
    EmptyDimension,
    InvalidEpsilon {
        epsilon: f32,
    },
    NonFiniteValue {
        operand: &'static str,
        index: usize,
        value: f32,
    },
    NonFiniteSquare {
        index: usize,
        value: f32,
    },
    NonFiniteReduction {
        through_index: usize,
    },
    NonFiniteInverseRms {
        mean_square: f32,
        epsilon: f32,
    },
    NonFiniteOutput {
        index: usize,
        value: f32,
    },
    Tensor(TensorError),
}

impl fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch {
                operand,
                expected,
                actual,
            } => write!(
                f,
                "RMSNorm {operand} rank mismatch: expected {expected}, got {actual}"
            ),
            Self::LengthMismatch { input, weight } => write!(
                f,
                "RMSNorm length mismatch: input has {input}, weight has {weight}"
            ),
            Self::EmptyDimension => f.write_str("RMSNorm dimension must not be empty"),
            Self::InvalidEpsilon { epsilon } => {
                write!(f, "RMSNorm epsilon must be finite and positive, got {epsilon}")
            }
            Self::NonFiniteValue {
                operand,
                index,
                value,
            } => write!(f, "RMSNorm {operand}[{index}] is non-finite: {value}"),
            Self::NonFiniteSquare { index, value } => write!(
                f,
                "RMSNorm input[{index}]={value} overflows when squared in f32"
            ),
            Self::NonFiniteReduction { through_index } => write!(
                f,
                "RMSNorm sum of squares became non-finite through index {through_index}"
            ),
            Self::NonFiniteInverseRms {
                mean_square,
                epsilon,
            } => write!(
                f,
                "RMSNorm inverse RMS is non-finite for mean square {mean_square} and epsilon {epsilon}"
            ),
            Self::NonFiniteOutput { index, value } => {
                write!(f, "RMSNorm output[{index}] is non-finite: {value}")
            }
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
        }
    }
}

impl std::error::Error for NormalizationError {}

impl From<TensorError> for NormalizationError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}
