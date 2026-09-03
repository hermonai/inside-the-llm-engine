use std::fmt;

use crate::embedding::{embedding_lookup_reference, EmbeddingError};
use crate::linear::{gemv_reference, KernelError};
use crate::tensor::{OwnedTensor, TensorError, TensorView};
use crate::tokenizer::TokenId;

/// One finite, unnormalized score per token in the model vocabulary.
///
/// This wrapper prevents the numerical model boundary from degrading into an
/// anonymous `Vec<f32>`. Chapter 4 will consume this type when it introduces
/// probability distributions and sampling policies.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    values: Vec<f32>,
}

impl Logits {
    pub fn try_from_values(values: Vec<f32>) -> Result<Self, ModelError> {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ModelError::NonFiniteLogit { index, value });
        }
        Ok(Self { values })
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Request-local activations produced by one forward execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardPass {
    pub input_token: TokenId,
    pub hidden: Vec<f32>,
    pub logits: Logits,
}

/// The numerical boundary of a language model: token history in, logits out.
pub trait Model {
    fn vocabulary_size(&self) -> usize;

    fn forward(&self, input: &[TokenId]) -> Result<ForwardPass, ModelError>;
}

/// ENGINE-1's immutable parameters in contiguous row-major `f32` storage.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyLanguageModel {
    vocab_size: usize,
    hidden_dim: usize,
    embedding: OwnedTensor,
    output_weight: OwnedTensor,
    output_bias: Vec<f32>,
}

impl TinyLanguageModel {
    pub fn try_new(
        vocab_size: usize,
        hidden_dim: usize,
        embedding: Vec<f32>,
        output_weight: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> Result<Self, ModelError> {
        if vocab_size == 0 {
            return Err(ModelError::InvalidDimensions {
                vocab_size,
                hidden_dim,
                reason: "vocabulary size must be greater than zero",
            });
        }
        if hidden_dim == 0 {
            return Err(ModelError::InvalidDimensions {
                vocab_size,
                hidden_dim,
                reason: "hidden dimension must be greater than zero",
            });
        }
        let matrix_len =
            vocab_size
                .checked_mul(hidden_dim)
                .ok_or(ModelError::DimensionOverflow {
                    vocab_size,
                    hidden_dim,
                })?;
        check_parameter_count("embedding", matrix_len, embedding.len())?;
        check_parameter_count("output projection", matrix_len, output_weight.len())?;
        check_parameter_count("output bias", vocab_size, output_bias.len())?;
        check_finite_parameters("embedding", &embedding)?;
        check_finite_parameters("output projection", &output_weight)?;
        check_finite_parameters("output bias", &output_bias)?;

        let embedding = OwnedTensor::from_vec(vec![vocab_size, hidden_dim], embedding)
            .map_err(ModelError::Tensor)?;
        let output_weight = OwnedTensor::from_vec(vec![vocab_size, hidden_dim], output_weight)
            .map_err(ModelError::Tensor)?;

        Ok(Self {
            vocab_size,
            hidden_dim,
            embedding,
            output_weight,
            output_bias,
        })
    }

    /// The hand-computable Chapter 3 fixture.
    ///
    /// Vocabulary: 0=`<eos>`, 1=`I`, 2=`like`, 3=`Rust`; hidden dimension 3.
    pub fn chapter3_fixture() -> Self {
        Self::try_new(
            4,
            3,
            vec![
                0.0, 0.0, 0.0, // <eos>
                0.0, 1.0, 0.0, // I
                1.0, -0.5, 2.0, // like
                -1.0, 0.0, 0.0, // Rust
            ],
            vec![
                -0.5, 0.4, 0.1, // candidate <eos>
                0.2, 0.2, 0.0, // candidate I
                0.3, 0.2, 0.1, // candidate like
                1.0, -0.4, 0.25, // candidate Rust
            ],
            vec![-0.2, 0.0, 0.0, 0.5],
        )
        .expect("the built-in Chapter 3 fixture has valid finite shapes")
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn parameter_count(&self) -> usize {
        self.embedding.len() + self.output_weight.len() + self.output_bias.len()
    }

    pub fn parameter_bytes(&self) -> usize {
        self.parameter_count() * std::mem::size_of::<f32>()
    }

    pub fn embedding_shape(&self) -> &[usize] {
        self.embedding.shape()
    }

    pub fn embedding_strides(&self) -> &[usize] {
        self.embedding.strides()
    }

    pub fn output_weight_shape(&self) -> &[usize] {
        self.output_weight.shape()
    }

    pub fn output_weight_strides(&self) -> &[usize] {
        self.output_weight.strides()
    }

    fn embedding_row(&self, token: TokenId) -> Result<Vec<f32>, ModelError> {
        embedding_lookup_reference(&self.embedding.view(), token)
            .map(OwnedTensor::into_vec)
            .map_err(|error| match error {
                EmbeddingError::TokenOutOfRange { token, vocab_size } => {
                    ModelError::TokenOutOfRange { token, vocab_size }
                }
                other => ModelError::Embedding(other),
            })
    }
}

impl Model for TinyLanguageModel {
    fn vocabulary_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&self, input: &[TokenId]) -> Result<ForwardPass, ModelError> {
        let input_token = input.last().copied().ok_or(ModelError::EmptyInput)?;

        // Token IDs select rows; their scalar magnitudes have no model
        // meaning. Copying the row makes this forward activation request-local
        // while the parameters remain shareable through `&self`.
        let hidden = self.embedding_row(input_token)?;
        let hidden_view = TensorView::try_from_parts(&hidden, vec![self.hidden_dim], vec![1], 0)
            .map_err(ModelError::Tensor)?;
        let mut values = gemv_reference(&self.output_weight.view(), &hidden_view)
            .map_err(ModelError::Kernel)?
            .into_vec();

        // Bias remains an explicit model operation: ENGINE-2's GEMV computes
        // W h, while the model owns the contract z = W h + b.
        for (value, bias) in values.iter_mut().zip(&self.output_bias) {
            *value += bias;
        }

        Ok(ForwardPass {
            input_token,
            hidden,
            logits: Logits::try_from_values(values)?,
        })
    }
}

fn check_parameter_count(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::InvalidParameterCount {
            name,
            expected,
            actual,
        })
    }
}

fn check_finite_parameters(name: &'static str, values: &[f32]) -> Result<(), ModelError> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Err(ModelError::NonFiniteParameter { name, index, value })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    InvalidDimensions {
        vocab_size: usize,
        hidden_dim: usize,
        reason: &'static str,
    },
    DimensionOverflow {
        vocab_size: usize,
        hidden_dim: usize,
    },
    InvalidParameterCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteParameter {
        name: &'static str,
        index: usize,
        value: f32,
    },
    EmptyInput,
    TokenOutOfRange {
        token: TokenId,
        vocab_size: usize,
    },
    NonFiniteLogit {
        index: usize,
        value: f32,
    },
    Tensor(TensorError),
    Kernel(KernelError),
    Embedding(EmbeddingError),
    InjectedFailure(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                vocab_size,
                hidden_dim,
                reason,
            } => write!(
                f,
                "invalid model dimensions V={vocab_size}, D={hidden_dim}: {reason}"
            ),
            Self::DimensionOverflow {
                vocab_size,
                hidden_dim,
            } => write!(
                f,
                "model dimensions overflow V*D for V={vocab_size}, D={hidden_dim}"
            ),
            Self::InvalidParameterCount {
                name,
                expected,
                actual,
            } => write!(
                f,
                "invalid {name} parameter count: expected {expected}, got {actual}"
            ),
            Self::NonFiniteParameter { name, index, value } => {
                write!(f, "non-finite {name} parameter at index {index}: {value}")
            }
            Self::EmptyInput => f.write_str("model input must contain at least one token"),
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token id {token} is outside model vocabulary 0..{vocab_size}"
            ),
            Self::NonFiniteLogit { index, value } => {
                write!(f, "non-finite logit at index {index}: {value}")
            }
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
            Self::Kernel(error) => write!(f, "linear algebra kernel error: {error}"),
            Self::Embedding(error) => write!(f, "embedding error: {error}"),
            Self::InjectedFailure(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ModelError {}
