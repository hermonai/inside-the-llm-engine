//! Checked embedding-table lookup for Transformer Primitives v1.
//!
//! Embedding parameters stay borrowed and immutable. Every public lookup
//! materializes a new canonical tensor so the returned residual activation has
//! request-local ownership and may later be mutated without aliasing weights.

use std::fmt;

use crate::tensor::{checked_element_count, OwnedTensor, TensorError, TensorView};
use crate::tokenizer::TokenId;

/// Select one row from a logical embedding table `[V,D] -> [D]`.
///
/// Any valid rank-2 immutable view is accepted, including strided and
/// zero-stride layouts. The result is always a fresh canonical owner.
pub fn embedding_lookup_reference(
    table: &TensorView<'_>,
    token: TokenId,
) -> Result<OwnedTensor, EmbeddingError> {
    let (vocab_size, model_dimension) = validate_table(table)?;
    let row = checked_token_index(token, vocab_size)?;

    let mut output = Vec::with_capacity(model_dimension);
    for column in 0..model_dimension {
        output.push(*table.get2(row, column)?);
    }
    OwnedTensor::from_vec(vec![model_dimension], output).map_err(EmbeddingError::Tensor)
}

/// Select a token sequence from `[V,D]` into a new canonical `[T,D]` tensor.
///
/// An empty token sequence is valid and produces shape `[0,D]`. The table
/// itself must still have positive vocabulary and model dimensions.
pub fn embedding_sequence_reference(
    table: &TensorView<'_>,
    tokens: &[TokenId],
) -> Result<OwnedTensor, EmbeddingError> {
    let (vocab_size, model_dimension) = validate_table(table)?;
    let count = checked_element_count(&[tokens.len(), model_dimension]).map_err(|_| {
        EmbeddingError::OutputShapeOverflow {
            tokens: tokens.len(),
            model_dimension,
        }
    })?;
    let mut output = Vec::with_capacity(count);

    for &token in tokens {
        let row = checked_token_index(token, vocab_size)?;
        for column in 0..model_dimension {
            output.push(*table.get2(row, column)?);
        }
    }
    OwnedTensor::from_vec(vec![tokens.len(), model_dimension], output)
        .map_err(EmbeddingError::Tensor)
}

fn validate_table(table: &TensorView<'_>) -> Result<(usize, usize), EmbeddingError> {
    if table.rank() != 2 {
        return Err(EmbeddingError::RankMismatch {
            expected: 2,
            actual: table.rank(),
        });
    }
    let vocab_size = table.shape()[0];
    let model_dimension = table.shape()[1];
    if vocab_size == 0 {
        return Err(EmbeddingError::EmptyVocabulary);
    }
    if model_dimension == 0 {
        return Err(EmbeddingError::EmptyModelDimension);
    }
    Ok((vocab_size, model_dimension))
}

fn checked_token_index(token: TokenId, vocab_size: usize) -> Result<usize, EmbeddingError> {
    let row = usize::try_from(token.0)
        .map_err(|_| EmbeddingError::TokenOutOfRange { token, vocab_size })?;
    if row >= vocab_size {
        Err(EmbeddingError::TokenOutOfRange { token, vocab_size })
    } else {
        Ok(row)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyVocabulary,
    EmptyModelDimension,
    TokenOutOfRange {
        token: TokenId,
        vocab_size: usize,
    },
    OutputShapeOverflow {
        tokens: usize,
        model_dimension: usize,
    },
    Tensor(TensorError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { expected, actual } => write!(
                f,
                "embedding table rank mismatch: expected {expected}, got {actual}"
            ),
            Self::EmptyVocabulary => f.write_str("embedding vocabulary must not be empty"),
            Self::EmptyModelDimension => f.write_str("embedding model dimension must not be empty"),
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token id {token} is outside embedding vocabulary 0..{vocab_size}"
            ),
            Self::OutputShapeOverflow {
                tokens,
                model_dimension,
            } => write!(
                f,
                "embedding output shape [{tokens},{model_dimension}] overflows"
            ),
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl From<TensorError> for EmbeddingError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}
