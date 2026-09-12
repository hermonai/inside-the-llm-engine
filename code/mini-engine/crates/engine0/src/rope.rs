//! Checked standard RoPE: one token's `[heads, head_dim]` activation.
//! F32 storage; F64 phases and pair arithmetic. No attention or cache state.
use crate::tensor::{OwnedTensor, TensorError, TensorView};
use std::fmt;

/// How coordinates in the rotary prefix form two-dimensional planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    Adjacent,
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    head_dim: usize,
    rotary_dim: usize,
    base: f64,
    pairing: Pairing,
}

/// Largest accepted magnitude; exact integer conversion is not phase accuracy.
pub const MAX_EXACT_POSITION: i64 = 1_i64 << 53;

impl RopeConfig {
    pub fn try_new(
        head_dim: usize,
        rotary_dim: usize,
        base: f64,
        pairing: Pairing,
    ) -> Result<Self, RopeError> {
        if head_dim == 0
            || rotary_dim == 0
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
        {
            return Err(RopeError::InvalidDimensions {
                head_dim,
                rotary_dim,
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(RopeError::InvalidBase);
        }
        Ok(Self {
            head_dim,
            rotary_dim,
            base,
            pairing,
        })
    }
    pub fn head_dim(self) -> usize {
        self.head_dim
    }
    pub fn rotary_dim(self) -> usize {
        self.rotary_dim
    }
    pub fn base(self) -> f64 {
        self.base
    }
    pub fn pairing(self) -> Pairing {
        self.pairing
    }

    fn indices(self, pair: usize) -> (usize, usize) {
        match self.pairing {
            Pairing::Adjacent => (2 * pair, 2 * pair + 1),
            Pairing::SplitHalf => (pair, pair + self.rotary_dim / 2),
        }
    }
}

struct Plan {
    phases: Vec<(f64, f64)>,
    identity: bool,
}

fn rotated(a: f32, b: f32, (sin, cos): (f64, f64)) -> (f32, f32) {
    let (a, b) = (f64::from(a), f64::from(b));
    ((a * cos - b * sin) as f32, (a * sin + b * cos) as f32)
}

fn prepare(input: &TensorView<'_>, position: i64, config: RopeConfig) -> Result<Plan, RopeError> {
    if input.rank() != 2 || input.shape()[0] == 0 || input.shape()[1] != config.head_dim {
        return Err(RopeError::ShapeMismatch {
            expected_width: config.head_dim,
            actual: input.shape().to_vec(),
        });
    }
    if !(-MAX_EXACT_POSITION..=MAX_EXACT_POSITION).contains(&position) {
        return Err(RopeError::PositionOutOfRange(position));
    }
    for h in 0..input.shape()[0] {
        for j in 0..config.head_dim {
            if !input.get2(h, j)?.is_finite() {
                return Err(RopeError::NonFiniteInput {
                    head: h,
                    coordinate: j,
                });
            }
        }
    }
    let mut phases = Vec::with_capacity(config.rotary_dim / 2);
    for pair in 0..config.rotary_dim / 2 {
        let omega = config
            .base
            .powf(-2.0 * (pair as f64) / (config.rotary_dim as f64));
        let angle = (position as f64) * omega;
        if !omega.is_finite() || omega <= 0.0 || !angle.is_finite() {
            return Err(RopeError::UnrepresentablePhase { pair });
        }
        phases.push(angle.sin_cos());
    }
    // Dry run every pair, so even a late F32 output overflow precedes all writes.
    if position != 0 {
        for h in 0..input.shape()[0] {
            for (pair, &phase) in phases.iter().enumerate() {
                let (i, j) = config.indices(pair);
                let (a, b) = rotated(*input.get2(h, i)?, *input.get2(h, j)?, phase);
                if !a.is_finite() || !b.is_finite() {
                    return Err(RopeError::NonFiniteOutput { head: h, pair });
                }
            }
        }
    }
    Ok(Plan {
        phases,
        identity: position == 0,
    })
}

fn commit(output: &mut OwnedTensor, config: RopeConfig, plan: &Plan) {
    if plan.identity {
        return;
    }
    let heads = output.shape()[0];
    let mut view = output.view_mut();
    for h in 0..heads {
        for (pair, &phase) in plan.phases.iter().enumerate() {
            let (i, j) = config.indices(pair);
            // Copy both original values before writing either coordinate.
            let a = *view.get_mut(&[h, i]).expect("validated pair index");
            let b = *view.get_mut(&[h, j]).expect("validated pair index");
            let (a, b) = rotated(a, b, phase);
            *view.get_mut(&[h, i]).expect("validated pair index") = a;
            *view.get_mut(&[h, j]).expect("validated pair index") = b;
        }
    }
}

/// Fresh canonical output from any validated immutable strided input view.
pub fn rope_reference(
    input: &TensorView<'_>,
    position: i64,
    config: RopeConfig,
) -> Result<OwnedTensor, RopeError> {
    let plan = prepare(input, position, config)?;
    let mut output = input.to_contiguous()?;
    commit(&mut output, config, &plan);
    Ok(output)
}

/// Rotate an exclusive canonical owner without an activation-sized copy.
///
/// Every returned error leaves its payload unchanged. Scratch holds one F64
/// sin/cos pair per plane. Vec allocation failure follows Rust's allocator
/// behavior and is not a typed error. Negative positions are diagnostic/inverse
/// rotations, not a claim that a serving API accepts negative token positions.
pub fn rope_in_place(
    output: &mut OwnedTensor,
    position: i64,
    config: RopeConfig,
) -> Result<(), RopeError> {
    let plan = prepare(&output.view(), position, config)?;
    commit(output, config, &plan);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeError {
    InvalidDimensions {
        head_dim: usize,
        rotary_dim: usize,
    },
    InvalidBase,
    ShapeMismatch {
        expected_width: usize,
        actual: Vec<usize>,
    },
    PositionOutOfRange(i64),
    NonFiniteInput {
        head: usize,
        coordinate: usize,
    },
    UnrepresentablePhase {
        pair: usize,
    },
    NonFiniteOutput {
        head: usize,
        pair: usize,
    },
    Tensor(TensorError),
}

impl From<TensorError> for RopeError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}
impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                head_dim,
                rotary_dim,
            } => write!(
                f,
                "RoPE requires 0 < even rotary width <= head width; got {rotary_dim}, {head_dim}"
            ),
            Self::InvalidBase => write!(f, "RoPE base must be finite and positive"),
            Self::ShapeMismatch {
                expected_width,
                actual,
            } => write!(
                f,
                "RoPE expects nonempty [H,{expected_width}], got {actual:?}"
            ),
            Self::PositionOutOfRange(p) => write!(
                f,
                "RoPE position {p} exceeds exact integer conversion policy"
            ),
            Self::NonFiniteInput { head, coordinate } => write!(
                f,
                "RoPE input is non-finite at head {head}, coordinate {coordinate}"
            ),
            Self::UnrepresentablePhase { pair } => {
                write!(f, "RoPE phase is unrepresentable at pair {pair}")
            }
            Self::NonFiniteOutput { head, pair } => write!(
                f,
                "RoPE F32 output is non-finite at head {head}, pair {pair}"
            ),
            Self::Tensor(e) => write!(f, "RoPE tensor: {e}"),
        }
    }
}
impl std::error::Error for RopeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => Some(e),
            _ => None,
        }
    }
}
