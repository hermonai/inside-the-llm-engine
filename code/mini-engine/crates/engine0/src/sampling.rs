//! Chapter 4 sampling: immutable policy plus request-owned mutable state.

use std::cmp::Ordering;
use std::fmt;

use crate::model::Logits;
use crate::tokenizer::TokenId;

/// The immutable sampling policy carried by one generation request.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SamplingConfig {
    /// Select the lowest token ID among logits tied for the maximum value.
    #[default]
    Greedy,
    /// Transform logits into a categorical distribution and draw from it.
    Stochastic(StochasticConfig),
}

impl SamplingConfig {
    pub fn stochastic(
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
    ) -> Self {
        Self::Stochastic(StochasticConfig {
            temperature,
            top_k,
            top_p,
            seed,
        })
    }

    pub fn validate(&self) -> Result<(), SamplingError> {
        let Self::Stochastic(config) = self else {
            return Ok(());
        };
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(config.temperature));
        }
        if config.top_k == Some(0) {
            return Err(SamplingError::InvalidTopK(0));
        }
        if let Some(top_p) = config.top_p {
            if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
                return Err(SamplingError::InvalidTopP(top_p));
            }
        }
        Ok(())
    }
}

/// Stochastic configuration is immutable; its seed initializes per-request RNG state.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticConfig {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: u64,
}

/// A normalized distribution in vocabulary/token-ID order.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityDistribution {
    values: Vec<f64>,
}

impl ProbabilityDistribution {
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    pub fn retained_token_ids(&self) -> Vec<TokenId> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, probability)| **probability > 0.0)
            .map(|(index, _)| {
                token_id(index).expect("a model vocabulary that fits Logits also fits TokenId")
            })
            .collect()
    }
}

/// Observable result of one selection step.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingStep {
    pub token_id: TokenId,
    pub probabilities: Option<ProbabilityDistribution>,
    pub draw: Option<f64>,
    pub sample_index: usize,
}

/// Mutable sampling state. A fresh value is constructed for every request.
#[derive(Debug, Clone)]
pub struct SamplerState {
    config: SamplingConfig,
    rng: Option<SplitMix64>,
    samples: usize,
}

impl SamplerState {
    pub fn try_new(config: SamplingConfig) -> Result<Self, SamplingError> {
        config.validate()?;
        let rng = match &config {
            SamplingConfig::Greedy => None,
            SamplingConfig::Stochastic(stochastic) => Some(SplitMix64::new(stochastic.seed)),
        };
        Ok(Self {
            config,
            rng,
            samples: 0,
        })
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn sample(&mut self, logits: &Logits) -> Result<SamplingStep, SamplingError> {
        let sample_index = self.samples;
        let (token_id, probabilities, draw) = match &self.config {
            SamplingConfig::Greedy => (greedy_argmax(logits)?, None, None),
            SamplingConfig::Stochastic(config) => {
                let probabilities = stochastic_distribution(logits, config)?;
                let draw = self
                    .rng
                    .as_mut()
                    .expect("stochastic config owns an RNG")
                    .next_unit_f64();
                let token_id = categorical_select(probabilities.as_slice(), draw)?;
                (token_id, Some(probabilities), Some(draw))
            }
        };
        self.samples += 1;
        Ok(SamplingStep {
            token_id,
            probabilities,
            draw,
            sample_index,
        })
    }
}

/// Select argmax without computing softmax. Strict `>` preserves the first ID on ties.
pub fn greedy_argmax(logits: &Logits) -> Result<TokenId, SamplingError> {
    let values = logits.as_slice();
    let mut best_index = 0usize;
    let mut best = *values.first().ok_or(SamplingError::EmptyLogits)?;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > best {
            best = value;
            best_index = index;
        }
    }
    token_id(best_index)
}

/// Stable softmax for a finite, non-empty vector.
pub fn stable_softmax(logits: &[f64]) -> Result<Vec<f64>, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if let Some((index, value)) = logits
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SamplingError::NonFiniteScore { index, value });
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<f64> = logits.iter().map(|value| (value - maximum).exp()).collect();
    normalize(&mut probabilities)?;
    Ok(probabilities)
}

/// ENGINE-1's fixed processing order for stochastic sampling.
pub fn stochastic_distribution(
    logits: &Logits,
    config: &StochasticConfig,
) -> Result<ProbabilityDistribution, SamplingError> {
    SamplingConfig::Stochastic(config.clone()).validate()?;
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }

    // This workspace is separate from the immutable raw Logits object. A
    // removed candidate is represented by `None`, not by corrupting the
    // finite-logit model boundary with -infinity.
    let mut processed: Vec<Option<f64>> = logits
        .as_slice()
        .iter()
        .map(|value| Some(f64::from(*value) / config.temperature))
        .collect();

    if let Some(k) = config.top_k {
        let mut order: Vec<usize> = (0..processed.len()).collect();
        order.sort_by(|left, right| {
            descending_then_id(
                processed[*left].expect("unmasked before top-k"),
                *left,
                processed[*right].expect("unmasked before top-k"),
                *right,
            )
        });
        for index in order.into_iter().skip(k.min(processed.len())) {
            processed[index] = None;
        }
    }

    let active_scores: Vec<f64> = processed.iter().flatten().copied().collect();
    if active_scores.is_empty() {
        return Err(SamplingError::AllCandidatesFiltered);
    }
    let active_probabilities = stable_softmax(&active_scores)?;
    let mut probabilities = vec![0.0; processed.len()];
    let mut source = active_probabilities.into_iter();
    for (index, score) in processed.iter().enumerate() {
        if score.is_some() {
            probabilities[index] = source.next().expect("one probability per active score");
        }
    }

    if let Some(top_p) = config.top_p.filter(|top_p| *top_p < 1.0) {
        let mut order: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter_map(|(index, probability)| (*probability > 0.0).then_some(index))
            .collect();
        order.sort_by(|left, right| {
            descending_then_id(probabilities[*left], *left, probabilities[*right], *right)
        });

        let mut cumulative = 0.0;
        let mut keep = vec![false; probabilities.len()];
        for index in order {
            keep[index] = true;
            cumulative += probabilities[index];
            if cumulative >= top_p {
                break;
            }
        }
        for (index, probability) in probabilities.iter_mut().enumerate() {
            if !keep[index] {
                *probability = 0.0;
            }
        }
        normalize(&mut probabilities)?;
    }

    validate_probabilities(&probabilities)?;
    Ok(ProbabilityDistribution {
        values: probabilities,
    })
}

/// Map one artificial or RNG-produced draw in `[0, 1)` to a token interval.
pub fn categorical_select(probabilities: &[f64], draw: f64) -> Result<TokenId, SamplingError> {
    if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(SamplingError::InvalidDraw(draw));
    }
    validate_probabilities(probabilities)?;

    let mut cumulative = 0.0;
    let mut final_positive = None;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if probability > 0.0 {
            final_positive = Some(index);
        }
        cumulative += probability;
        if draw < cumulative {
            return token_id(index);
        }
    }

    // Rounding may leave the final cumulative sum a few ulps below one.
    token_id(final_positive.ok_or(SamplingError::AllCandidatesFiltered)?)
}

fn normalize(probabilities: &mut [f64]) -> Result<(), SamplingError> {
    let sum: f64 = probabilities.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SamplingError::InvalidProbabilitySum(sum));
    }
    for probability in probabilities {
        *probability /= sum;
    }
    Ok(())
}

fn validate_probabilities(probabilities: &[f64]) -> Result<(), SamplingError> {
    if probabilities.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() {
            return Err(SamplingError::NonFiniteProbability { index, probability });
        }
        if probability < 0.0 {
            return Err(SamplingError::NegativeProbability { index, probability });
        }
    }
    if !probabilities.iter().any(|probability| *probability > 0.0) {
        return Err(SamplingError::AllCandidatesFiltered);
    }
    let sum: f64 = probabilities.iter().sum();
    if (sum - 1.0).abs() > 1e-12 {
        return Err(SamplingError::InvalidProbabilitySum(sum));
    }
    Ok(())
}

fn descending_then_id(left: f64, left_id: usize, right: f64, right_id: usize) -> Ordering {
    right.total_cmp(&left).then_with(|| left_id.cmp(&right_id))
}

fn token_id(index: usize) -> Result<TokenId, SamplingError> {
    u32::try_from(index)
        .map(TokenId)
        .map_err(|_| SamplingError::TokenIdOverflow(index))
}

/// SplitMix64, used only as a small deterministic educational PRNG.
///
/// This is not a cryptographic random-number generator. `next_unit_f64`
/// takes the high 53 output bits, so every returned draw is in `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub fn next_unit_f64(&mut self) -> f64 {
        const DENOMINATOR: f64 = (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 / DENOMINATOR
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    EmptyLogits,
    InvalidTemperature(f64),
    InvalidTopK(usize),
    InvalidTopP(f64),
    InvalidDraw(f64),
    NonFiniteScore { index: usize, value: f64 },
    NonFiniteProbability { index: usize, probability: f64 },
    NegativeProbability { index: usize, probability: f64 },
    InvalidProbabilitySum(f64),
    AllCandidatesFiltered,
    TokenIdOverflow(usize),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => f.write_str("cannot sample an empty logit vector"),
            Self::InvalidTemperature(value) => {
                write!(
                    f,
                    "temperature must be finite and greater than zero, got {value}"
                )
            }
            Self::InvalidTopK(value) => write!(f, "top_k must be greater than zero, got {value}"),
            Self::InvalidTopP(value) => write!(f, "top_p must be in (0, 1], got {value}"),
            Self::InvalidDraw(value) => {
                write!(f, "categorical draw must be in [0, 1), got {value}")
            }
            Self::NonFiniteScore { index, value } => {
                write!(f, "non-finite processed score at index {index}: {value}")
            }
            Self::NonFiniteProbability { index, probability } => {
                write!(f, "non-finite probability at index {index}: {probability}")
            }
            Self::NegativeProbability { index, probability } => {
                write!(f, "negative probability at index {index}: {probability}")
            }
            Self::InvalidProbabilitySum(sum) => {
                write!(
                    f,
                    "probability sum must be finite, positive, and approximately one, got {sum}"
                )
            }
            Self::AllCandidatesFiltered => f.write_str("sampling removed every candidate"),
            Self::TokenIdOverflow(index) => {
                write!(f, "candidate index {index} does not fit TokenId")
            }
        }
    }
}

impl std::error::Error for SamplingError {}
