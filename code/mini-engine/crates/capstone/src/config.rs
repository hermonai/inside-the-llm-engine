//! Model shapes, and the byte, FLOP and cache counts that follow from them.

use std::fmt;

/// A Llama-style decoder: RMSNorm, rotary positions, grouped-query attention
/// and a SwiGLU feed-forward network, with untied input and output matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Zero(&'static str),
    HeadsNotGrouped { n_heads: usize, n_kv_heads: usize },
    OddHeadWidth(usize),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero(field) => write!(f, "{field} must be positive"),
            Self::HeadsNotGrouped {
                n_heads,
                n_kv_heads,
            } => {
                write!(
                    f,
                    "{n_heads} query heads cannot share {n_kv_heads} key/value heads evenly"
                )
            }
            Self::OddHeadWidth(d) => write!(f, "rotary positions need an even head width, got {d}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl ModelConfig {
    /// Small enough for the exact-oracle tests to run in milliseconds.
    pub const fn tiny() -> Self {
        Self {
            vocab: 256,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            d_head: 16,
            d_ff: 160,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
        }
    }

    /// The measured configuration of Chapter 42: 97.0 million parameters,
    /// 388 MB in f32, so one decode step at batch one reads far more than any
    /// cache holds and the machine's memory bandwidth is visible.
    pub const fn bench() -> Self {
        Self {
            vocab: 256,
            d_model: 1024,
            n_layers: 8,
            n_heads: 16,
            n_kv_heads: 4,
            d_head: 64,
            d_ff: 3072,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("vocab", self.vocab),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("d_head", self.d_head),
            ("d_ff", self.d_ff),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero(name));
            }
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(ConfigError::HeadsNotGrouped {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
            });
        }
        if !self.d_head.is_multiple_of(2) {
            return Err(ConfigError::OddHeadWidth(self.d_head));
        }
        Ok(())
    }

    pub const fn q_width(&self) -> usize {
        self.n_heads * self.d_head
    }

    pub const fn kv_width(&self) -> usize {
        self.n_kv_heads * self.d_head
    }

    /// Query heads that read the same key/value head.
    pub const fn group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Parameters in one decoder layer: four attention matrices, three
    /// feed-forward matrices and two norm vectors.
    pub const fn layer_parameters(&self) -> usize {
        let d = self.d_model;
        d * self.q_width()
            + 2 * d * self.kv_width()
            + self.q_width() * d
            + 3 * d * self.d_ff
            + 2 * d
    }

    pub const fn parameters(&self) -> usize {
        2 * self.vocab * self.d_model + self.n_layers * self.layer_parameters() + self.d_model
    }

    /// Bytes a decode step must read for weights: every matrix except the
    /// input embedding, from which a step reads one row per token.
    pub const fn weight_bytes_per_step(&self) -> usize {
        4 * (self.parameters() - self.vocab * self.d_model)
    }

    /// Cache bytes one token adds: keys and values, every layer, f32.
    pub const fn kv_bytes_per_token(&self) -> usize {
        2 * self.n_layers * self.kv_width() * 4
    }

    /// FLOPs to produce one token with `context` earlier tokens in the cache:
    /// two per weight used (a multiply and an add), plus the attention scores
    /// and the weighted sum over the context, two FLOPs per element each.
    pub fn flops_per_token(&self, context: usize) -> f64 {
        let matmul = 2.0
            * (self.parameters()
                - self.vocab * self.d_model
                - self.n_layers * 2 * self.d_model
                - self.d_model) as f64;
        let attention =
            4.0 * (self.n_layers * self.n_heads * self.d_head) as f64 * (context + 1) as f64;
        matmul + attention
    }
}
