//! Seeded random weights in the shapes of a Llama-style decoder.
//!
//! The engine is correct when it computes exactly what the oracle computes;
//! what the model says is not the point, so the weights are random. Matrices
//! are uniform in `±1/sqrt(fan_in)`, which keeps activations near unit size
//! through the residual stream; norm weights start at one.

use engine0::sampling::SplitMix64;

use crate::config::ModelConfig;

/// Matrices are row-major `[out, in]`: row `r` holds the weights of output `r`.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub output: Vec<f32>,
}

fn uniform(rng: &mut SplitMix64, len: usize, fan_in: usize) -> Vec<f32> {
    let bound = 1.0 / (fan_in as f64).sqrt();
    (0..len)
        .map(|_| ((rng.next_unit_f64() * 2.0 - 1.0) * bound) as f32)
        .collect()
}

impl Weights {
    pub fn random(cfg: &ModelConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let d = cfg.d_model;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_norm: vec![1.0; d],
                wq: uniform(&mut rng, cfg.q_width() * d, d),
                wk: uniform(&mut rng, cfg.kv_width() * d, d),
                wv: uniform(&mut rng, cfg.kv_width() * d, d),
                wo: uniform(&mut rng, d * cfg.q_width(), cfg.q_width()),
                ffn_norm: vec![1.0; d],
                w_gate: uniform(&mut rng, cfg.d_ff * d, d),
                w_up: uniform(&mut rng, cfg.d_ff * d, d),
                w_down: uniform(&mut rng, d * cfg.d_ff, cfg.d_ff),
            })
            .collect();
        Self {
            embed: uniform(&mut rng, cfg.vocab * d, 1),
            layers,
            final_norm: vec![1.0; d],
            output: uniform(&mut rng, cfg.vocab * d, d),
        }
    }

    /// Every stored f32, which must equal `ModelConfig::parameters`.
    pub fn parameter_count(&self) -> usize {
        let layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm.len()
                    + l.wq.len()
                    + l.wk.len()
                    + l.wv.len()
                    + l.wo.len()
                    + l.ffn_norm.len()
                    + l.w_gate.len()
                    + l.w_up.len()
                    + l.w_down.len()
            })
            .sum();
        self.embed.len() + layer + self.final_norm.len() + self.output.len()
    }
}
