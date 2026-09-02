//! Pedagogical sampling-only timing probe for Chapter 4.
//!
//! This is not an LLM benchmark. It excludes model forward execution and
//! measures the straightforward scalar teaching implementations.

use std::hint::black_box;
use std::time::Instant;

use engine0::model::Logits;
use engine0::sampling::{SamplerState, SamplingConfig};

const REPETITIONS: usize = 7;

fn fixture(size: usize) -> Logits {
    let values = (0..size)
        .map(|index| {
            let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((mixed % 10_007) as f32 / 1_001.0) - 5.0
        })
        .collect();
    Logits::try_from_values(values).expect("finite synthetic fixture")
}

fn measure(logits: &Logits, config: SamplingConfig, iterations: usize) -> u128 {
    let mut samples = Vec::with_capacity(REPETITIONS);
    for repetition in 0..=REPETITIONS {
        let mut sampler = SamplerState::try_new(config.clone()).expect("valid benchmark config");
        for _ in 0..32 {
            black_box(sampler.sample(black_box(logits)).expect("warmup sample"));
        }
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(sampler.sample(black_box(logits)).expect("timed sample"));
        }
        let nanos_per_call = started.elapsed().as_nanos() / iterations as u128;
        if repetition > 0 {
            samples.push(nanos_per_call);
        }
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    println!("profile=release repetitions={REPETITIONS} statistic=median warmup=32");
    println!("vocab,iterations,greedy_ns,softmax_categorical_ns,top_k_40_ns,top_p_0_9_ns");
    for (vocabulary, iterations) in [(16, 50_000), (256, 10_000), (4_096, 500)] {
        let logits = fixture(vocabulary);
        let greedy = measure(&logits, SamplingConfig::Greedy, iterations);
        let categorical = measure(
            &logits,
            SamplingConfig::stochastic(1.0, None, None, 0x5eed),
            iterations,
        );
        let top_k = measure(
            &logits,
            SamplingConfig::stochastic(1.0, Some(40), None, 0x5eed),
            iterations,
        );
        let top_p = measure(
            &logits,
            SamplingConfig::stochastic(1.0, None, Some(0.9), 0x5eed),
            iterations,
        );
        println!("{vocabulary},{iterations},{greedy},{categorical},{top_k},{top_p}");
    }
}
