use engine0::model::Logits;
use engine0::sampling::{
    categorical_select, greedy_argmax, stable_softmax, stochastic_distribution, SamplerState,
    SamplingConfig, SamplingError, SplitMix64, StochasticConfig,
};
use engine0::tokenizer::TokenId;

fn logits(values: &[f32]) -> Logits {
    Logits::try_from_values(values.to_vec()).unwrap()
}

fn config(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
) -> StochasticConfig {
    StochasticConfig {
        temperature,
        top_k,
        top_p,
        seed,
    }
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-12 + 1e-7 * expected.abs(),
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn stable_softmax_matches_the_three_logit_oracle() {
    assert_close(
        &stable_softmax(&[1.0, 2.0, 3.0]).unwrap(),
        &[0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
    );
}

#[test]
fn stable_softmax_resists_large_logit_overflow() {
    let probabilities = stable_softmax(&[1000.0, 999.0, 998.0]).unwrap();
    assert!(probabilities.iter().all(|value| value.is_finite()));
    assert_close(
        &probabilities,
        &[0.6652409557748218, 0.24472847105479764, 0.09003057317038046],
    );
}

#[test]
fn stable_softmax_is_shift_invariant() {
    let base = stable_softmax(&[-2.5, 0.0, 4.0]).unwrap();
    let shifted = stable_softmax(&[997.5, 1000.0, 1004.0]).unwrap();
    assert_close(&base, &shifted);
}

#[test]
fn softmax_probabilities_are_nonnegative_and_sum_to_one() {
    for offset in -16..=16 {
        let input = [
            f64::from(offset) * 0.25,
            f64::from(offset - 3) * 0.5,
            f64::from(7 - offset) * 0.125,
            -9.0,
        ];
        let probabilities = stable_softmax(&input).unwrap();
        assert!(probabilities.iter().all(|value| *value >= 0.0));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn empty_and_nonfinite_softmax_inputs_are_typed_errors() {
    assert_eq!(stable_softmax(&[]), Err(SamplingError::EmptyLogits));
    assert!(matches!(
        stable_softmax(&[0.0, f64::NAN]),
        Err(SamplingError::NonFiniteScore { index: 1, .. })
    ));
}

#[test]
fn greedy_uses_argmax_and_lowest_token_id_tie_breaking() {
    assert_eq!(
        greedy_argmax(&logits(&[-0.7, 0.1, 0.4, 2.2])),
        Ok(TokenId(3))
    );
    assert_eq!(
        greedy_argmax(&logits(&[-1.0, 4.0, 4.0, 3.0])),
        Ok(TokenId(1))
    );
}

#[test]
fn greedy_rejects_empty_logits() {
    assert_eq!(greedy_argmax(&logits(&[])), Err(SamplingError::EmptyLogits));
}

#[test]
fn temperature_changes_spread_without_changing_order() {
    let raw = logits(&[1.0, 2.0, 3.0]);
    let cold = stochastic_distribution(&raw, &config(0.5, None, None, 7)).unwrap();
    let neutral = stochastic_distribution(&raw, &config(1.0, None, None, 7)).unwrap();
    let hot = stochastic_distribution(&raw, &config(2.0, None, None, 7)).unwrap();
    assert_close(
        cold.as_slice(),
        &[
            0.015876239976466765,
            0.11731042782619835,
            0.8668133321973349,
        ],
    );
    assert_close(
        neutral.as_slice(),
        &[0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
    );
    assert_close(
        hot.as_slice(),
        &[0.1863237232258476, 0.3071958857184984, 0.506480391055654],
    );
    assert_eq!(
        cold.retained_token_ids(),
        vec![TokenId(0), TokenId(1), TokenId(2)]
    );
    assert!(cold.as_slice()[2] > neutral.as_slice()[2]);
    assert!(neutral.as_slice()[2] > hot.as_slice()[2]);
}

#[test]
fn top_k_keeps_exactly_the_highest_candidates_with_deterministic_ties() {
    let distribution = stochastic_distribution(
        &logits(&[3.0, 2.0, 2.0, 1.0]),
        &config(1.0, Some(2), None, 0),
    )
    .unwrap();
    assert_eq!(
        distribution.retained_token_ids(),
        vec![TokenId(0), TokenId(1)]
    );
    assert_eq!(distribution.as_slice()[2], 0.0);
    assert_eq!(distribution.as_slice()[3], 0.0);
}

#[test]
fn top_k_one_is_effectively_greedy_and_large_k_is_a_noop() {
    let raw = logits(&[-2.0, 4.0, 1.0]);
    let one = stochastic_distribution(&raw, &config(9.0, Some(1), None, 0)).unwrap();
    assert_eq!(one.as_slice(), &[0.0, 1.0, 0.0]);
    let large = stochastic_distribution(&raw, &config(1.0, Some(99), None, 0)).unwrap();
    let disabled = stochastic_distribution(&raw, &config(1.0, None, None, 0)).unwrap();
    assert_close(large.as_slice(), disabled.as_slice());
}

#[test]
fn top_p_includes_the_candidate_that_crosses_the_threshold() {
    let raw = logits(&[
        0.4f32.ln(),
        0.3f32.ln(),
        0.15f32.ln(),
        0.1f32.ln(),
        0.05f32.ln(),
    ]);
    let distribution = stochastic_distribution(&raw, &config(1.0, None, Some(0.8), 0)).unwrap();
    assert_eq!(
        distribution.retained_token_ids(),
        vec![TokenId(0), TokenId(1), TokenId(2)]
    );
    assert_close(
        distribution.as_slice(),
        &[
            0.4705882307038438,
            0.3529411794299387,
            0.1764705898662175,
            0.0,
            0.0,
        ],
    );
}

#[test]
fn top_p_one_is_disabled_and_top_k_runs_before_top_p() {
    let raw = logits(&[3.0, 2.0, 1.0, 0.0]);
    let disabled = stochastic_distribution(&raw, &config(1.0, None, None, 0)).unwrap();
    let one = stochastic_distribution(&raw, &config(1.0, None, Some(1.0), 0)).unwrap();
    assert_close(disabled.as_slice(), one.as_slice());

    let combined = stochastic_distribution(&raw, &config(1.0, Some(2), Some(0.6), 0)).unwrap();
    assert_eq!(combined.retained_token_ids(), vec![TokenId(0)]);
}

#[test]
fn categorical_selection_uses_half_open_cumulative_intervals() {
    let probabilities = [0.2, 0.3, 0.5];
    assert_eq!(categorical_select(&probabilities, 0.0), Ok(TokenId(0)));
    assert_eq!(categorical_select(&probabilities, 0.199999), Ok(TokenId(0)));
    assert_eq!(categorical_select(&probabilities, 0.2), Ok(TokenId(1)));
    assert_eq!(categorical_select(&probabilities, 0.499999), Ok(TokenId(1)));
    assert_eq!(categorical_select(&probabilities, 0.5), Ok(TokenId(2)));
    assert_eq!(
        categorical_select(&probabilities, 1.0 - f64::EPSILON),
        Ok(TokenId(2))
    );
}

#[test]
fn categorical_selection_rejects_invalid_draws_and_distributions() {
    assert!(matches!(
        categorical_select(&[0.5, 0.5], 1.0),
        Err(SamplingError::InvalidDraw(_))
    ));
    assert!(matches!(
        categorical_select(&[-0.1, 1.1], 0.2),
        Err(SamplingError::NegativeProbability { .. })
    ));
    assert!(matches!(
        categorical_select(&[0.0, 0.0], 0.2),
        Err(SamplingError::AllCandidatesFiltered)
    ));
}

#[test]
fn splitmix64_has_a_pinned_output_vector_and_unit_interval() {
    let mut rng = SplitMix64::new(0);
    let values = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
    assert_eq!(
        values,
        [
            0xe220_a839_7b1d_cdafu64,
            0x6e78_9e6a_a1b9_65f4u64,
            0x06c4_5d18_8009_454fu64,
        ]
    );
    for seed in 0..100 {
        let draw = SplitMix64::new(seed).next_unit_f64();
        assert!((0.0..1.0).contains(&draw));
    }
}

#[test]
fn same_seed_repeats_the_same_sampling_sequence() {
    let config = SamplingConfig::stochastic(1.0, None, None, 42);
    let mut left = SamplerState::try_new(config.clone()).unwrap();
    let mut right = SamplerState::try_new(config).unwrap();
    let raw = logits(&[0.0, 0.1, 0.2, 0.3]);
    let left_tokens: Vec<TokenId> = (0..16)
        .map(|_| left.sample(&raw).unwrap().token_id)
        .collect();
    let right_tokens: Vec<TokenId> = (0..16)
        .map(|_| right.sample(&raw).unwrap().token_id)
        .collect();
    assert_eq!(left_tokens, right_tokens);
    assert_eq!(left.samples(), 16);
    assert_eq!(right.samples(), 16);
}

#[test]
fn different_seeds_diverge_on_a_non_degenerate_distribution() {
    let raw = logits(&[0.0, 0.1, 0.2, 0.3]);
    let sample = |seed| {
        let mut sampler =
            SamplerState::try_new(SamplingConfig::stochastic(1.0, None, None, seed)).unwrap();
        (0..16)
            .map(|_| sampler.sample(&raw).unwrap().token_id)
            .collect::<Vec<_>>()
    };
    assert_ne!(sample(1), sample(2));
}

#[test]
fn invalid_sampling_configuration_is_rejected_without_fallback() {
    for invalid in [
        SamplingConfig::stochastic(0.0, None, None, 0),
        SamplingConfig::stochastic(-1.0, None, None, 0),
        SamplingConfig::stochastic(f64::NAN, None, None, 0),
        SamplingConfig::stochastic(1.0, Some(0), None, 0),
        SamplingConfig::stochastic(1.0, None, Some(0.0), 0),
        SamplingConfig::stochastic(1.0, None, Some(1.1), 0),
    ] {
        assert!(SamplerState::try_new(invalid).is_err());
    }
}

#[test]
fn sampling_never_mutates_raw_model_logits() {
    let raw = logits(&[-0.7, 0.1, 0.4, 2.2]);
    let before = raw.clone();
    let _ = stochastic_distribution(&raw, &config(0.7, Some(3), Some(0.9), 99)).unwrap();
    assert_eq!(raw, before);
}

#[test]
fn deterministic_property_grid_preserves_sampling_invariants() {
    for size in 1..=32 {
        let values: Vec<f32> = (0..size)
            .map(|index| ((index * 17 + size * 3) % 23) as f32 / 7.0 - 1.5)
            .collect();
        let raw = logits(&values);
        for temperature in [0.25, 1.0, 3.0] {
            let distribution = stochastic_distribution(
                &raw,
                &config(temperature, Some(size.min(7)), Some(0.91), 0),
            )
            .unwrap();
            assert!(distribution
                .as_slice()
                .iter()
                .all(|value| value.is_finite()));
            assert!(distribution.as_slice().iter().all(|value| *value >= 0.0));
            assert!((distribution.as_slice().iter().sum::<f64>() - 1.0).abs() <= 1e-12);
            assert!(distribution.retained_token_ids().len() <= size.min(7));
        }
    }
}
