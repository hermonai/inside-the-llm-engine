//! The capstone's differential tests: every engine path against the oracle,
//! compared bit for bit, because the kernels make exactness achievable.

use std::collections::HashMap;

use capstone::engine::{Event, Outcome, Sampling};
use capstone::kv::KvPool;
use capstone::model::SeqChunk;
use capstone::oracle::{oracle_generate, oracle_logits};
use capstone::{Engine, EngineConfig, GenerationRequest, Model, ModelConfig};
use engine0::StopReason;

fn tiny_model() -> Model {
    Model::new(ModelConfig::tiny(), 42)
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| (seed * 31 + i * 17 + 7) % 256)
        .collect()
}

fn greedy(prompt: Vec<u32>, max_new_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        prompt,
        max_new_tokens,
        stop: None,
        sampling: Sampling::Greedy,
    }
}

fn tokens_by_request(events: &[Event]) -> HashMap<u64, Vec<u32>> {
    let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
    for e in events {
        if let Event::Token { id, index, token } = e {
            let v = out.entry(*id).or_default();
            assert_eq!(v.len(), *index, "token indices arrive in order");
            v.push(*token);
        }
    }
    out
}

#[test]
fn model_forward_reproduces_the_oracle_bit_for_bit_through_chunks_and_scattered_blocks() {
    let model = tiny_model();
    let tokens = prompt(3, 13);
    let mut pool = KvPool::new(16, 2, model.cfg.n_layers, model.cfg.kv_width());
    // Take blocks out of order, so the sequence's cache is not contiguous.
    let spare = pool.alloc(3).unwrap();
    let mut table = pool.alloc(4).unwrap();
    pool.release(&spare).unwrap();
    table.extend(pool.alloc(5).unwrap()); // 9 blocks of 2 cover the 17 positions
                                          // Prefill in chunks of 5, 5 and 3, then decode four tokens one at a time.
    let mut seq = tokens.clone();
    let mut start = 0;
    for len in [5, 5, 3] {
        let chunk = SeqChunk {
            tokens: &seq[start..start + len],
            start,
            table: &table,
            want_logits: true,
        };
        let logits = model
            .forward(&[chunk], &mut pool)
            .unwrap()
            .remove(0)
            .unwrap();
        let expected = oracle_logits(&model, &seq[..start + len]);
        assert!(logits
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        start += len;
    }
    for step in 0..4 {
        let next = capstone::kernels::argmax(&oracle_logits(&model, &seq)) as u32;
        seq.push(next);
        let chunk = SeqChunk {
            tokens: &seq[start..start + 1],
            start,
            table: &table,
            want_logits: true,
        };
        let logits = model
            .forward(&[chunk], &mut pool)
            .unwrap()
            .remove(0)
            .unwrap();
        let expected = oracle_logits(&model, &seq);
        assert!(
            logits
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "decode step {step} differs from the oracle"
        );
        start += 1;
    }
}

#[test]
fn continuous_batching_reproduces_the_oracle_token_for_token() {
    let model = tiny_model();
    let requests: Vec<(Vec<u32>, usize)> = vec![
        (prompt(1, 1), 9),
        (prompt(2, 19), 7),
        (prompt(3, 6), 12),
        (prompt(4, 11), 5),
        (prompt(5, 3), 10),
    ];
    let expected: Vec<Vec<u32>> = requests
        .iter()
        .map(|(p, n)| oracle_generate(&model, p, *n, None))
        .collect();

    // A small budget forces chunked prefill; max_running 3 forces queueing;
    // block_size 4 spreads every sequence over several blocks.
    let cfg = EngineConfig {
        n_blocks: 40,
        block_size: 4,
        max_running: 3,
        token_budget: 7,
    };
    let mut engine = Engine::new(tiny_model(), cfg);
    let ids: Vec<u64> = requests
        .iter()
        .map(|(p, n)| engine.submit(greedy(p.clone(), *n)).unwrap())
        .collect();
    let events = engine.run_to_completion().unwrap();
    let got = tokens_by_request(&events);
    for (id, want) in ids.iter().zip(&expected) {
        assert_eq!(&got[id], want, "request {id} differs from the oracle");
    }
    assert!(engine.stats.prefill_tokens > 0 && engine.stats.decode_tokens > 0);
    assert_eq!(
        engine.free_blocks(),
        cfg.n_blocks,
        "every block is returned"
    );
}

#[test]
fn threaded_matrix_products_reproduce_the_oracle_token_for_token() {
    // Splitting output rows among threads never splits a reduction, so the
    // tokens (and every logit) must not depend on the thread count.
    let requests: Vec<(Vec<u32>, usize)> =
        vec![(prompt(6, 13), 8), (prompt(7, 4), 11), (prompt(8, 9), 6)];
    let model = tiny_model();
    let expected: Vec<Vec<u32>> = requests
        .iter()
        .map(|(p, n)| oracle_generate(&model, p, *n, None))
        .collect();
    let cfg = EngineConfig {
        n_blocks: 32,
        block_size: 4,
        max_running: 3,
        token_budget: 8,
    };
    for threads in [2, 3, 4] {
        let mut engine = Engine::new(tiny_model().with_threads(threads), cfg);
        let ids: Vec<u64> = requests
            .iter()
            .map(|(p, n)| engine.submit(greedy(p.clone(), *n)).unwrap())
            .collect();
        let got = tokens_by_request(&engine.run_to_completion().unwrap());
        for (id, want) in ids.iter().zip(&expected) {
            assert_eq!(
                &got[id], want,
                "threads={threads}: request {id} differs from the oracle"
            );
        }
    }
}

#[test]
fn a_request_generates_the_same_tokens_alone_or_in_a_crowd() {
    let target = GenerationRequest {
        prompt: prompt(9, 7),
        max_new_tokens: 12,
        stop: None,
        sampling: Sampling::Temperature {
            temperature: 0.8,
            seed: 2024,
        },
    };
    let cfg = EngineConfig {
        n_blocks: 64,
        block_size: 4,
        max_running: 8,
        token_budget: 16,
    };

    let mut alone = Engine::new(tiny_model(), cfg);
    let id = alone.submit(target.clone()).unwrap();
    let solo = tokens_by_request(&alone.run_to_completion().unwrap())
        .remove(&id)
        .unwrap();

    let mut crowd = Engine::new(tiny_model(), cfg);
    for i in 0..3 {
        crowd
            .submit(greedy(prompt(20 + i, 5 + 4 * i as usize), 9))
            .unwrap();
    }
    let id = crowd.submit(target).unwrap();
    for i in 0..3 {
        crowd
            .submit(greedy(prompt(30 + i, 2 + 3 * i as usize), 6))
            .unwrap();
    }
    let batched = tokens_by_request(&crowd.run_to_completion().unwrap())
        .remove(&id)
        .unwrap();
    assert_eq!(solo, batched);
}

#[test]
fn every_request_ends_exactly_once_and_cancellation_frees_its_blocks() {
    let cfg = EngineConfig {
        n_blocks: 24,
        block_size: 4,
        max_running: 2,
        token_budget: 8,
    };
    let mut engine = Engine::new(tiny_model(), cfg);
    let a = engine.submit(greedy(prompt(1, 5), 20)).unwrap();
    let b = engine.submit(greedy(prompt(2, 5), 20)).unwrap();
    let c = engine.submit(greedy(prompt(3, 5), 4)).unwrap(); // waits: max_running is 2
    let mut events = Vec::new();
    for _ in 0..3 {
        events.extend(engine.step().unwrap());
    }
    engine.cancel(a); // running
    engine.cancel(c); // still waiting
    events.extend(engine.run_to_completion().unwrap());

    let mut finished: HashMap<u64, Vec<Outcome>> = HashMap::new();
    let mut after_finish = Vec::new();
    for e in &events {
        match e {
            Event::Finished { id, outcome, .. } => {
                finished.entry(*id).or_default().push(outcome.clone())
            }
            Event::Token { id, .. } | Event::Text { id, .. } if finished.contains_key(id) => {
                after_finish.push(*id)
            }
            _ => {}
        }
    }
    assert_eq!(finished[&a], vec![Outcome::Cancelled]);
    assert_eq!(
        finished[&b],
        vec![Outcome::Completed(StopReason::MaxTokens)]
    );
    assert_eq!(finished[&c], vec![Outcome::Cancelled]);
    assert!(after_finish.is_empty(), "no events after a terminal event");
    assert_eq!(engine.free_blocks(), cfg.n_blocks);
}

#[test]
fn admission_reserves_memory_and_rejects_the_impossible() {
    let cfg = EngineConfig {
        n_blocks: 6,
        block_size: 4,
        max_running: 4,
        token_budget: 32,
    };
    let mut engine = Engine::new(tiny_model(), cfg);
    assert!(engine.submit(greedy(prompt(1, 20), 10)).is_err()); // 30 tokens: 8 blocks > 6
    engine.submit(greedy(prompt(1, 8), 8)).unwrap(); // 4 blocks
    engine.submit(greedy(prompt(2, 4), 4)).unwrap(); // 2 blocks: fits beside the first
    engine.submit(greedy(prompt(3, 4), 4)).unwrap(); // must wait for memory
    engine.step().unwrap();
    assert_eq!(engine.running(), 2);
    assert_eq!(engine.free_blocks(), 0);
    engine.run_to_completion().unwrap();
    assert_eq!(engine.free_blocks(), 6);
}

#[test]
fn a_stop_token_ends_generation_with_end_of_sequence() {
    let model = tiny_model();
    let p = prompt(7, 6);
    let free_run = oracle_generate(&model, &p, 6, None);
    let stop = free_run[2];
    let cfg = EngineConfig {
        n_blocks: 16,
        block_size: 4,
        max_running: 1,
        token_budget: 8,
    };
    let mut engine = Engine::new(tiny_model(), cfg);
    let id = engine
        .submit(GenerationRequest {
            prompt: p,
            max_new_tokens: 6,
            stop: Some(stop),
            sampling: Sampling::Greedy,
        })
        .unwrap();
    let events = engine.run_to_completion().unwrap();
    let tokens = &tokens_by_request(&events)[&id];
    let first_stop = free_run.iter().position(|&t| t == stop).unwrap();
    assert_eq!(tokens, &free_run[..=first_stop]);
    assert!(events.contains(&Event::Finished {
        id,
        outcome: Outcome::Completed(StopReason::EndOfSequence),
        generated: first_stop + 1
    }));
}

#[test]
fn configured_sizes_match_the_allocated_weights() {
    let model = tiny_model();
    assert_eq!(model.w.parameter_count(), model.cfg.parameters());
    let bench = ModelConfig::bench();
    assert_eq!(bench.parameters(), 97_010_688);
    assert_eq!(bench.kv_bytes_per_token(), 2 * 8 * 256 * 4);
}
