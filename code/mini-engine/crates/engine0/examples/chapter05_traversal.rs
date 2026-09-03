use std::hint::black_box;
use std::time::{Duration, Instant};

use engine0::tensor::OwnedTensor;

const DEFAULT_SIDE: usize = 2048;
const DEFAULT_REPETITIONS: usize = 7;

fn row_major(storage: &[f32], side: usize) -> f64 {
    let mut checksum = 0.0_f64;
    for row in 0..side {
        for column in 0..side {
            checksum += f64::from(storage[row * side + column]);
        }
    }
    black_box(checksum)
}

fn column_wise(storage: &[f32], side: usize) -> f64 {
    let mut checksum = 0.0_f64;
    for column in 0..side {
        for row in 0..side {
            checksum += f64::from(storage[row * side + column]);
        }
    }
    black_box(checksum)
}

fn time(operation: impl FnOnce() -> f64) -> (Duration, f64) {
    let start = Instant::now();
    let checksum = operation();
    (start.elapsed(), checksum)
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let side = arguments
        .next()
        .map(|value| value.parse().expect("side must be a positive integer"))
        .unwrap_or(DEFAULT_SIDE);
    let repetitions = arguments
        .next()
        .map(|value| {
            value
                .parse()
                .expect("repetitions must be a positive integer")
        })
        .unwrap_or(DEFAULT_REPETITIONS);
    assert!(side > 0, "side must be positive");
    assert!(repetitions > 0, "repetitions must be positive");

    let count = side.checked_mul(side).expect("side squared must fit usize");
    let values = (0..count).map(|index| (index % 251) as f32).collect();
    let tensor = OwnedTensor::from_vec(vec![side, side], values).expect("valid square tensor");
    let storage = black_box(tensor.as_slice());

    let expected = row_major(storage, side);
    assert_eq!(column_wise(storage, side), expected);

    let mut row_samples = Vec::with_capacity(repetitions);
    let mut column_samples = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let (first, first_sum, second, second_sum) = if repetition % 2 == 0 {
            let (row_time, row_sum) = time(|| row_major(storage, side));
            let (column_time, column_sum) = time(|| column_wise(storage, side));
            (row_time, row_sum, column_time, column_sum)
        } else {
            let (column_time, column_sum) = time(|| column_wise(storage, side));
            let (row_time, row_sum) = time(|| row_major(storage, side));
            (row_time, row_sum, column_time, column_sum)
        };
        assert_eq!(first_sum, expected);
        assert_eq!(second_sum, expected);
        row_samples.push(first);
        column_samples.push(second);
    }

    println!("shape=[{side},{side}] repetitions={repetitions}");
    println!("checksum={expected:.1}");
    println!("row_major_median_ns={}", median(row_samples).as_nanos());
    println!(
        "column_wise_median_ns={}",
        median(column_samples).as_nanos()
    );
}
