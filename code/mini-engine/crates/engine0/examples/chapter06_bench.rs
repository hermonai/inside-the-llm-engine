//! Reproducible, dependency-free Chapter 6 release benchmark harness.
//!
//! This is a teaching experiment, not a machine leaderboard. It isolates loop
//! order, tile size, and right-hand-column reuse with deterministic inputs.

use std::hint::black_box;
use std::time::Instant;

use engine0::linear::{matmul_blocked, BlockSize};
use engine0::tensor::OwnedTensor;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 17 + seed * 13) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

fn matmul_ijk(left: &[f32], right: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += left[i * k + inner] * right[inner * n + j];
            }
            output[i * n + j] = sum;
        }
    }
    output
}

fn matmul_ikj(left: &[f32], right: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; m * n];
    for i in 0..m {
        for inner in 0..k {
            let left_value = left[i * k + inner];
            for j in 0..n {
                output[i * n + j] += left_value * right[inner * n + j];
            }
        }
    }
    output
}

fn gemv(left: &[f32], vector: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; m];
    for i in 0..m {
        let mut sum = 0.0_f32;
        for inner in 0..k {
            sum += left[i * k + inner] * vector[inner];
        }
        output[i] = sum;
    }
    output
}

fn median_ns(repetitions: usize, mut operation: impl FnMut() -> Vec<f32>) -> (u128, f32) {
    let mut samples = Vec::with_capacity(repetitions);
    let mut checksum = 0.0_f32;
    for _ in 0..repetitions {
        let start = Instant::now();
        let output = black_box(operation());
        samples.push(start.elapsed().as_nanos());
        checksum = output.iter().copied().sum();
        black_box(&output);
    }
    samples.sort_unstable();
    (samples[samples.len() / 2], checksum)
}

fn gflops(m: usize, k: usize, n: usize, nanoseconds: u128) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / nanoseconds as f64
}

fn intensity(m: usize, k: usize, n: usize) -> f64 {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let compulsory_bytes = 4.0 * (m * k + k * n + m * n) as f64;
    flops / compulsory_bytes
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(right) {
        let tolerance = 1.0e-4_f32 + 1.0e-5_f32 * right.abs();
        assert!((left - right).abs() <= tolerance);
    }
}

fn main() {
    println!("chapter06 benchmark: release build, warm process, median wall-clock");
    println!("allocation and zero-initialization are included; inputs are deterministic f32");

    println!("\n[experiment 1] loop order on square row-major matrices");
    println!("size,repetitions,ijk_ns,ikj_ns,ijk_gflops,ikj_gflops,speedup");
    for (size, repetitions) in [(64, 15), (128, 9), (256, 5)] {
        let left = values(size * size, 1);
        let right = values(size * size, 2);
        let expected = matmul_ijk(&left, &right, size, size, size);
        assert_close(&matmul_ikj(&left, &right, size, size, size), &expected);
        let (ijk_ns, checksum) =
            median_ns(repetitions, || matmul_ijk(&left, &right, size, size, size));
        let (ikj_ns, _) = median_ns(repetitions, || matmul_ikj(&left, &right, size, size, size));
        println!(
            "{size},{repetitions},{ijk_ns},{ikj_ns},{:.3},{:.3},{:.2}x # checksum={checksum:.6}",
            gflops(size, size, size, ijk_ns),
            gflops(size, size, size, ikj_ns),
            ijk_ns as f64 / ikj_ns as f64
        );
    }

    println!("\n[experiment 2a] tile sweep at M=K=N=192, 7 repetitions");
    println!("tile,median_ns,gflops,relative_to_ijk");
    let size = 192;
    let left_values = values(size * size, 3);
    let right_values = values(size * size, 4);
    let left = OwnedTensor::from_vec(vec![size, size], left_values.clone()).unwrap();
    let right = OwnedTensor::from_vec(vec![size, size], right_values.clone()).unwrap();
    let reference = matmul_ijk(&left_values, &right_values, size, size, size);
    let (ijk_ns, _) = median_ns(7, || {
        matmul_ijk(&left_values, &right_values, size, size, size)
    });
    for tile in [8, 16, 24, 32, 48, 64] {
        let block = BlockSize::try_new(tile, tile, tile).unwrap();
        let candidate = matmul_blocked(&left.view(), &right.view(), block).unwrap();
        assert_close(candidate.as_slice(), &reference);
        let (elapsed, checksum) = median_ns(7, || {
            matmul_blocked(&left.view(), &right.view(), block)
                .unwrap()
                .into_vec()
        });
        println!(
            "{tile},{elapsed},{:.3},{:.2}x # checksum={checksum:.6}",
            gflops(size, size, size, elapsed),
            ijk_ns as f64 / elapsed as f64
        );
    }

    println!("\n[experiment 2b] reference-to-blocked crossover, tile 32");
    println!("size,repetitions,ijk_ns,blocked_ns,speedup");
    for (size, repetitions) in [(8, 31), (16, 31), (32, 21), (64, 15), (128, 9)] {
        let left_values = values(size * size, 5);
        let right_values = values(size * size, 6);
        let left = OwnedTensor::from_vec(vec![size, size], left_values.clone()).unwrap();
        let right = OwnedTensor::from_vec(vec![size, size], right_values.clone()).unwrap();
        let (ijk_ns, expected_checksum) = median_ns(repetitions, || {
            matmul_ijk(&left_values, &right_values, size, size, size)
        });
        let (blocked_ns, checksum) = median_ns(repetitions, || {
            matmul_blocked(&left.view(), &right.view(), BlockSize::DEFAULT)
                .unwrap()
                .into_vec()
        });
        assert!((checksum - expected_checksum).abs() < 0.1);
        println!(
            "{size},{repetitions},{ijk_ns},{blocked_ns},{:.2}x # checksum={checksum:.6}",
            ijk_ns as f64 / blocked_ns as f64
        );
    }

    println!("\n[experiment 3] one weight matrix, growing right-hand column reuse");
    println!("n,repetitions,kernel,median_ns,gflops,ideal_flop_per_byte");
    let (m, k) = (512, 512);
    let weights_values = values(m * k, 7);
    let weights = OwnedTensor::from_vec(vec![m, k], weights_values.clone()).unwrap();
    for (n, repetitions) in [(1, 15), (8, 9), (64, 5)] {
        let inputs = values(k * n, 8 + n);
        let (elapsed, checksum) = if n == 1 {
            median_ns(repetitions, || gemv(&weights_values, &inputs, m, k))
        } else {
            let inputs = OwnedTensor::from_vec(vec![k, n], inputs).unwrap();
            median_ns(repetitions, || {
                matmul_blocked(&weights.view(), &inputs.view(), BlockSize::DEFAULT)
                    .unwrap()
                    .into_vec()
            })
        };
        println!(
            "{n},{repetitions},{},{elapsed},{:.3},{:.3} # checksum={checksum:.6}",
            if n == 1 { "gemv" } else { "blocked_gemm" },
            gflops(m, k, n, elapsed),
            intensity(m, k, n)
        );
    }
}
