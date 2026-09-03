use engine0::normalization::rms_norm_reference;
use engine0::tensor::OwnedTensor;

fn tensor(values: Vec<f32>) -> OwnedTensor {
    OwnedTensor::from_vec(vec![values.len()], values).expect("example shape is valid")
}

fn max_abs_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn main() {
    const EPSILON: f32 = 1.0e-5;
    let base = [1.0_f32, -2.0, 3.0, -4.0];
    let weight = tensor(vec![1.0, 0.5, 2.0, -1.0]);
    let baseline = rms_norm_reference(&tensor(base.to_vec()).view(), &weight.view(), EPSILON)
        .expect("baseline is finite");

    println!("scale experiment (epsilon={EPSILON}, delta from alpha=1):");
    for alpha in [1.0e-8_f32, 0.1, 1.0, 10.0, 100.0] {
        let scaled = tensor(base.iter().map(|value| value * alpha).collect());
        match rms_norm_reference(&scaled.view(), &weight.view(), EPSILON) {
            Ok(output) => println!(
                "alpha={alpha:>8e} output={:?} max_abs_delta={:.9e}",
                output.as_slice(),
                max_abs_delta(output.as_slice(), baseline.as_slice())
            ),
            Err(error) => println!("alpha={alpha:>8e} error={error}"),
        }
    }

    println!("magnitude stress (alternating signs, unit weights):");
    let unit_weight = tensor(vec![1.0, 1.0]);
    for magnitude in [1.0e-20_f32, 1.0e-10, 1.0, 1.0e10, 1.0e20] {
        let input = tensor(vec![magnitude, -magnitude]);
        match rms_norm_reference(&input.view(), &unit_weight.view(), EPSILON) {
            Ok(output) => println!("magnitude={magnitude:>8e} output={:?}", output.as_slice()),
            Err(error) => println!("magnitude={magnitude:>8e} error={error}"),
        }
    }
}
