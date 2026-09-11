//! Restricted numeric-array fixture reader, for tests/examples only.
use engine0::tensor::OwnedTensor;
const DATA: &str = include_str!("../../../../reference/fixtures/chapter07-visual.json");

pub fn numbers(name: &str) -> Vec<f32> {
    let key = format!("\"{name}\": [");
    DATA.split_once(&key)
        .expect("fixture key")
        .1
        .split_once(']')
        .expect("array end")
        .0
        .split(',')
        .map(|s| s.trim().parse().expect("numeric fixture"))
        .collect()
}

pub fn inputs() -> (OwnedTensor, OwnedTensor, f32) {
    let shape: Vec<usize> = numbers("shape").into_iter().map(|n| n as usize).collect();
    let weight = numbers("weight");
    (
        OwnedTensor::from_vec(shape, numbers("table")).unwrap(),
        OwnedTensor::from_vec(vec![weight.len()], weight).unwrap(),
        numbers("epsilon")[0],
    )
}
