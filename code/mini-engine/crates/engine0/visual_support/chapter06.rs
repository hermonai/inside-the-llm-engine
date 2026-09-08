//! Tiny fixture reader shared only by examples/tests; not a public engine API.
//! The deliberately restricted input is flat JSON numeric arrays, checked
//! against dimensions before constructing real tensor owners.
use engine0::tensor::OwnedTensor;
const DATA: &str = include_str!("../../../../reference/fixtures/chapter06-visual.json");
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
pub fn inputs() -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let shape: Vec<usize> = numbers("shape").iter().map(|v| *v as usize).collect();
    let (m, k, n) = (shape[0], shape[1], shape[2]);
    (
        OwnedTensor::from_vec(vec![m, k], numbers("w")).unwrap(),
        OwnedTensor::from_vec(vec![k], numbers("x")).unwrap(),
        OwnedTensor::from_vec(vec![k, n], numbers("b")).unwrap(),
    )
}
