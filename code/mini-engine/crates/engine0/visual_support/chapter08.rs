//! Restricted numeric-array fixture reader for this checked-in test fixture.
use engine0::qkv::QkvConfig;
use engine0::tensor::OwnedTensor;
const DATA: &str = include_str!("../../../../reference/fixtures/chapter08-qkv.json");

pub fn numbers(name: &str) -> Vec<f32> {
    DATA.split_once(&format!("\"{name}\": ["))
        .expect("fixture key")
        .1
        .split_once(']')
        .expect("array end")
        .0
        .split(',')
        .map(|n| n.trim().parse().expect("numeric fixture"))
        .collect()
}

pub fn inputs() -> (QkvConfig, OwnedTensor, [OwnedTensor; 3]) {
    let g = numbers("geometry");
    let c = QkvConfig::try_new(g[0] as usize, g[1] as usize, g[2] as usize, g[3] as usize).unwrap();
    let weights = ["query", "key", "value"].map(|name| {
        let width = if name == "query" {
            c.query_width()
        } else {
            c.kv_width()
        };
        OwnedTensor::from_vec(vec![width, c.model_dim()], numbers(name)).unwrap()
    });
    (
        c,
        OwnedTensor::from_vec(vec![c.model_dim()], numbers("input")).unwrap(),
        weights,
    )
}
