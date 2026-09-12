#[path = "../visual_support/chapter08.rs"]
mod previous;
use engine0::embedding::embedding_lookup_reference;
use engine0::normalization::rms_norm_reference;
use engine0::qkv::project_qkv_reference;
use engine0::rope::{rope_in_place, rope_reference, Pairing, RopeConfig};
use engine0::tensor::OwnedTensor;
use engine0::tokenizer::TokenId;

const DATA: &str = include_str!("../../../../reference/fixtures/chapter09-rope.json");
fn numbers(name: &str) -> Vec<f32> {
    DATA.split_once(&format!("\"{name}\": ["))
        .unwrap()
        .1
        .split_once(']')
        .unwrap()
        .0
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn scalar(name: &str) -> f64 {
    DATA.split_once(&format!("\"{name}\": "))
        .unwrap()
        .1
        .split([',', '\n'])
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = scalar("head_dim") as usize;
    let r = scalar("rotary_dim") as usize;
    let input = numbers("input");
    let x = OwnedTensor::from_vec(vec![input.len() / d, d], input)?;
    let c = RopeConfig::try_new(d, r, scalar("base"), Pairing::Adjacent)?;
    let adjacent = rope_reference(&x.view(), scalar("position") as i64, c)?;
    let split = rope_reference(
        &x.view(),
        scalar("position") as i64,
        RopeConfig::try_new(d, r, scalar("base"), Pairing::SplitHalf)?,
    )?;
    let q = OwnedTensor::from_vec(vec![1, d], x.as_slice()[..d].to_vec())?;
    let k = OwnedTensor::from_vec(vec![1, d], numbers("key"))?;
    let p = scalar("query_position") as i64;
    let n = scalar("key_position") as i64;
    let qp = rope_reference(&q.view(), p, c)?;
    let kn = rope_reference(&k.view(), n, c)?;
    let kd = rope_reference(&k.view(), n - p, c)?;
    let relative_rhs = dot(q.as_slice(), kd.as_slice());
    let (geometry, input, [wq, wk, wv]) = previous::inputs();
    let mut rows = vec![0.; 4];
    rows.extend_from_slice(input.as_slice());
    rows.extend([2., 1., 0., -1.]);
    let table = OwnedTensor::from_vec(vec![3, 4], rows)?;
    let residual = embedding_lookup_reference(&table.view(), TokenId(1))?;
    let gain = OwnedTensor::from_vec(vec![4], previous::numbers("gain"))?;
    let norm = rms_norm_reference(
        &residual.view(),
        &gain.view(),
        previous::numbers("epsilon")[0],
    )?;
    let (mut q, mut k, v) =
        project_qkv_reference(geometry, &norm.view(), &wq.view(), &wk.view(), &wv.view())?
            .into_parts();
    let composed_config = RopeConfig::try_new(2, 2, 10000., Pairing::Adjacent)?;
    rope_in_place(
        &mut q,
        scalar("composition_position") as i64,
        composed_config,
    )?;
    rope_in_place(
        &mut k,
        scalar("composition_position") as i64,
        composed_config,
    )?;
    println!("{{\"adjacent\": {:?}, \"split_half\": {:?}, \"relative_lhs\": {}, \"relative_rhs\": {}, \"composed\": {{\"query\": {:?}, \"key\": {:?}, \"value\": {:?}}}}}",
        adjacent.as_slice(),split.as_slice(),dot(qp.as_slice(),kn.as_slice()),relative_rhs,q.as_slice(),k.as_slice(),v.as_slice());
    Ok(())
}
