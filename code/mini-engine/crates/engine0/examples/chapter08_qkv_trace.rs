#[path = "../visual_support/chapter08.rs"]
mod fixture;
use engine0::embedding::embedding_lookup_reference;
use engine0::normalization::rms_norm_reference;
use engine0::qkv::{project_qkv_reference, QkvProjection};
use engine0::tensor::OwnedTensor;
use engine0::tokenizer::TokenId;

fn json(p: &QkvProjection) -> String {
    format!(
        "{{\"query\": {:?}, \"key\": {:?}, \"value\": {:?}}}",
        p.query().as_slice(),
        p.key().as_slice(),
        p.value().as_slice()
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, x, [wq, wk, wv]) = fixture::inputs();
    let raw = project_qkv_reference(config, &x.view(), &wq.view(), &wk.view(), &wv.view())?;
    // Chapter 7's visual table, including a zero row and an unrelated row.
    let mut data = vec![0.0; 4];
    data.extend_from_slice(x.as_slice());
    data.extend_from_slice(&[2.0, 1.0, 0.0, -1.0]);
    let table = OwnedTensor::from_vec(vec![3, 4], data)?;
    let residual = embedding_lookup_reference(&table.view(), TokenId(1))?;
    let gain = OwnedTensor::from_vec(vec![4], fixture::numbers("gain"))?;
    let normalized = rms_norm_reference(
        &residual.view(),
        &gain.view(),
        fixture::numbers("epsilon")[0],
    )?;
    let composed = project_qkv_reference(
        config,
        &normalized.view(),
        &wq.view(),
        &wk.view(),
        &wv.view(),
    )?;
    let offsets: Vec<_> = (0..config.query_heads())
        .flat_map(|h| (0..config.head_dim()).map(move |j| h * config.head_dim() + j))
        .collect();
    let head = composed.query_head(1)?;
    println!("{{\"input\": {:?}, \"normalized\": {:?}, \"raw\": {}, \"composed\": {}, \"offsets\": {:?}, \"head1\": {:?}}}",
        x.as_slice(), normalized.as_slice(), json(&raw), json(&composed), offsets, head.as_contiguous_slice()?);
    Ok(())
}
