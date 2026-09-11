#[path = "../visual_support/chapter07.rs"]
mod fixture;
use engine0::embedding::{embedding_lookup_reference, embedding_sequence_reference};
use engine0::normalization::rms_norm_reference;
use engine0::tensor::checked_offset;
use engine0::tokenizer::TokenId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (table, weight, epsilon) = fixture::inputs();
    let tokens: Vec<_> = fixture::numbers("tokens")
        .into_iter()
        .map(|n| TokenId(n as u32))
        .collect();
    let x = embedding_lookup_reference(&table.view(), tokens[0])?;
    let sequence = embedding_sequence_reference(&table.view(), &tokens)?;
    let y = rms_norm_reference(&x.view(), &weight.view(), epsilon)?;
    let mut offsets = Vec::new();
    let mut squares = Vec::new();
    let mut sums = Vec::new();
    let mut total = 0.0_f32;
    for (i, value) in x.as_slice().iter().enumerate() {
        offsets.push(checked_offset(
            table.shape(),
            table.strides(),
            0,
            table.len(),
            &[1, i],
        )?);
        squares.push(value * value);
        total += value * value;
        sums.push(total);
    }
    println!("{{\"x\": {:?},\"sequence\": {:?},\"output\": {:?},\"offsets\": {:?},\"squares\": {:?},\"sums\": {:?}}}", x.as_slice(), sequence.as_slice(), y.as_slice(), offsets, squares, sums);
    Ok(())
}
