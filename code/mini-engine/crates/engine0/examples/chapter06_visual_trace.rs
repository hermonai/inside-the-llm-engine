#[path = "../visual_support/chapter06.rs"]
mod fixture;
use engine0::linear::{dot_reference, gemv_reference, matmul_blocked, matmul_reference, BlockSize};
use engine0::tensor::{checked_offset, TensorView};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, x, b) = fixture::inputs();
    let (m, k, n) = (w.shape()[0], w.shape()[1], b.shape()[1]);
    let y = gemv_reference(&w.view(), &x.view())?;
    let c = matmul_reference(&w.view(), &b.view())?;
    let block = fixture::numbers("block");
    let candidate = matmul_blocked(
        &w.view(),
        &b.view(),
        BlockSize::try_new(block[0] as usize, block[1] as usize, block[2] as usize)?,
    )?;
    let row = TensorView::try_from_parts(w.as_slice(), vec![k], vec![1], 0)?;
    let dot = dot_reference(&row, &x.view())?;
    let mut offsets = Vec::new();
    let mut terms = Vec::new();
    let mut sums = Vec::new();
    let mut acc = 0.0_f32;
    for i in 0..m {
        for p in 0..k {
            offsets.push(checked_offset(w.shape(), w.strides(), 0, w.len(), &[i, p])?);
            if i == 0 {
                let value = *w.get2(i, p)? * x.as_slice()[p];
                terms.push(value);
                acc += value;
                sums.push(acc);
            }
        }
    }
    println!("{{\"shape\": [{m},{k},{n}],\"y\": {:?},\"c\": {:?},\"candidate\": {:?},\"dot\": {dot},\"products\": {:?},\"sums\": {:?},\"w_offsets\": {:?},\"w_strides\": {:?},\"b_strides\": {:?},\"c_strides\": {:?}}}",y.as_slice(),c.as_slice(),candidate.as_slice(),terms,sums,offsets,w.strides(),b.strides(),c.strides());
    Ok(())
}
