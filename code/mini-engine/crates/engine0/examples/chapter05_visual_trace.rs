//! Execute the exact Chapter 5 plates; stdout is JSON for the independent oracle.
use engine0::tensor::{checked_offset, OwnedTensor, TensorView};

fn emit(name: &str, view: &TensorView<'_>, storage_len: usize) {
    let mut offsets = Vec::new();
    let mut values = Vec::new();
    for i in 0..view.shape()[0] {
        for j in 0..view.shape()[1] {
            offsets.push(
                checked_offset(
                    view.shape(),
                    view.strides(),
                    view.base_offset(),
                    storage_len,
                    &[i, j],
                )
                .expect("valid visual fixture"),
            );
            values.push(*view.get2(i, j).expect("valid fixture coordinate"));
        }
    }
    println!(
        "\"{name}\": {{\"shape\": {:?}, \"strides\": {:?}, \"base\": {}, \"values\": {:?}, \"offsets\": {:?}}}",
        view.shape(), view.strides(), view.base_offset(), values, offsets
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = OwnedTensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let view = source.view();
    let transpose = view.transpose()?;
    let copy = transpose.to_contiguous()?;
    let reshape = view.reshape_view(vec![3, 2])?;
    let slice = view.slice_axis(1, 1, 3)?;
    println!("{{");
    emit("source", &view, source.len());
    println!(",");
    emit("transpose", &transpose, source.len());
    println!(",");
    emit("copy", &copy.view(), copy.len());
    println!(",");
    emit("reshape", &reshape, source.len());
    println!(",");
    emit("slice", &slice, source.len());
    println!("}}");
    Ok(())
}
