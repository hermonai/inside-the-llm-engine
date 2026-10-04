use ch21_multi_lora::{dynamic, merged_oracle, Adapter, Residency};
fn main() {
    let base = vec![1., 0., 2., 0., 0., 1., 0., 2., 1., 1., 1., 1.];
    let ad = Adapter {
        id: 42,
        rank: 2,
        scale: 0.5,
        a: vec![1., 0., 1., 0., 0., 1., 0., 1.],
        b: vec![1., 0., 0., 2., 1., -1.],
    };
    let x = vec![1., 2., 3., 4.];
    println!("dynamic={:?}", dynamic(&base, 3, 4, &x, Some(&ad)));
    println!("merged ={:?}", merged_oracle(&base, 3, 4, &x, Some(&ad)));
    let mut r = Residency::new(1);
    let h = r.load(0, 42);
    println!("lease={h:?} pin={}", r.pin(h));
}
