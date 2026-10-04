use ch22_speculative::{exact_output_mass, wrong_output_mass, Frontiers};
fn main() {
    let p = [0.5, 0.3, 0.2];
    let q = [0.6, 0.1, 0.3];
    println!("target={p:?}");
    println!("exact ={:?}", exact_output_mass(&p, &q));
    println!("wrong ={:?}", wrong_output_mass(&p, &q));
    let mut f = Frontiers::new(100);
    f.draft(4);
    f.evaluate_drafts();
    println!("tentative={f:?}");
    f.reconcile(2);
    println!("reconciled={f:?}");
}
