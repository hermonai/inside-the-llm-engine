use paged_kv_lab::Pool;

fn main() {
    let mut p = Pool::new(6, 4);
    p.create("A");
    for v in [10, 11, 12, 13, 14, 15] {
        p.append("A", v).expect("append A");
    }
    println!("A table: {:?}", p.table("A"));
    println!("A logical history: {:?}", p.materialize("A").unwrap());

    p.fork("A", "B").expect("fork");
    println!("fork B table: {:?}", p.table("B"));
    p.append("B", 99).expect("COW append");
    println!("after B append:");
    println!(
        "  A table/history: {:?} {:?}",
        p.table("A"),
        p.materialize("A").unwrap()
    );
    println!(
        "  B table/history: {:?} {:?}",
        p.table("B"),
        p.materialize("B").unwrap()
    );
    println!("free blocks: {}", p.free_blocks());
}
