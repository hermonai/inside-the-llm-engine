use ch17_prefix_caching::PrefixCache;
fn main() {
    let mut c = PrefixCache::new(3);
    c.insert(10, vec![1, 2, 3, 4, 5, 6], 2);
    println!(
        "same tenant hit tokens = {}",
        c.longest_hit(10, &[1, 2, 3, 4, 9])
    );
    println!(
        "other tenant hit tokens = {}",
        c.longest_hit(11, &[1, 2, 3, 4, 9])
    );
    c.insert(10, vec![8, 8, 8, 8], 1);
    println!(
        "resident entries = {}, blocks = {}",
        c.len(),
        c.used_blocks()
    );
}
