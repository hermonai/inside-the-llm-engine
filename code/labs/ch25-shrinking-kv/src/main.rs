use ch25_shrinking_kv::*;
fn main() {
    println!("Llama-style: {} KiB/token", gqa_bytes(28, 8, 128, 2) / 1024);
    println!("Qwen-style: {} KiB/token", gqa_bytes(36, 2, 128, 2) / 1024);
    println!(
        "MLA example: {:0.1} KiB/token",
        mla_bytes(61, 512, 64, 2) as f64 / 1024.0
    );
    println!(
        "8Q:2KV mapping: {:?}",
        (0..8).map(|i| kv_head(i, 8, 2)).collect::<Vec<_>>()
    );
}
