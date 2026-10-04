use ch18_chunked_prefill::{schedule_step, Request};
fn main() {
    let mut r = Request::new(10_240, 6_144);
    for k in 0..5 {
        if r.remaining() == 0 {
            break;
        }
        let s = schedule_step(&mut r, 1024, 64 + k * 16, 32, true);
        println!(
            "step {k}: decode={} prefill={} committed {} -> {}",
            s.decodes, s.scheduled_prefill, s.before, s.after
        );
    }
}
