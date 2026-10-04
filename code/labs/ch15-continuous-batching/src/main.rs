use continuous_batching_lab::{cohort, Scheduler};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("trace") | None => {
            let mut s = Scheduler::new(2, 3, cohort());
            s.run(12);
            s.assert_independent_accounting();
            for (tick, event) in s.events() {
                println!("t={tick:02} {event:?}");
            }
        }
        Some("broken") if args.get(2).map(String::as_str) == Some("late-result") => {
            eprintln!("The intentionally broken late-result variant is documented in README.md.");
            eprintln!("Run `cargo test late_result_is_rejected_after_generation_changes` to see the guard.");
            std::process::exit(2);
        }
        _ => {
            eprintln!("usage: cargo run -- [trace | broken late-result]");
            std::process::exit(2);
        }
    }
}
