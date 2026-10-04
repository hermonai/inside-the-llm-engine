use ch24_break_even::*;
fn main() {
    let trace = [
        Obs {
            budget: 4,
            baseline_ms: 100.,
            draft_ms: 20.,
            verify_ms: 120.,
            overhead_ms: 10.,
            committed: 4.,
            kv_pressure: 0.50,
            slo_miss: false,
        },
        Obs {
            budget: 4,
            baseline_ms: 100.,
            draft_ms: 180.,
            verify_ms: 250.,
            overhead_ms: 20.,
            committed: 4.,
            kv_pressure: 0.50,
            slo_miss: false,
        },
        Obs {
            budget: 4,
            baseline_ms: 100.,
            draft_ms: 0.,
            verify_ms: 300.,
            overhead_ms: 0.,
            committed: 5.,
            kv_pressure: 0.95,
            slo_miss: false,
        },
    ];
    let mut baseline = Arm::new(0);
    baseline.observe(1.0, 0.5);
    let mut speculative = Arm::new(4);
    for x in trace {
        speculative.observe(score(x), 0.5);
        let selected = choose(&baseline, &[baseline.clone(), speculative.clone()], 0.05);
        println!(
            "budget={} speedup={:0.3} profitable={}",
            x.budget,
            oracle_speedup(x),
            profitable(x)
        );
        println!("EWMA={:.3} selected_budget={selected}", speculative.ewma);
    }
}
