//! Deterministic break-even/control lab for Chapter 41.
//! No wall-clock benchmark is hidden here: every timing value is an explicit trace
//! input. This makes the controller reproducible and the break-even oracle independent.

#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub budget: usize,
    pub baseline_ms: f64,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub overhead_ms: f64,
    pub committed: f64,
    pub kv_pressure: f64,
    pub slo_miss: bool,
}
/// Independent raw inequality: speedup = ordinary time needed for the same committed
/// progress divided by speculative iteration time.
pub fn oracle_speedup(o: Obs) -> f64 {
    let spec = o.draft_ms + o.verify_ms + o.overhead_ms;
    (o.committed * o.baseline_ms) / spec
}
pub fn profitable(o: Obs) -> bool {
    oracle_speedup(o) > 1.0 && o.kv_pressure < 0.90 && !o.slo_miss
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub budget: usize,
    pub ewma: f64,
    pub initialized: bool,
}
impl Arm {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            ewma: 0.0,
            initialized: false,
        }
    }
    pub fn observe(&mut self, score: f64, alpha: f64) {
        assert!(score.is_finite() && alpha.is_finite() && (0.0..=1.0).contains(&alpha));
        self.ewma = if self.initialized {
            alpha * score + (1.0 - alpha) * self.ewma
        } else {
            score
        };
        self.initialized = true;
    }
}
/// Controller score deliberately penalizes optional speculation under KV/SLO pressure.
/// Budget zero is the ordinary-decode arm and should be fed baseline observations.
pub fn score(o: Obs) -> f64 {
    let mut s = oracle_speedup(o);
    if o.kv_pressure >= 0.90 {
        s -= 2.0;
    }
    if o.slo_miss {
        s -= 2.0;
    }
    s
}
/// Hysteresis: switch only if improvement exceeds margin times current magnitude.
/// Absolute magnitude also handles negative resource-penalized scores correctly.
/// Production controllers also need minimum samples/exploration; this teaching function
/// isolates the anti-oscillation rule.
pub fn choose(current: &Arm, arms: &[Arm], margin: f64) -> usize {
    assert!(margin.is_finite() && margin >= 0.0);
    let mut best = current;
    for a in arms.iter().filter(|a| a.initialized) {
        if a.ewma > best.ewma {
            best = a;
        }
    }
    if best.budget != current.budget
        && best.ewma > current.ewma + margin * current.ewma.abs().max(f64::EPSILON)
    {
        best.budget
    } else {
        current.budget
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ewma_initialization_and_blend_match_hand_calculation() {
        let mut arm = Arm::new(4);
        arm.observe(1.0, 0.5);
        assert_eq!(arm.ewma, 1.0);
        arm.observe(3.0, 0.5);
        assert_eq!(arm.ewma, 2.0);
    }
    #[test]
    fn hysteresis_also_handles_negative_penalized_scores() {
        let current = Arm {
            budget: 0,
            ewma: -1.0,
            initialized: true,
        };
        let small = Arm {
            budget: 4,
            ewma: -0.95,
            initialized: true,
        };
        let large = Arm {
            budget: 4,
            ewma: -0.8,
            initialized: true,
        };
        assert_eq!(choose(&current, &[small], 0.1), 0);
        assert_eq!(choose(&current, &[large], 0.1), 4);
    }
    fn o(b: usize, base: f64, d: f64, v: f64, k: f64, c: f64, kv: f64) -> Obs {
        Obs {
            budget: b,
            baseline_ms: base,
            draft_ms: d,
            verify_ms: v,
            overhead_ms: k,
            committed: c,
            kv_pressure: kv,
            slo_miss: false,
        }
    }
    #[test]
    fn low_load_speculation_wins() {
        assert!(oracle_speedup(o(4, 100., 20., 120., 10., 4., 0.5)) > 1.0);
    }
    #[test]
    fn expensive_drafter_loses() {
        assert!(oracle_speedup(o(4, 100., 180., 250., 20., 4., 0.5)) < 1.0);
    }
    #[test]
    fn perfect_acceptance_can_still_lose() {
        assert!(oracle_speedup(o(4, 100., 200., 350., 0., 5., 0.5)) < 1.0);
    }
    #[test]
    fn kv_pressure_disables_optional_work() {
        let x = o(4, 100., 10., 100., 0., 4., 0.95);
        assert!(!profitable(x));
    }
    #[test]
    fn concurrency_can_reduce_gain_without_acceptance_change() {
        let one = o(4, 100., 0., 120., 0., 5., 0.5);
        let busy = o(4, 100., 0., 300., 0., 5., 0.5);
        assert_eq!(one.committed, busy.committed);
        assert!(oracle_speedup(one) > oracle_speedup(busy));
    }
    #[test]
    fn hysteresis_prevents_small_switch() {
        let cur = Arm {
            budget: 0,
            ewma: 1.0,
            initialized: true,
        };
        let arms = vec![
            cur.clone(),
            Arm {
                budget: 4,
                ewma: 1.03,
                initialized: true,
            },
        ];
        assert_eq!(choose(&cur, &arms, 0.05), 0);
    }
}
