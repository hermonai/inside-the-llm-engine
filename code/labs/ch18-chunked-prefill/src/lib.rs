//! Deterministic scheduler model for Chapter 35.
//! It models accounting and fairness, not GPU execution time.

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: usize,
    pub hit: usize,
    pub committed: usize,
    pub age: usize,
}
impl Request {
    pub fn new(prompt: usize, hit: usize) -> Self {
        assert!(hit <= prompt);
        Self {
            prompt,
            hit,
            committed: hit,
            age: 0,
        }
    }
    pub fn remaining(&self) -> usize {
        self.prompt - self.committed
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub decodes: usize,
    pub scheduled_prefill: usize,
    pub before: usize,
    pub after: usize,
    pub committed: bool,
}

pub fn schedule_step(
    r: &mut Request,
    budget: usize,
    decodes: usize,
    min_prefill_share: usize,
    succeed: bool,
) -> Step {
    let before = r.committed;
    // Decode-first, but when a prefill is waiting reserve a bounded non-zero share
    // so a persistent decode population cannot starve it forever.
    let reserve = if r.remaining() > 0 {
        min_prefill_share.min(budget)
    } else {
        0
    };
    let decode_cap = budget.saturating_sub(reserve);
    let d = decodes.min(decode_cap);
    let room = budget - d;
    let q = r.remaining().min(room);

    // Scheduling is only a reservation. Logical progress changes after success.
    if succeed {
        r.committed += q;
    }
    r.age += 1;
    Step {
        decodes: d,
        scheduled_prefill: q,
        before,
        after: r.committed,
        committed: succeed,
    }
}

pub fn run_to_prompt(
    r: &mut Request,
    budget: usize,
    decodes_each: usize,
    min_prefill_share: usize,
) -> Vec<Step> {
    assert!(
        budget > 0 && (decodes_each < budget || min_prefill_share > 0),
        "prefill must have a progress opportunity"
    );
    let mut out = Vec::new();
    while r.remaining() > 0 {
        out.push(schedule_step(
            r,
            budget,
            decodes_each,
            min_prefill_share,
            true,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_hit_reduces_compute_work() {
        let mut r = Request::new(32_000, 28_000);
        let steps = run_to_prompt(&mut r, 1024, 0, 1);
        let work: usize = steps.iter().map(|s| s.scheduled_prefill).sum();
        assert_eq!(work, 4_000);
        assert_eq!(r.committed, 32_000);
    }

    #[test]
    fn failed_step_does_not_advance_commit_boundary() {
        let mut r = Request::new(10_240, 6_144);
        let s = schedule_step(&mut r, 1024, 64, 1, false);
        assert_eq!(s.before, 6_144);
        assert_eq!(s.after, 6_144);
        assert_eq!(r.committed, 6_144);
    }

    #[test]
    fn fairness_share_advances_prefill_under_decode_saturation() {
        let mut r = Request::new(100, 0);
        let s = schedule_step(&mut r, 32, 32, 4, true);
        assert_eq!(s.decodes, 28);
        assert_eq!(s.scheduled_prefill, 4);
        assert_eq!(r.committed, 4);
    }

    #[test]
    fn conservation_oracle() {
        let mut r = Request::new(257, 17);
        let start = r.committed;
        let steps = run_to_prompt(&mut r, 31, 13, 3);
        let committed_new: usize = steps
            .iter()
            .filter(|s| s.committed)
            .map(|s| s.after - s.before)
            .sum();
        // Independent conservation statement: hit + committed new work = prompt.
        assert_eq!(start + committed_new, 257);
    }

    #[test]
    fn budget_is_never_exceeded() {
        let mut r = Request::new(500, 0);
        for _ in 0..10 {
            if r.remaining() == 0 {
                break;
            }
            let s = schedule_step(&mut r, 64, 61, 8, true);
            assert!(s.decodes + s.scheduled_prefill <= 64);
        }
    }
}
