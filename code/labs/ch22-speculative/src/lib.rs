//! Deterministic Chapter 39 lab.
//! Part 1 proves exact one-position speculative probability mass by enumeration.
//! Part 2 models speculative frontiers and rollback without pretending to be an LLM.

pub fn exact_output_mass(p: &[f64], q: &[f64]) -> Vec<f64> {
    assert_eq!(p.len(), q.len());
    let accept: Vec<f64> = p.iter().zip(q).map(|(a, b)| a.min(*b)).collect();
    let beta: f64 = accept.iter().sum();
    let residual: Vec<f64> = p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)).collect();
    let z: f64 = residual.iter().sum();
    let mut out = accept;
    if 1.0 - beta > 1e-14 {
        assert!(z > 0.0);
        for i in 0..out.len() {
            out[i] += (1.0 - beta) * residual[i] / z;
        }
    }
    out
}
/// Deliberately wrong algorithm: after rejection, resample from unadjusted target p.
pub fn wrong_output_mass(p: &[f64], q: &[f64]) -> Vec<f64> {
    let accept: Vec<f64> = p.iter().zip(q).map(|(a, b)| a.min(*b)).collect();
    let beta: f64 = accept.iter().sum();
    accept
        .iter()
        .zip(p)
        .map(|(a, pi)| a + (1.0 - beta) * pi)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frontiers {
    pub committed: usize,
    pub drafted: usize,
    pub evaluated: usize,
    pub published: usize,
}
impl Frontiers {
    pub fn new(n: usize) -> Self {
        Self {
            committed: n,
            drafted: n,
            evaluated: n,
            published: n,
        }
    }
    pub fn draft(&mut self, n: usize) {
        self.drafted = self.committed + n;
    }
    pub fn evaluate_drafts(&mut self) {
        self.evaluated = self.drafted;
    }
    /// Commit `accepted` draft tokens plus one target correction/bonus token.
    /// The correction/bonus is committed but has not been fed into the target.
    /// Its physical KV must be materialized before it is used as cached history.
    pub fn reconcile(&mut self, accepted: usize) -> usize {
        assert!(self.committed + accepted <= self.drafted);
        let new_c = self.committed + accepted + 1;
        self.committed = new_c;
        self.evaluated = new_c - 1;
        self.drafted = new_c;
        new_c
    }
    pub fn publish_to(&mut self, n: usize) -> bool {
        if n > self.committed || n < self.published {
            return false;
        }
        self.published = n;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
    #[test]
    fn exact_mass_equals_target_token_by_token() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.6, 0.1, 0.3];
        let o = exact_output_mass(&p, &q);
        for i in 0..p.len() {
            close(o[i], p[i]);
        }
    }
    #[test]
    fn exact_mass_handles_crossing_distributions() {
        let p = [0.4, 0.35, 0.25];
        let q = [0.2, 0.5, 0.3];
        let o = exact_output_mass(&p, &q);
        for i in 0..p.len() {
            close(o[i], p[i]);
        }
    }
    #[test]
    fn wrong_residual_is_detected_without_monte_carlo() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.6, 0.1, 0.3];
        let o = wrong_output_mass(&p, &q);
        assert!((o[0] - p[0]).abs() > 0.05);
    }
    #[test]
    fn rejection_collapses_tentative_frontier() {
        let mut f = Frontiers::new(80);
        f.draft(5);
        f.evaluate_drafts();
        assert_eq!(f.evaluated, 85);
        f.reconcile(3);
        assert_eq!(f.committed, 84);
        assert_eq!(f.evaluated, 83);
        assert_eq!(f.drafted, 84);
    }
    #[test]
    fn cannot_publish_uncommitted_drafts() {
        let mut f = Frontiers::new(10);
        f.draft(4);
        f.evaluate_drafts();
        assert!(!f.publish_to(14));
        assert_eq!(f.published, 10);
        f.reconcile(1);
        assert!(f.publish_to(12));
        assert!(!f.publish_to(11));
    }
}
