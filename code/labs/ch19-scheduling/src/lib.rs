//! Finite-capacity scheduling model for Chapter 36.
//! The lab keeps policy small enough that invariants are inspectable by hand.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Req {
    pub id: usize,
    pub tenant: usize,
    pub blocks: usize,
    pub remaining: usize,
    pub priority: i32,
    pub age: usize,
}
#[derive(Debug)]
pub struct Sim {
    pub capacity: usize,
    pub used: usize,
    pub queue: VecDeque<Req>,
    pub running: Vec<Req>,
    pub service: Vec<usize>,
    pub completed: Vec<usize>,
    pub rejected: Vec<usize>,
    pub preemptions: usize,
}
impl Sim {
    pub fn new(capacity: usize, tenants: usize) -> Self {
        Self {
            capacity,
            used: 0,
            queue: VecDeque::new(),
            running: Vec::new(),
            service: vec![0; tenants],
            completed: vec![],
            rejected: vec![],
            preemptions: 0,
        }
    }
    pub fn submit(&mut self, r: Req, max_queue: usize) {
        if r.blocks > self.capacity || self.queue.len() >= max_queue {
            self.rejected.push(r.id);
        } else {
            self.queue.push_back(r);
        }
    }
    pub fn admit_one(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }
        // Weighted-fair teaching policy with equal weights: choose tenant with least
        // accumulated service, then oldest request in that tenant.
        let idx = self
            .queue
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| (self.service[r.tenant], std::cmp::Reverse(r.age)))
            .map(|(i, _)| i)
            .unwrap();
        let r = self.queue.remove(idx).unwrap();
        if self.used + r.blocks <= self.capacity {
            self.used += r.blocks;
            self.running.push(r);
            true
        } else {
            self.queue.push_front(r);
            false
        }
    }
    pub fn preempt_for(&mut self, needed: usize) -> bool {
        if self.capacity - self.used >= needed {
            return true;
        }
        if self.running.is_empty() {
            return false;
        }
        // Victim: lowest priority; ties choose newest/least-aged.
        let idx = self
            .running
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| (r.priority, r.age))
            .map(|(i, _)| i)
            .unwrap();
        let mut v = self.running.remove(idx);
        self.used -= v.blocks;
        v.age += 1;
        self.queue.push_front(v);
        self.preemptions += 1;
        self.capacity - self.used >= needed
    }
    pub fn tick(&mut self) {
        for r in &mut self.queue {
            r.age += 1;
        }
        if self.running.is_empty() {
            let _ = self.admit_one();
        }
        if self.running.is_empty() {
            return;
        }
        // Serve the running tenant with least service.
        let idx = self
            .running
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| self.service[r.tenant])
            .map(|(i, _)| i)
            .unwrap();
        let tenant = self.running[idx].tenant;
        self.running[idx].remaining -= 1;
        self.service[tenant] += 1;
        if self.running[idx].remaining == 0 {
            let r = self.running.remove(idx);
            self.used -= r.blocks;
            self.completed.push(r.id);
        }
        while self.admit_one() {}
    }
    pub fn invariant(&self) -> bool {
        let sum: usize = self.running.iter().map(|r| r.blocks).sum();
        sum == self.used && self.used <= self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(id: usize, t: usize, b: usize, w: usize) -> Req {
        Req {
            id,
            tenant: t,
            blocks: b,
            remaining: w,
            priority: 0,
            age: 0,
        }
    }
    #[test]
    fn never_exceeds_capacity() {
        let mut s = Sim::new(4, 2);
        s.submit(r(1, 0, 3, 3), 10);
        s.submit(r(2, 1, 3, 3), 10);
        for _ in 0..10 {
            s.tick();
            assert!(s.invariant());
        }
    }
    #[test]
    fn bounded_queue_rejects_early() {
        let mut s = Sim::new(2, 1);
        s.submit(r(1, 0, 1, 1), 1);
        s.submit(r(2, 0, 1, 1), 1);
        assert_eq!(s.rejected, vec![2]);
        assert_eq!(s.used, 0); // rejection happened before allocation
    }
    #[test]
    fn preemption_reclaims_before_retry() {
        let mut s = Sim::new(4, 2);
        s.submit(r(1, 0, 4, 5), 10);
        assert!(s.admit_one());
        assert!(s.preempt_for(3));
        assert_eq!(s.used, 0);
        assert_eq!(s.preemptions, 1);
        assert!(s.invariant());
    }
    #[test]
    fn service_conservation_for_completed_requests() {
        let mut s = Sim::new(4, 2);
        s.submit(r(1, 0, 2, 3), 10);
        s.submit(r(2, 1, 2, 2), 10);
        for _ in 0..10 {
            s.tick();
        }
        assert!(s.completed.contains(&1) && s.completed.contains(&2));
        assert_eq!(s.service.iter().sum::<usize>(), 5);
    }
    #[test]
    fn both_backlogged_tenants_receive_service() {
        let mut s = Sim::new(4, 2);
        s.submit(r(1, 0, 2, 20), 10);
        s.submit(r(2, 1, 2, 20), 10);
        for _ in 0..10 {
            s.tick();
        }
        assert!(s.service[0] > 0 && s.service[1] > 0);
        assert!((s.service[0] as isize - s.service[1] as isize).abs() <= 1);
    }
}
