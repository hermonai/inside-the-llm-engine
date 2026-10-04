//! Teaching prefix cache for Chapter 34.
//!
//! The implementation intentionally separates *candidate lookup* from *proof*.
//! A weak hash is used so tests can manufacture collisions. Correctness comes from
//! exact namespace + token-prefix verification, not from trusting the digest.

use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub namespace: u64,
    pub tokens: Vec<u32>,
    pub cost_blocks: usize,
    pub last_touch: u64,
    pub weak_hash: u64,
}

#[derive(Debug)]
pub struct PrefixCache {
    capacity_blocks: usize,
    used_blocks: usize,
    clock: u64,
    entries: VecDeque<Entry>,
}

fn weak_hash(namespace: u64, tokens: &[u32]) -> u64 {
    // Deliberately collision-prone: never use this as proof of equality.
    tokens
        .iter()
        .fold(namespace & 0xff, |h, t| (h + (*t as u64)) & 0xff)
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixCache {
    pub fn new(capacity_blocks: usize) -> Self {
        Self {
            capacity_blocks,
            used_blocks: 0,
            clock: 0,
            entries: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, namespace: u64, tokens: Vec<u32>, cost_blocks: usize) {
        assert!(
            cost_blocks <= self.capacity_blocks,
            "entry exceeds cache capacity"
        );
        self.clock += 1;
        while self.used_blocks + cost_blocks > self.capacity_blocks {
            // Teaching LRU: choose the least recently touched unpinned entry.
            // Chapter 33 generation-bearing block handles are the next exercise.
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_touch)
                .map(|(i, _)| i)
                .expect("capacity pressure with no evictable entry");
            let old = self.entries.remove(victim).unwrap();
            self.used_blocks -= old.cost_blocks;
        }
        let h = weak_hash(namespace, &tokens);
        self.entries.push_back(Entry {
            namespace,
            tokens,
            cost_blocks,
            last_touch: self.clock,
            weak_hash: h,
        });
        self.used_blocks += cost_blocks;
    }

    pub fn longest_hit(&mut self, namespace: u64, request: &[u32]) -> usize {
        self.clock += 1;
        let mut best = 0usize;
        let mut best_idx = None;
        for (i, e) in self.entries.iter().enumerate() {
            // Namespace equality is authorization, not merely an optimization key.
            if e.namespace != namespace {
                continue;
            }
            let n = lcp(&e.tokens, request);
            if n > best {
                best = n;
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            self.entries[i].last_touch = self.clock;
        }
        best
    }

    pub fn candidate_by_weak_hash(&self, namespace: u64, tokens: &[u32]) -> Option<&Entry> {
        let h = weak_hash(namespace, tokens);
        self.entries.iter().find(|e| e.weak_hash == h)
    }

    pub fn verified_exact_hit(&self, namespace: u64, tokens: &[u32]) -> bool {
        // The hash narrows candidates; exact fields prove identity.
        let h = weak_hash(namespace, tokens);
        self.entries
            .iter()
            .filter(|e| e.weak_hash == h)
            .any(|e| e.namespace == namespace && e.tokens == tokens)
    }

    pub fn used_blocks(&self) -> usize {
        self.used_blocks
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_namespace_reuses_longest_prefix() {
        let mut c = PrefixCache::new(8);
        c.insert(7, vec![1, 2, 3, 4, 5, 6], 2);
        assert_eq!(c.longest_hit(7, &[1, 2, 3, 4, 9]), 4);
    }

    #[test]
    fn tenants_are_isolated_even_for_identical_tokens() {
        let mut c = PrefixCache::new(8);
        c.insert(7, vec![1, 2, 3, 4], 1);
        assert_eq!(c.longest_hit(8, &[1, 2, 3, 4]), 0);
    }

    #[test]
    fn weak_hash_collision_is_not_accepted_as_equality() {
        let mut c = PrefixCache::new(8);
        // [1,2] and [0,3] have the same teaching hash under the same namespace.
        c.insert(1, vec![1, 2], 1);
        assert!(c.candidate_by_weak_hash(1, &[0, 3]).is_some());
        assert!(!c.verified_exact_hit(1, &[0, 3]));
    }

    #[test]
    fn lru_pressure_evicts_old_entry() {
        let mut c = PrefixCache::new(2);
        c.insert(1, vec![1, 1], 1);
        c.insert(1, vec![2, 2], 1);
        assert_eq!(c.longest_hit(1, &[2, 2]), 2); // refresh second
        c.insert(1, vec![3, 3], 1);
        assert_eq!(c.longest_hit(1, &[1, 1]), 0);
        assert_eq!(c.longest_hit(1, &[2, 2]), 2);
    }

    #[test]
    fn cache_matches_independent_flat_oracle() {
        let resident = vec![
            (5u64, vec![1, 2, 3, 4]),
            (5, vec![1, 2, 8]),
            (6, vec![1, 2, 3, 4]),
        ];
        let mut c = PrefixCache::new(10);
        for (ns, t) in resident.clone() {
            c.insert(ns, t, 1);
        }
        let req = vec![1, 2, 3, 9];
        let oracle = resident
            .iter()
            .filter(|(ns, _)| *ns == 5)
            .map(|(_, t)| {
                (0..=t.len().min(req.len()))
                    .filter(|&n| t[..n] == req[..n])
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap_or(0);
        assert_eq!(c.longest_hit(5, &req), oracle);
    }

    #[test]
    fn collision_does_not_hide_a_later_exact_match() {
        let mut c = PrefixCache::new(8);
        c.insert(1, vec![0, 3], 1);
        c.insert(1, vec![1, 2], 1);
        assert!(c.verified_exact_hit(1, &[1, 2]));
    }
}
