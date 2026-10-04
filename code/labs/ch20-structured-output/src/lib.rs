//! Tiny tokenizer-aware constraint engine for Chapter 37.
//! Language: exactly {"ok":true} or {"ok":false}.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State(pub usize);

const TRUE: &[u8] = br#"{"ok":true}"#;
const FALSE: &[u8] = br#"{"ok":false}"#;

fn prefixes() -> Vec<Vec<u8>> {
    let mut p = Vec::new();
    for s in [TRUE, FALSE] {
        for i in 0..=s.len() {
            let x = s[..i].to_vec();
            if !p.contains(&x) {
                p.push(x);
            }
        }
    }
    p.sort();
    p
}

#[derive(Clone)]
pub struct Parser {
    pref: Vec<Vec<u8>>,
    bytes: Vec<u8>,
}
impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}
impl Parser {
    pub fn new() -> Self {
        Self {
            pref: prefixes(),
            bytes: vec![],
        }
    }
    pub fn state(&self) -> State {
        State(
            self.pref
                .iter()
                .position(|p| p == &self.bytes)
                .expect("reachable prefix"),
        )
    }
    pub fn feed(&mut self, token: &[u8]) -> bool {
        let mut n = self.bytes.clone();
        n.extend_from_slice(token);
        if self.pref.iter().any(|p| p == &n) {
            self.bytes = n;
            true
        } else {
            false
        }
    }
    pub fn accepting(&self) -> bool {
        self.bytes == TRUE || self.bytes == FALSE
    }
    pub fn committed(&self) -> &[u8] {
        &self.bytes
    }
}

pub fn vocab() -> Vec<&'static [u8]> {
    vec![
        br#"{"ok":"#,
        b"true",
        b"false",
        b"tr",
        b"ue}",
        b"fa",
        b"lse}",
        b"null",
        b"}",
    ]
}

pub fn oracle_mask(p: &Parser) -> Vec<usize> {
    vocab()
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let mut q = p.clone();
            if q.feed(t) {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

pub struct CachedMask {
    cache: HashMap<State, Vec<usize>>,
}
impl Default for CachedMask {
    fn default() -> Self {
        Self::new()
    }
}
impl CachedMask {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    pub fn mask(&mut self, p: &Parser) -> Vec<usize> {
        let s = p.state();
        if let Some(v) = self.cache.get(&s) {
            return v.clone();
        }
        let v = oracle_mask(p); // teaching cache; production builds transitions more efficiently.
        self.cache.insert(s, v.clone());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Independent of Parser::feed and its precomputed prefix-state table.
    fn reference_mask(p: &Parser) -> Vec<usize> {
        vocab()
            .iter()
            .enumerate()
            .filter_map(|(i, token)| {
                let candidate = [p.committed(), *token].concat();
                [TRUE, FALSE]
                    .iter()
                    .any(|s| s.starts_with(&candidate))
                    .then_some(i)
            })
            .collect()
    }
    #[test]
    fn optimized_matches_independent_transition_oracle() {
        let mut c = CachedMask::new();
        for target in [TRUE, FALSE] {
            let mut p = Parser::new();
            assert_eq!(c.mask(&p), reference_mask(&p));
            for b in target {
                assert!(p.feed(&[*b]));
                assert_eq!(c.mask(&p), reference_mask(&p));
            }
        }
    }
    #[test]
    fn whole_token_suffix_is_checked() {
        let mut p = Parser::new();
        assert!(p.feed(br#"{"ok":"#));
        assert!(!p.feed(b"trueX")); // legal prefix "true", illegal suffix X
    }
    #[test]
    fn eos_only_at_accepting_state() {
        let mut p = Parser::new();
        assert!(!p.accepting());
        assert!(p.feed(br#"{"ok":"#));
        assert!(!p.accepting());
        assert!(p.feed(b"true"));
        assert!(!p.accepting());
        assert!(p.feed(b"}"));
        assert!(p.accepting());
    }
    #[test]
    fn rollback_replays_committed_bytes() {
        let mut committed = Parser::new();
        assert!(committed.feed(br#"{"ok":"#));
        let checkpoint = committed.committed().to_vec();
        let mut branch = committed.clone();
        assert!(branch.feed(b"tr"));
        assert!(branch.feed(b"ue}"));
        assert!(branch.accepting());
        let mut restored = Parser::new();
        assert!(restored.feed(&checkpoint));
        assert_eq!(restored.committed(), committed.committed());
        assert!(!restored.accepting());
    }
}
