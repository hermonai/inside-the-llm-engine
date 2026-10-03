//! Executable protocol model, not a GPU allocator or HTTP server.
//! One work ticket per request; bounded data, separate terminal metadata.
use super::Outcome;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    slot: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ticket {
    key: Key,
    serial: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum Delivery {
    Text(String),
    Terminal(Outcome),
}

struct Owner {
    key: Key,
    capacity: usize,
    data: VecDeque<String>,
    next_serial: u64,
    pending: Option<Ticket>,
    outcome: Option<Outcome>,
    terminal_sent: bool,
}

impl Owner {
    fn new(key: Key, capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            key,
            capacity,
            data: VecDeque::new(),
            next_serial: 0,
            pending: None,
            outcome: None,
            terminal_sent: false,
        }
    }

    fn submit(&mut self) -> Option<Ticket> {
        if self.outcome.is_some() || self.pending.is_some() || self.data.len() == self.capacity {
            return None;
        }
        let ticket = Ticket {
            key: self.key,
            serial: self.next_serial,
        };
        self.next_serial = self.next_serial.checked_add(1).expect("ticket exhaustion");
        self.pending = Some(ticket);
        Some(ticket)
    }

    // A matching retirement acknowledges finished work, even after cancellation.
    fn retire(&mut self, ticket: Ticket, text: &str) -> bool {
        if self.pending != Some(ticket) {
            return false;
        }
        self.pending = None;
        if self.outcome.is_none() {
            assert!(self.data.len() < self.capacity);
            self.data.push_back(text.into());
        }
        true
    }

    // One irreversible decision. Cancellation/failure discard unsent data;
    // natural completion drains it before publishing its terminal metadata.
    fn finish(&mut self, outcome: Outcome) -> bool {
        if self.outcome.is_some() {
            return false;
        }
        self.outcome = Some(outcome);
        if matches!(outcome, Outcome::Cancelled | Outcome::Failed) {
            self.data.clear();
        }
        true
    }

    fn receive(&mut self) -> Option<Delivery> {
        if let Some(text) = self.data.pop_front() {
            return Some(Delivery::Text(text));
        }
        if !self.terminal_sent {
            if let Some(outcome) = self.outcome {
                self.terminal_sent = true;
                return Some(Delivery::Terminal(outcome));
            }
        }
        None
    }

    // A model of device-memory reuse, not destruction of output metadata.
    fn reusable(&self) -> bool {
        self.outcome.is_some() && self.pending.is_none()
    }
}

// Deterministic event schedule: unrelated notifications restart an idle wait.
// All times are illustrative integer milliseconds. Wakes must be sorted.
fn idle_expiry(wakes: &[u64], slice: u64) -> u64 {
    assert!(slice > 0 && wakes.windows(2).all(|w| w[0] <= w[1]));
    let mut expiry = slice;
    for &wake in wakes {
        if wake >= expiry {
            break;
        }
        expiry = wake.checked_add(slice).expect("time overflow");
    }
    expiry
}

pub fn demonstrate() {
    let wakes: Vec<_> = (4..=2000).step_by(4).collect();
    println!(
        "Protocol model (illustrative integer ms): idle expiry {}, fixed deadline 100",
        idle_expiry(&wakes, 100)
    );
    let mut owner = Owner::new(
        Key {
            slot: 7,
            generation: 12,
        },
        2,
    );
    let ticket = owner.submit().unwrap();
    owner.finish(Outcome::Cancelled);
    println!(
        "cancel selected: reusable={}, delivery={:?}",
        owner.reusable(),
        owner.receive()
    );
    owner.retire(ticket, "late token");
    println!(
        "work retired: reusable={}, late delivery={:?}",
        owner.reusable(),
        owner.receive()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn owner(generation: u64) -> Owner {
        Owner::new(
            Key {
                slot: 7,
                generation,
            },
            2,
        )
    }

    #[test]
    fn cancel_can_publish_terminal_but_not_reuse_pending_work() {
        let mut o = owner(12);
        let work = o.submit().unwrap();
        assert!(o.finish(Outcome::Cancelled));
        assert_eq!(o.receive(), Some(Delivery::Terminal(Outcome::Cancelled)));
        assert!(!o.reusable());
        assert!(o.submit().is_none());
        assert!(o.retire(work, "discarded"));
        assert!(o.reusable());
        assert_eq!(o.receive(), None);
        assert!(!o.retire(work, "duplicate"));
    }

    #[test]
    fn reused_slot_rejects_old_generation_and_same_generation_old_ticket() {
        let mut old = owner(12);
        let stale = old.submit().unwrap();
        old.finish(Outcome::Cancelled);
        old.retire(stale, "");
        let mut new = owner(13);
        let current = new.submit().unwrap();
        assert!(!new.retire(stale, "wrong request"));
        assert_eq!(new.pending, Some(current));
        new.retire(current, "a");
        let second = new.submit().unwrap();
        assert!(!new.retire(current, "same generation, wrong serial"));
        assert_eq!(new.pending, Some(second));
    }

    #[test]
    fn full_data_queue_cannot_block_cancellation() {
        let mut o = owner(1);
        for s in ["a", "b"] {
            let work = o.submit().unwrap();
            o.retire(work, s);
        }
        assert!(o.submit().is_none());
        assert!(o.finish(Outcome::Cancelled));
        assert_eq!(o.receive(), Some(Delivery::Terminal(Outcome::Cancelled)));
        assert_eq!(o.receive(), None);
        assert!(o.reusable());
    }

    #[test]
    fn natural_completion_drains_prefix_before_one_terminal() {
        let mut o = owner(1);
        for s in ["a", "b"] {
            let work = o.submit().unwrap();
            o.retire(work, s);
        }
        o.finish(Outcome::MaxTokens);
        assert!(!o.finish(Outcome::Cancelled));
        assert_eq!(o.receive(), Some(Delivery::Text("a".into())));
        assert_eq!(o.receive(), Some(Delivery::Text("b".into())));
        assert_eq!(o.receive(), Some(Delivery::Terminal(Outcome::MaxTokens)));
        assert_eq!(o.receive(), None);
    }

    #[test]
    fn idle_wait_is_not_a_fixed_deadline() {
        let wakes: Vec<_> = (4..=2000).step_by(4).collect();
        assert_eq!(idle_expiry(&[], 100), 100);
        assert_eq!(idle_expiry(&wakes, 100), 2100);
        assert_eq!(idle_expiry(&[40, 80, 180], 100), 180);
        // An independently enumerated quiet-interval oracle.
        for mask in 0..256_u32 {
            let wakes: Vec<_> = (1..=8).filter(|k| mask & (1 << (k - 1)) != 0).collect();
            let expected = (3..=11)
                .find(|&t| wakes.iter().all(|&w| !(w < t && w > t - 3)))
                .unwrap();
            assert_eq!(idle_expiry(&wakes, 3), expected);
        }
    }

    #[test]
    fn all_short_command_orders_preserve_publication_and_retirement() {
        // 5^7 orders: submit, retire, cancel, EOS, receive. No clocks or sleeps.
        for encoded in 0..5_usize.pow(7) {
            let mut n = encoded;
            let mut o = owner(1);
            let mut ticket = None;
            let mut terminals = 0;
            let mut sent_terminal = false;
            for _ in 0..7 {
                match n % 5 {
                    0 => {
                        if let Some(t) = o.submit() {
                            ticket = Some(t);
                        }
                    }
                    1 => {
                        if let Some(t) = ticket {
                            o.retire(t, "x");
                        }
                    }
                    2 => {
                        o.finish(Outcome::Cancelled);
                    }
                    3 => {
                        o.finish(Outcome::EndOfSequence);
                    }
                    _ => {
                        if let Some(event) = o.receive() {
                            assert!(!sent_terminal, "output after terminal");
                            if matches!(event, Delivery::Terminal(_)) {
                                terminals += 1;
                                sent_terminal = true;
                            }
                        }
                    }
                }
                assert!(o.data.len() <= o.capacity);
                assert!(terminals <= 1);
                assert!(!o.reusable() || o.pending.is_none());
                n /= 5;
            }
            // Completed, responsive runs drain one terminal; incomplete runs
            // are not claimed to have terminated.
            if o.outcome.is_some() {
                while let Some(event) = o.receive() {
                    assert!(!sent_terminal);
                    if matches!(event, Delivery::Terminal(_)) {
                        terminals += 1;
                        sent_terminal = true;
                    }
                }
                assert_eq!(terminals, 1);
            }
        }
    }

    #[test]
    fn chapter_unit_examples_are_recomputed() {
        // Constant-cache occupancy, not a measured growing KV trace.
        assert_eq!(576 * 37, 21_312); // MiB seconds
        assert_eq!(576_f64 * 37.0 / 1024.0, 20.8125); // GiB seconds
        assert_eq!(0.5 * 37.0, 18.5); // average abandoned requests
        assert_eq!(18.5 * 576.0 / 1024.0, 10.40625); // GiB
        assert_eq!(306 * 25, 7_650); // wire bytes/s, one stream
        assert_eq!(306 * 25 * 10_000, 76_500_000); // bytes/s
        assert_eq!(64_f64 / (25.0 - 5.0), 3.2); // queue fill seconds
        assert_eq!(64 * 306, 19_584); // payload bytes, not queue allocation
    }
}
