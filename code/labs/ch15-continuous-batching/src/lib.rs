use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terminal {
    Completed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct RequestSpec {
    pub id: &'static str,
    pub arrival: u64,
    pub prompt: u32,
    pub output: u32,
    pub cancel_at: Option<u64>,
}

#[derive(Clone, Debug)]
struct Request {
    spec: RequestSpec,
    prompt_done: u32,
    output_done: u32,
    slot: Option<usize>,
    generation: u64,
    outstanding: u32,
    terminal: Option<Terminal>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ticket {
    pub id: &'static str,
    pub slot: usize,
    pub generation: u64,
    pub tokens: u32,
    pub serial: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Arrive(&'static str),
    Admit(&'static str, usize, u64),
    Schedule(&'static str, u32),
    Terminal(&'static str, Terminal),
    Retire(&'static str, usize, u64),
    RejectLate(&'static str, usize, u64),
}

pub struct Scheduler {
    max_seqs: usize,
    max_tokens: u32,
    tick: u64,
    next_generation: u64,
    next_ticket: u64,
    last_planned_tick: Option<u64>,
    pending: BTreeMap<u64, Ticket>,
    specs: Vec<RequestSpec>,
    waiting: VecDeque<&'static str>,
    reqs: BTreeMap<&'static str, Request>,
    slots: Vec<Option<(&'static str, u64)>>,
    events: Vec<(u64, Event)>,
}

impl Scheduler {
    /// FCFS admission with slot-order scheduling; tickets finish within a tick.
    pub fn new(max_seqs: usize, max_tokens: u32, specs: Vec<RequestSpec>) -> Self {
        assert!(max_seqs > 0 && max_tokens > 0);
        let mut ids = std::collections::BTreeSet::new();
        for spec in &specs {
            assert!(ids.insert(spec.id), "request IDs must be unique");
            assert!(
                spec.prompt > 0 && spec.prompt <= max_tokens,
                "unchunked prompts must fit the iteration token budget"
            );
            assert!(
                spec.output > 0,
                "this teaching model requires nonempty output"
            );
            assert!(
                spec.cancel_at.is_none_or(|t| t >= spec.arrival),
                "cancellation cannot precede arrival"
            );
        }
        Self {
            max_seqs,
            max_tokens,
            tick: 0,
            next_generation: 1,
            next_ticket: 1,
            last_planned_tick: None,
            pending: BTreeMap::new(),
            specs,
            waiting: VecDeque::new(),
            reqs: BTreeMap::new(),
            slots: vec![None; max_seqs],
            events: Vec::new(),
        }
    }

    fn push(&mut self, e: Event) {
        self.events.push((self.tick, e));
    }

    fn arrivals_and_cancels(&mut self) {
        let arrivals: Vec<_> = self
            .specs
            .iter()
            .filter(|s| s.arrival == self.tick)
            .cloned()
            .collect();
        for spec in arrivals {
            self.push(Event::Arrive(spec.id));
            self.waiting.push_back(spec.id);
            self.reqs.insert(
                spec.id,
                Request {
                    spec,
                    prompt_done: 0,
                    output_done: 0,
                    slot: None,
                    generation: 0,
                    outstanding: 0,
                    terminal: None,
                },
            );
        }
        let ids: Vec<_> = self.reqs.keys().copied().collect();
        for id in ids {
            let cancel = self.reqs[&id].spec.cancel_at == Some(self.tick);
            if cancel && self.reqs[&id].terminal.is_none() {
                self.reqs.get_mut(&id).unwrap().terminal = Some(Terminal::Cancelled);
                self.push(Event::Terminal(id, Terminal::Cancelled));
            }
        }
    }

    fn retire_safe(&mut self) {
        let ids: Vec<_> = self.reqs.keys().copied().collect();
        for id in ids {
            let r = &self.reqs[&id];
            if r.terminal.is_some() && r.outstanding == 0 {
                if let Some(slot) = r.slot {
                    let gen = r.generation;
                    self.slots[slot] = None;
                    self.reqs.get_mut(&id).unwrap().slot = None;
                    self.push(Event::Retire(id, slot, gen));
                }
            }
        }
    }

    fn admit(&mut self) {
        loop {
            let Some(slot) = self.slots.iter().position(Option::is_none) else {
                break;
            };
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            if self.reqs[&id].terminal.is_some() {
                continue;
            }
            let gen = self.next_generation;
            self.next_generation += 1;
            self.slots[slot] = Some((id, gen));
            let r = self.reqs.get_mut(&id).unwrap();
            r.slot = Some(slot);
            r.generation = gen;
            self.push(Event::Admit(id, slot, gen));
        }
    }

    pub fn plan(&mut self) -> Vec<Ticket> {
        assert_ne!(
            self.last_planned_tick,
            Some(self.tick),
            "one plan per tick prevents double spending iteration budgets"
        );
        self.last_planned_tick = Some(self.tick);
        let mut token_budget = self.max_tokens;
        let mut seq_budget = self.max_seqs;
        let mut out = Vec::new();
        let owners: Vec<_> = self.slots.iter().flatten().copied().collect();

        for (id, gen) in owners {
            if seq_budget == 0 || token_budget == 0 {
                break;
            }
            let r = &self.reqs[&id];
            if r.terminal.is_some() || r.outstanding > 0 {
                continue;
            }

            let remaining_prompt = r.spec.prompt - r.prompt_done;
            // This chapter intentionally does not implement chunked prefill.
            let q = if remaining_prompt > 0 {
                if remaining_prompt <= token_budget {
                    remaining_prompt
                } else {
                    0
                }
            } else if r.output_done < r.spec.output {
                1
            } else {
                0
            };

            if q == 0 {
                continue;
            }
            let slot = r.slot.unwrap();
            token_budget -= q;
            seq_budget -= 1;
            self.reqs.get_mut(&id).unwrap().outstanding += 1;
            self.push(Event::Schedule(id, q));
            let ticket = Ticket {
                id,
                slot,
                generation: gen,
                tokens: q,
                serial: self.next_ticket,
            };
            self.next_ticket += 1;
            self.pending.insert(ticket.serial, ticket.clone());
            out.push(ticket);
        }
        out
    }

    pub fn complete(&mut self, t: Ticket) {
        let current = self.slots.get(t.slot).and_then(|x| *x);
        if current != Some((t.id, t.generation)) || self.pending.get(&t.serial) != Some(&t) {
            self.push(Event::RejectLate(t.id, t.slot, t.generation));
            return;
        }
        self.pending.remove(&t.serial);
        let r = self.reqs.get_mut(t.id).unwrap();
        r.outstanding -= 1;
        if r.terminal.is_some() {
            return;
        }

        let rem_prompt = r.spec.prompt - r.prompt_done;
        if rem_prompt > 0 {
            assert_eq!(t.tokens, rem_prompt);
            r.prompt_done += t.tokens;
        } else {
            assert_eq!(t.tokens, 1);
            r.output_done += 1;
            if r.output_done == r.spec.output {
                r.terminal = Some(Terminal::Completed);
                self.push(Event::Terminal(t.id, Terminal::Completed));
            }
        }
    }

    pub fn step(&mut self) {
        self.arrivals_and_cancels();
        self.retire_safe();
        self.admit();
        let tickets = self.plan();
        // Deterministic teaching device: all tickets complete before next tick.
        for t in tickets {
            self.complete(t);
        }
        self.tick += 1;
    }

    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    pub fn events(&self) -> &[(u64, Event)] {
        &self.events
    }

    pub fn assert_independent_accounting(&self) {
        // Oracle: reconstruct scheduled token/sequence totals from the trace only.
        let mut by_tick: BTreeMap<u64, (usize, u32)> = BTreeMap::new();
        for (tick, event) in &self.events {
            if let Event::Schedule(_, q) = event {
                let e = by_tick.entry(*tick).or_insert((0, 0));
                e.0 += 1;
                e.1 += *q;
            }
        }
        for (tick, (seqs, tokens)) in by_tick {
            assert!(
                seqs <= self.max_seqs,
                "sequence budget exceeded at tick {tick}"
            );
            assert!(
                tokens <= self.max_tokens,
                "token budget exceeded at tick {tick}"
            );
        }
    }
}

pub fn cohort() -> Vec<RequestSpec> {
    vec![
        RequestSpec {
            id: "A",
            arrival: 0,
            prompt: 3,
            output: 4,
            cancel_at: None,
        },
        RequestSpec {
            id: "B",
            arrival: 0,
            prompt: 1,
            output: 2,
            cancel_at: None,
        },
        RequestSpec {
            id: "C",
            arrival: 1,
            prompt: 2,
            output: 3,
            cancel_at: None,
        },
        RequestSpec {
            id: "D",
            arrival: 3,
            prompt: 1,
            output: 2,
            cancel_at: Some(5),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_cohort_respects_budgets() {
        let mut s = Scheduler::new(2, 3, cohort());
        s.run(12);
        s.assert_independent_accounting();
    }

    #[test]
    fn first_tick_uses_token_budget_not_only_sequence_budget() {
        let mut s = Scheduler::new(2, 3, cohort());
        s.step();
        let scheduled: Vec<_> = s
            .events()
            .iter()
            .filter_map(|(_, e)| {
                if let Event::Schedule(id, q) = e {
                    Some((*id, *q))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(scheduled, vec![("A", 3)]);
    }

    #[test]
    fn waiting_cancellation_needs_no_slot_retirement() {
        let mut s = Scheduler::new(2, 3, cohort());
        s.run(12);
        assert!(s
            .events()
            .iter()
            .any(|(_, e)| *e == Event::Terminal("D", Terminal::Cancelled)));
        assert!(!s
            .events()
            .iter()
            .any(|(_, e)| matches!(e, Event::Retire("D", _, _))));
    }

    #[test]
    fn active_cancellation_waits_for_matching_completion() {
        let mut s = Scheduler::new(
            1,
            3,
            vec![RequestSpec {
                id: "X",
                arrival: 0,
                prompt: 1,
                output: 2,
                cancel_at: Some(1),
            }],
        );
        s.arrivals_and_cancels();
        s.admit();
        let ticket = s.plan().pop().unwrap();
        s.tick = 1;
        s.arrivals_and_cancels();
        s.retire_safe();
        assert!(s.slots[0].is_some());
        s.complete(ticket);
        assert_eq!(s.reqs["X"].prompt_done, 0);
        s.retire_safe();
        assert!(s.slots[0].is_none());
    }

    #[test]
    fn duplicate_reply_is_rejected_even_before_slot_reuse() {
        let mut s = Scheduler::new(2, 3, cohort());
        s.arrivals_and_cancels();
        s.admit();
        let ticket = s.plan().pop().unwrap();
        s.complete(ticket.clone());
        s.complete(ticket);
        assert_eq!(s.reqs["A"].prompt_done, 3);
        assert_eq!(s.reqs["A"].outstanding, 0);
        assert!(matches!(
            s.events.last().unwrap().1,
            Event::RejectLate("A", _, _)
        ));
    }

    #[test]
    #[should_panic(expected = "request IDs must be unique")]
    fn duplicate_ids_are_rejected() {
        let mut specs = cohort();
        specs.push(specs[0].clone());
        Scheduler::new(2, 3, specs);
    }

    #[test]
    #[should_panic(expected = "unchunked prompts must fit")]
    fn oversized_unchunked_prompt_is_rejected() {
        Scheduler::new(2, 2, cohort());
    }

    #[test]
    #[should_panic(expected = "one plan per tick")]
    fn repeated_plan_cannot_reset_iteration_budget() {
        let mut s = Scheduler::new(2, 3, cohort());
        s.step();
        s.tick = 0;
        s.plan();
    }

    #[test]
    fn late_result_is_rejected_after_generation_changes() {
        let mut s = Scheduler::new(
            1,
            3,
            vec![
                RequestSpec {
                    id: "X",
                    arrival: 0,
                    prompt: 1,
                    output: 1,
                    cancel_at: Some(1),
                },
                RequestSpec {
                    id: "Y",
                    arrival: 2,
                    prompt: 1,
                    output: 1,
                    cancel_at: None,
                },
            ],
        );
        s.arrivals_and_cancels();
        s.admit();
        let old = s.plan().pop().unwrap();
        s.complete(old.clone());
        s.tick = 1;
        s.arrivals_and_cancels();
        s.retire_safe();
        s.tick = 2;
        s.arrivals_and_cancels();
        s.retire_safe();
        s.admit();
        s.complete(old);
        assert!(s
            .events()
            .iter()
            .any(|(_, e)| matches!(e, Event::RejectLate("X", _, _))));
    }

    #[test]
    fn capacity_never_admits_more_than_slots() {
        let specs = vec![
            RequestSpec {
                id: "A",
                arrival: 0,
                prompt: 1,
                output: 5,
                cancel_at: None,
            },
            RequestSpec {
                id: "B",
                arrival: 0,
                prompt: 1,
                output: 5,
                cancel_at: None,
            },
            RequestSpec {
                id: "C",
                arrival: 0,
                prompt: 1,
                output: 5,
                cancel_at: None,
            },
        ];
        let mut s = Scheduler::new(2, 8, specs);
        s.step();
        let admits = s
            .events()
            .iter()
            .filter(|(t, e)| *t == 0 && matches!(e, Event::Admit(_, _, _)))
            .count();
        assert_eq!(admits, 2);
    }
}
