//! Chapter 38 teaching lab: mixed LoRA arithmetic and safe adapter residency.
//! Matrices are row-major. The dimensions are intentionally tiny so every value can
//! be checked without external numerical libraries.

#[derive(Clone, Debug)]
pub struct Adapter {
    pub id: u64,
    pub rank: usize,
    pub scale: f64,
    pub a: Vec<f64>, // rank x din
    pub b: Vec<f64>, // dout x rank
}
fn matvec(m: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), cols);
    assert_eq!(m.len(), rows * cols);
    (0..rows)
        .map(|r| (0..cols).map(|c| m[r * cols + c] * x[c]).sum())
        .collect()
}
pub fn dynamic(base: &[f64], dout: usize, din: usize, x: &[f64], ad: Option<&Adapter>) -> Vec<f64> {
    let mut y = matvec(base, dout, din, x);
    if let Some(a) = ad {
        let h = matvec(&a.a, a.rank, din, x);
        let d = matvec(&a.b, dout, a.rank, &h);
        for i in 0..dout {
            y[i] += a.scale * d[i];
        }
    }
    y
}
/// Independent oracle: materialize W + s B A first, then execute one dense matvec.
/// This intentionally does not call `dynamic`, so a broken low-rank execution path
/// can disagree with the oracle.
pub fn merged_oracle(
    base: &[f64],
    dout: usize,
    din: usize,
    x: &[f64],
    ad: Option<&Adapter>,
) -> Vec<f64> {
    let mut w = base.to_vec();
    if let Some(a) = ad {
        for o in 0..dout {
            for i in 0..din {
                let mut v = 0.0;
                for k in 0..a.rank {
                    v += a.b[o * a.rank + k] * a.a[k * din + i];
                }
                w[o * din + i] += a.scale * v;
            }
        }
    }
    matvec(&w, dout, din, x)
}
pub fn mixed_batch(
    base: &[f64],
    dout: usize,
    din: usize,
    rows: &[Vec<f64>],
    adapter_index: &[Option<usize>],
    ads: &[Adapter],
) -> Vec<Vec<f64>> {
    assert_eq!(rows.len(), adapter_index.len());
    rows.iter()
        .zip(adapter_index)
        .map(|(x, idx)| dynamic(base, dout, din, x, idx.map(|j| &ads[j])))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle {
    pub slot: usize,
    pub generation: u64,
    pub adapter_id: u64,
}
#[derive(Clone, Debug)]
struct Slot {
    generation: u64,
    adapter_id: Option<u64>,
    pins: usize,
}
pub struct Residency {
    slots: Vec<Slot>,
}
impl Residency {
    pub fn new(n: usize) -> Self {
        Self {
            slots: (0..n)
                .map(|_| Slot {
                    generation: 0,
                    adapter_id: None,
                    pins: 0,
                })
                .collect(),
        }
    }
    pub fn load(&mut self, slot: usize, id: u64) -> Handle {
        assert_eq!(self.slots[slot].pins, 0, "cannot replace pinned adapter");
        self.slots[slot].generation += 1;
        self.slots[slot].adapter_id = Some(id);
        Handle {
            slot,
            generation: self.slots[slot].generation,
            adapter_id: id,
        }
    }
    pub fn pin(&mut self, h: Handle) -> bool {
        let s = &mut self.slots[h.slot];
        if s.generation != h.generation || s.adapter_id != Some(h.adapter_id) {
            return false;
        }
        s.pins += 1;
        true
    }
    pub fn unpin(&mut self, h: Handle) -> bool {
        let s = &mut self.slots[h.slot];
        if s.generation != h.generation || s.adapter_id != Some(h.adapter_id) || s.pins == 0 {
            return false;
        }
        s.pins -= 1;
        true
    }
    pub fn evict(&mut self, slot: usize) -> bool {
        let s = &mut self.slots[slot];
        if s.pins != 0 {
            return false;
        }
        s.adapter_id = None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn adapter(id: u64, rank: usize) -> Adapter {
        let din = 4;
        let dout = 3;
        let a = (0..rank * din)
            .map(|i| ((i + 1 + id as usize) % 7) as f64 / 7.0)
            .collect();
        let b = (0..dout * rank)
            .map(|i| ((i + 2 * id as usize) % 5) as f64 / 5.0)
            .collect();
        Adapter {
            id,
            rank,
            scale: 0.5,
            a,
            b,
        }
    }
    #[test]
    fn dynamic_matches_merged_oracle_for_heterogeneous_ranks() {
        let base = vec![1., 0., 2., 0., 0., 1., 0., 2., 1., 1., 1., 1.];
        let x = vec![1., 2., 3., 4.];
        for r in [1, 2, 3] {
            let a = adapter(10 + r as u64, r);
            let y = dynamic(&base, 3, 4, &x, Some(&a));
            let o = merged_oracle(&base, 3, 4, &x, Some(&a));
            for i in 0..3 {
                assert!((y[i] - o[i]).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn mixed_rows_keep_their_adapter_identity() {
        let base = vec![1., 0., 2., 0., 0., 1., 0., 2., 1., 1., 1., 1.];
        let ads = vec![adapter(1, 1), adapter(2, 3)];
        let rows = vec![
            vec![1., 2., 3., 4.],
            vec![2., 0., 1., 3.],
            vec![1., 1., 1., 1.],
        ];
        let ids = vec![Some(0), None, Some(1)];
        let got = mixed_batch(&base, 3, 4, &rows, &ids, &ads);
        for i in 0..rows.len() {
            let want = merged_oracle(&base, 3, 4, &rows[i], ids[i].map(|j| &ads[j]));
            for j in 0..3 {
                assert!((got[i][j] - want[j]).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn stale_slot_handle_is_rejected() {
        let mut r = Residency::new(1);
        let old = r.load(0, 101);
        assert!(r.pin(old));
        assert!(r.unpin(old));
        assert!(r.evict(0));
        let new = r.load(0, 202);
        assert_ne!(old.generation, new.generation);
        assert!(!r.pin(old));
        assert!(r.pin(new));
    }
    #[test]
    fn pinned_adapter_cannot_be_evicted_or_replaced() {
        let mut r = Residency::new(1);
        let h = r.load(0, 7);
        assert!(r.pin(h));
        assert!(!r.evict(0));
        assert!(r.unpin(h));
        assert!(r.evict(0));
    }
}
