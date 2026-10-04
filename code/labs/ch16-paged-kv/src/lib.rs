use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle {
    pub block: usize,
    pub generation: u64,
}

#[derive(Clone, Debug)]
struct Block {
    cells: Vec<Option<i32>>,
    refs: usize,
    generation: u64,
}

#[derive(Clone, Debug, Default)]
struct Sequence {
    table: Vec<Handle>,
    len: usize,
}

/// A deliberately small CPU block pool. `i32` stands in for a token's KV
/// payload so address translation, ownership, COW, OOM and stale handles are
/// visible without GPU code.
pub struct Pool {
    block_size: usize,
    blocks: Vec<Block>,
    free: Vec<usize>,
    seqs: BTreeMap<String, Sequence>,
}

impl Pool {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0 && block_size > 0);
        let blocks = (0..num_blocks)
            .map(|_| Block {
                cells: vec![None; block_size],
                refs: 0,
                generation: 0,
            })
            .collect();
        // Reverse so allocation is deterministic: 0,1,2,...
        let free = (0..num_blocks).rev().collect();
        Self {
            block_size,
            blocks,
            free,
            seqs: BTreeMap::new(),
        }
    }

    pub fn create(&mut self, id: &str) {
        assert!(!self.seqs.contains_key(id));
        self.seqs.insert(id.to_owned(), Sequence::default());
    }

    fn alloc(&mut self) -> Result<Handle, &'static str> {
        let id = self.free.pop().ok_or("out of blocks")?;
        let b = &mut self.blocks[id];
        assert_eq!(b.refs, 0);
        b.generation = b.generation.wrapping_add(1);
        b.cells.fill(None);
        b.refs = 1;
        Ok(Handle {
            block: id,
            generation: b.generation,
        })
    }

    fn valid(&self, h: Handle) -> bool {
        self.blocks
            .get(h.block)
            .map(|b| b.generation == h.generation && b.refs > 0)
            .unwrap_or(false)
    }

    fn release_handle(&mut self, h: Handle) {
        assert!(self.valid(h), "release of stale block handle");
        let b = &mut self.blocks[h.block];
        b.refs -= 1;
        if b.refs == 0 {
            b.cells.fill(None);
            self.free.push(h.block);
        }
    }

    /// Append one logical payload. Allocation happens only at a block boundary.
    /// If a fork shares a partial tail, append performs copy-on-write first.
    pub fn append(&mut self, id: &str, value: i32) -> Result<(), &'static str> {
        let (len, tail) = {
            let s = self.seqs.get(id).ok_or("unknown sequence")?;
            (s.len, s.table.last().copied())
        };
        let offset = len % self.block_size;

        if offset == 0 {
            let h = self.alloc()?;
            self.seqs.get_mut(id).unwrap().table.push(h);
        } else {
            let h = tail.expect("nonzero offset requires a tail");
            if !self.valid(h) {
                return Err("stale tail");
            }
            if self.blocks[h.block].refs > 1 {
                // COW must be failure-atomic: allocate before changing ownership.
                let new_h = self.alloc()?;
                let old_cells = self.blocks[h.block].cells.clone();
                self.blocks[new_h.block].cells = old_cells;
                self.release_handle(h);
                *self.seqs.get_mut(id).unwrap().table.last_mut().unwrap() = new_h;
            }
        }

        let (h, off) = {
            let s = self.seqs.get(id).unwrap();
            (*s.table.last().unwrap(), s.len % self.block_size)
        };
        if !self.valid(h) {
            return Err("stale write handle");
        }
        self.blocks[h.block].cells[off] = Some(value);
        self.seqs.get_mut(id).unwrap().len += 1;
        Ok(())
    }

    /// Fork shares all current blocks. A later write to a shared partial tail
    /// will COW; full blocks remain safely shared.
    pub fn fork(&mut self, src: &str, dst: &str) -> Result<(), &'static str> {
        if self.seqs.contains_key(dst) {
            return Err("destination exists");
        }
        let s = self.seqs.get(src).ok_or("unknown source")?.clone();
        for h in &s.table {
            if !self.valid(*h) {
                return Err("stale source handle");
            }
            self.blocks[h.block].refs += 1;
        }
        self.seqs.insert(dst.to_owned(), s);
        Ok(())
    }

    pub fn read(&self, id: &str, pos: usize) -> Result<i32, &'static str> {
        let s = self.seqs.get(id).ok_or("unknown sequence")?;
        if pos >= s.len {
            return Err("position out of range");
        }
        let logical = pos / self.block_size;
        let offset = pos % self.block_size;
        let h = s.table[logical];
        if !self.valid(h) {
            return Err("stale read handle");
        }
        self.blocks[h.block].cells[offset].ok_or("uninitialized cell")
    }

    pub fn free_sequence(&mut self, id: &str) -> Result<(), &'static str> {
        let s = self.seqs.remove(id).ok_or("unknown sequence")?;
        for h in s.table {
            self.release_handle(h);
        }
        Ok(())
    }

    pub fn table(&self, id: &str) -> Vec<Handle> {
        self.seqs[id].table.clone()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Independent logical oracle helper: materialize through public reads.
    pub fn materialize(&self, id: &str) -> Result<Vec<i32>, &'static str> {
        let len = self.seqs.get(id).ok_or("unknown sequence")?.len;
        (0..len).map(|p| self.read(id, p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_address_translation_matches_flat_oracle() {
        let mut p = Pool::new(4, 4);
        p.create("A");
        let flat = vec![10, 11, 12, 13, 14, 15];
        for &v in &flat {
            p.append("A", v).unwrap();
        }
        assert_eq!(p.table("A").len(), 2);
        assert_eq!(p.materialize("A").unwrap(), flat);
        assert_eq!(p.read("A", 5).unwrap(), 15);
    }

    #[test]
    fn shared_partial_tail_copies_on_write() {
        let mut p = Pool::new(5, 4);
        p.create("P");
        for v in 0..6 {
            p.append("P", v).unwrap();
        }
        p.fork("P", "X").unwrap();
        let old_tail = p.table("P")[1];
        p.append("X", 99).unwrap();
        let new_tail = p.table("X")[1];
        assert_ne!(old_tail.block, new_tail.block);
        assert_eq!(p.materialize("P").unwrap(), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(p.materialize("X").unwrap(), vec![0, 1, 2, 3, 4, 5, 99]);
    }

    #[test]
    fn freeing_fork_does_not_destroy_shared_blocks() {
        let mut p = Pool::new(4, 4);
        p.create("A");
        for v in 0..4 {
            p.append("A", v).unwrap();
        }
        p.fork("A", "B").unwrap();
        p.free_sequence("A").unwrap();
        assert_eq!(p.materialize("B").unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn oom_is_failure_atomic_at_new_block_boundary() {
        let mut p = Pool::new(1, 2);
        p.create("A");
        p.append("A", 7).unwrap();
        p.append("A", 8).unwrap();
        let before = p.materialize("A").unwrap();
        assert_eq!(p.append("A", 9), Err("out of blocks"));
        assert_eq!(p.materialize("A").unwrap(), before);
    }

    #[test]
    fn stale_generation_is_detectable_after_reuse() {
        let mut p = Pool::new(1, 2);
        p.create("A");
        p.append("A", 1).unwrap();
        let old = p.table("A")[0];
        p.free_sequence("A").unwrap();
        p.create("B");
        p.append("B", 2).unwrap();
        let new = p.table("B")[0];
        assert_eq!(old.block, new.block);
        assert_ne!(old.generation, new.generation);
        assert!(!p.valid(old));
        assert!(p.valid(new));
    }
}
