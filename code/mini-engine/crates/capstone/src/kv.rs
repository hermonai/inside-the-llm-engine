//! The paged KV cache: a pool of fixed-size blocks, a free list, and a block
//! table per sequence that maps its token positions to blocks.
//!
//! Block `b` holds `block_size` consecutive positions of one sequence for every
//! layer. Keys and values live in two flat arrays laid out as
//! `[block][layer][slot][kv_width]`, so position `p` of a sequence is slot
//! `p % block_size` of block `table[p / block_size]`. A sequence's blocks need
//! not be adjacent, which is the point: memory is handed out and returned in
//! blocks, and no request reserves a contiguous region for its longest
//! possible length.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    OutOfBlocks { wanted: usize, free: usize },
    PositionBeyondTable { pos: usize, capacity: usize },
    DoubleFree(BlockId),
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBlocks { wanted, free } => write!(f, "wanted {wanted} blocks, {free} free"),
            Self::PositionBeyondTable { pos, capacity } => {
                write!(
                    f,
                    "position {pos} is beyond the {capacity} positions its block table covers"
                )
            }
            Self::DoubleFree(id) => write!(f, "block {} freed twice", id.0),
        }
    }
}

impl std::error::Error for KvError {}

#[derive(Debug)]
pub struct KvPool {
    block_size: usize,
    n_layers: usize,
    kv_width: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    free: Vec<BlockId>,
    in_use: Vec<bool>,
}

impl KvPool {
    pub fn new(n_blocks: usize, block_size: usize, n_layers: usize, kv_width: usize) -> Self {
        let floats = n_blocks * n_layers * block_size * kv_width;
        Self {
            block_size,
            n_layers,
            kv_width,
            k: vec![0.0; floats],
            v: vec![0.0; floats],
            // Pop from the end, so blocks are handed out in ascending order.
            free: (0..n_blocks as u32).rev().map(BlockId).collect(),
            in_use: vec![false; n_blocks],
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.in_use.len()
    }

    /// Blocks a sequence of `tokens` positions needs.
    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }

    /// All or nothing: either every block is handed out or none is.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<BlockId>, KvError> {
        if n > self.free.len() {
            return Err(KvError::OutOfBlocks {
                wanted: n,
                free: self.free.len(),
            });
        }
        let blocks: Vec<BlockId> = (0..n)
            .map(|_| self.free.pop().expect("checked above"))
            .collect();
        for b in &blocks {
            self.in_use[b.0 as usize] = true;
        }
        Ok(blocks)
    }

    pub fn release(&mut self, blocks: &[BlockId]) -> Result<(), KvError> {
        for &b in blocks {
            if !self.in_use[b.0 as usize] {
                return Err(KvError::DoubleFree(b));
            }
            self.in_use[b.0 as usize] = false;
            self.free.push(b);
        }
        Ok(())
    }

    fn offset(&self, table: &[BlockId], layer: usize, pos: usize) -> Result<usize, KvError> {
        let block = table
            .get(pos / self.block_size)
            .ok_or(KvError::PositionBeyondTable {
                pos,
                capacity: table.len() * self.block_size,
            })?;
        let slot = pos % self.block_size;
        Ok(((block.0 as usize * self.n_layers + layer) * self.block_size + slot) * self.kv_width)
    }

    pub fn write(
        &mut self,
        table: &[BlockId],
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), KvError> {
        let o = self.offset(table, layer, pos)?;
        self.k[o..o + self.kv_width].copy_from_slice(k);
        self.v[o..o + self.kv_width].copy_from_slice(v);
        Ok(())
    }

    /// The key and value rows (all key/value heads) of one position.
    pub fn read(
        &self,
        table: &[BlockId],
        layer: usize,
        pos: usize,
    ) -> Result<(&[f32], &[f32]), KvError> {
        let o = self.offset(table, layer, pos)?;
        Ok((&self.k[o..o + self.kv_width], &self.v[o..o + self.kv_width]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sequence_can_live_in_scattered_blocks() {
        let mut pool = KvPool::new(6, 2, 1, 3);
        let a = pool.alloc(1).unwrap();
        let b = pool.alloc(2).unwrap(); // blocks 1 and 2
        pool.release(&a).unwrap(); // block 0 back on the free list
        let c = pool.alloc(1).unwrap(); // reuses block 0
        let table = vec![b[0], c[0], b[1]]; // positions 0-1, 2-3, 4-5 in blocks 1, 0, 2
        for pos in 0..6 {
            let x = pos as f32;
            pool.write(&table, 0, pos, &[x, x, x], &[-x, -x, -x])
                .unwrap();
        }
        for pos in 0..6 {
            let (k, v) = pool.read(&table, 0, pos).unwrap();
            assert_eq!((k[0], v[0]), (pos as f32, -(pos as f32)));
        }
        assert!(matches!(
            pool.read(&table, 0, 6),
            Err(KvError::PositionBeyondTable { .. })
        ));
    }

    #[test]
    fn allocation_is_all_or_nothing_and_double_free_is_caught() {
        let mut pool = KvPool::new(3, 4, 2, 8);
        assert_eq!(
            pool.alloc(4),
            Err(KvError::OutOfBlocks { wanted: 4, free: 3 })
        );
        assert_eq!(pool.free_blocks(), 3);
        let blocks = pool.alloc(2).unwrap();
        pool.release(&blocks).unwrap();
        assert_eq!(
            pool.release(&blocks[..1]),
            Err(KvError::DoubleFree(blocks[0]))
        );
        assert_eq!(pool.blocks_for(9), 3);
    }
}
