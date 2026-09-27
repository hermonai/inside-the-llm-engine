//! A minimal GGUF v3 reader, and ggml's reference decoders for four block
//! formats, transcribed from `ggml/src/ggml-quants.c` (llama.cpp d006858) so
//! that they produce the same bits.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;

/// Elements and bytes per block of a ggml type.
pub fn block(ty: u32) -> (usize, usize) {
    match ty {
        F32 => (1, 4),
        F16 => (1, 2),
        Q4_0 => (32, 18),
        Q8_0 => (32, 34),
        Q4_K => (256, 144),
        Q6_K => (256, 210),
        _ => panic!("unsupported ggml type {ty}"),
    }
}

pub fn type_name(ty: u32) -> &'static str {
    match ty {
        F32 => "F32",
        F16 => "F16",
        Q4_0 => "Q4_0",
        Q8_0 => "Q8_0",
        Q4_K => "Q4_K",
        Q6_K => "Q6_K",
        _ => "other",
    }
}

/// IEEE half to single precision, exactly.
pub fn f16(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalize the mantissa
            let (mut e, mut m) = (127 - 15 + 1, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Single to half precision, rounding to nearest even (as ggml's FP32_TO_FP16).
pub fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let full = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let (r, rem, half) = (
            full >> shift,
            full & ((1 << shift) - 1),
            1u32 << (shift - 1),
        );
        let r = if rem > half || (rem == half && r & 1 == 1) {
            r + 1
        } else {
            r
        };
        return sign | r as u16;
    }
    let r = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r + 1
    } else {
        r
    };
    sign | r as u16
}

fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

/// The six-bit scale and minimum of sub-block `j` of a Q4_K super-block.
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Decodes whole blocks of `ty` to f32, in ggml's element order.
pub fn decode(ty: u32, bytes: &[u8]) -> Vec<f32> {
    let (be, bb) = block(ty);
    let mut y = Vec::with_capacity(bytes.len() / bb * be);
    for b in bytes.chunks_exact(bb) {
        match ty {
            F32 => y.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            F16 => y.push(f16(le16(b, 0))),
            Q4_0 => {
                let d = f16(le16(b, 0));
                let qs = &b[2..18];
                y.extend(qs.iter().map(|q| ((q & 0xf) as i32 - 8) as f32 * d));
                y.extend(qs.iter().map(|q| ((q >> 4) as i32 - 8) as f32 * d));
            }
            Q8_0 => {
                let d = f16(le16(b, 0));
                y.extend(b[2..34].iter().map(|&q| (q as i8) as f32 * d));
            }
            Q4_K => {
                let (d, min) = (f16(le16(b, 0)), f16(le16(b, 2)));
                let scales = &b[4..16];
                for (j, q) in b[16..144].chunks_exact(32).enumerate() {
                    let (s1, m1) = scale_min_k4(2 * j, scales);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (d1, m1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, min * m2 as f32);
                    y.extend(q.iter().map(|x| d1 * (x & 0xf) as f32 - m1));
                    y.extend(q.iter().map(|x| d2 * (x >> 4) as f32 - m2));
                }
            }
            Q6_K => {
                let d = f16(le16(b, 208));
                let mut out = [0.0f32; 256];
                for n in 0..2 {
                    let ql = &b[64 * n..64 * n + 64];
                    let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                    let sc = &b[192 + 8 * n..192 + 8 * n + 8];
                    let y0 = &mut out[128 * n..128 * n + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                        let s = |k: usize| d * (sc[is + k] as i8) as f32;
                        y0[l] = s(0) * q1 as f32;
                        y0[l + 32] = s(2) * q2 as f32;
                        y0[l + 64] = s(4) * q3 as f32;
                        y0[l + 96] = s(6) * q4 as f32;
                    }
                }
                y.extend_from_slice(&out);
            }
            _ => unreachable!(),
        }
    }
    y
}

pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub offset: u64,
}

impl TensorInfo {
    /// Bytes of one row (dims[0] elements).
    pub fn row_bytes(&self) -> usize {
        let (be, bb) = block(self.ty);
        self.dims[0] as usize / be * bb
    }
}

pub struct Gguf {
    file: File,
    data_start: u64,
    pub tensors: Vec<TensorInfo>,
    /// Numeric metadata; strings and arrays are skipped.
    pub numbers: HashMap<String, f64>,
}

fn u32_<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64_<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn string<R: Read>(r: &mut R) -> io::Result<String> {
    let n = u64_(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Reads one metadata value; returns it if it is a number.
fn value<R: Read>(r: &mut R, ty: u32) -> io::Result<Option<f64>> {
    let mut b = [0u8; 8];
    Ok(match ty {
        0 | 1 | 7 => {
            r.read_exact(&mut b[..1])?;
            Some(if ty == 1 {
                b[0] as i8 as f64
            } else {
                b[0] as f64
            })
        }
        2 | 3 => {
            r.read_exact(&mut b[..2])?;
            let v = u16::from_le_bytes([b[0], b[1]]);
            Some(if ty == 3 { v as i16 as f64 } else { v as f64 })
        }
        4..=6 => {
            r.read_exact(&mut b[..4])?;
            let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Some(match ty {
                4 => v as f64,
                5 => v as i32 as f64,
                _ => f32::from_bits(v) as f64,
            })
        }
        10..=12 => {
            r.read_exact(&mut b)?;
            let v = u64::from_le_bytes(b);
            Some(match ty {
                10 => v as f64,
                11 => v as i64 as f64,
                _ => f64::from_bits(v),
            })
        }
        8 => {
            string(r)?;
            None
        }
        9 => {
            let et = u32_(r)?;
            let n = u64_(r)?;
            for _ in 0..n {
                value(r, et)?;
            }
            None
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown GGUF value type",
            ))
        }
    })
}

impl Gguf {
    pub fn open(path: &str) -> io::Result<Gguf> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a GGUF file",
            ));
        }
        let _version = u32_(&mut r)?;
        let n_tensors = u64_(&mut r)?;
        let n_kv = u64_(&mut r)?;
        let mut numbers = HashMap::new();
        for _ in 0..n_kv {
            let key = string(&mut r)?;
            let ty = u32_(&mut r)?;
            if let Some(v) = value(&mut r, ty)? {
                numbers.insert(key, v);
            }
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = string(&mut r)?;
            let nd = u32_(&mut r)? as usize;
            let dims = (0..nd)
                .map(|_| u64_(&mut r))
                .collect::<io::Result<Vec<_>>>()?;
            let ty = u32_(&mut r)?;
            let offset = u64_(&mut r)?;
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }
        let align = numbers.get("general.alignment").copied().unwrap_or(32.0) as u64;
        let data_start = r.stream_position()?.div_ceil(align) * align;
        Ok(Gguf {
            file: r.into_inner(),
            data_start,
            tensors,
            numbers,
        })
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// The raw bytes of `rows` rows of tensor `i`, starting at row `row0`.
    pub fn raw_rows(&mut self, i: usize, row0: usize, rows: usize) -> io::Result<Vec<u8>> {
        let t = &self.tensors[i];
        let rb = t.row_bytes();
        let mut buf = vec![0u8; rows * rb];
        self.file.seek(SeekFrom::Start(
            self.data_start + t.offset + (row0 * rb) as u64,
        ))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Rows of tensor `i` decoded to f32.
    pub fn rows(&mut self, i: usize, row0: usize, rows: usize) -> io::Result<Vec<f32>> {
        let ty = self.tensors[i].ty;
        Ok(decode(ty, &self.raw_rows(i, row0, rows)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_round_trips() {
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xc000), -2.0);
        assert_eq!(f16(0x7bff), 65504.0);
        assert_eq!(f16(0x0001), 2f32.powi(-24));
        for &x in &[1.0f32, -2.0, 65504.0, 0.1, 1.2345, 1e-5, -7.5e-7] {
            let h = to_f16(x);
            assert!((f16(h) - x).abs() <= x.abs() * 1e-3 + 6e-8, "{x}");
        }
        for h in [0u16, 1, 0x3c00, 0x3c01, 0x7bff, 0x8001, 0x0400, 0x03ff] {
            assert_eq!(to_f16(f16(h)), h);
        }
    }

    #[test]
    fn q4_0_and_q8_0_decode_known_blocks() {
        // d = 0.5 (0x3800); nibbles 0..15 give (q - 8) * d.
        let mut b = vec![0x00, 0x38];
        b.extend((0..16u8).map(|j| j | (15 - j) << 4));
        let y = decode(Q4_0, &b);
        assert_eq!(y[0], -4.0);
        assert_eq!(y[15], 3.5);
        assert_eq!(y[16], 3.5);
        assert_eq!(y[31], -4.0);
        let mut b = vec![0x00, 0x3c]; // d = 1
        b.extend((0..32).map(|j| (j as i8 - 16) as u8));
        let y = decode(Q8_0, &b);
        assert_eq!(y[0], -16.0);
        assert_eq!(y[31], 15.0);
    }

    #[test]
    fn k_quant_scales_unpack() {
        // First four sub-blocks: low six bits of bytes 0..4 (scales) and 4..8 (mins).
        let mut q = [0u8; 12];
        q[0] = 0b11_000101; // scale 5, top bits 3 go to sub-block 4's scale
        q[4] = 0b10_000111; // min 7, top bits 2 go to sub-block 4's min
        q[8] = 0x9a; // sub-block 4: low nibble 0xa (scale), high nibble 0x9 (min)
        assert_eq!(scale_min_k4(0, &q), (5, 7));
        assert_eq!(scale_min_k4(4, &q), (0xa | (3 << 4), 0x9 | (2 << 4)));
    }
}
