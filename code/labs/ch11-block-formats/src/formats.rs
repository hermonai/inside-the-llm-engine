//! Small floating-point formats written from their specifications, and the
//! block-scaled tensor formats built on them.
//!
//! Elements: FP8 E4M3 (the OCP "FN" variant: no infinities, one NaN code per
//! sign, largest value 448) and E5M2 (IEEE-style infinities and NaNs, largest
//! 57,344); FP6 E2M3 and E3M2 and FP4 E2M1 (no special values; largest 7.5, 28
//! and 6). Conversion rounds to the nearest representable value, ties to the
//! even code, and saturates to the largest finite value.
//!
//! Tensors: FP8 with one FP32 scale per tensor; the OCP MX formats (32 elements
//! share an E8M0 power-of-two scale, chosen by Algorithm 1 of Rouhani et al.:
//! shared exponent = floor(log2(max |v|)) - emax of the element format); and
//! NVFP4 (16 E2M1 elements share an E4M3 scale, under an FP32 scale per tensor,
//! following equations 1-4 of NVIDIA's NVFP4 pretraining paper).

#[derive(Clone, Copy, Debug)]
pub struct Minifloat {
    pub name: &'static str,
    pub e: u32,
    pub m: u32,
    pub bias: i32,
    /// The all-ones exponent holds infinities and NaNs (IEEE style).
    pub ieee_specials: bool,
    /// E4M3's single NaN magnitude (exponent and mantissa all ones).
    pub nan_all_ones: bool,
}

pub const E4M3: Minifloat = Minifloat {
    name: "E4M3",
    e: 4,
    m: 3,
    bias: 7,
    ieee_specials: false,
    nan_all_ones: true,
};
pub const E5M2: Minifloat = Minifloat {
    name: "E5M2",
    e: 5,
    m: 2,
    bias: 15,
    ieee_specials: true,
    nan_all_ones: false,
};
pub const E2M3: Minifloat = Minifloat {
    name: "E2M3",
    e: 2,
    m: 3,
    bias: 1,
    ieee_specials: false,
    nan_all_ones: false,
};
pub const E3M2: Minifloat = Minifloat {
    name: "E3M2",
    e: 3,
    m: 2,
    bias: 3,
    ieee_specials: false,
    nan_all_ones: false,
};
pub const E2M1: Minifloat = Minifloat {
    name: "E2M1",
    e: 2,
    m: 1,
    bias: 1,
    ieee_specials: false,
    nan_all_ones: false,
};

impl Minifloat {
    pub fn bits(&self) -> u32 {
        1 + self.e + self.m
    }

    /// The value of a code.
    pub fn decode(&self, code: u32) -> f32 {
        let sign = if (code >> (self.e + self.m)) & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        let exp = (code >> self.m) & ((1 << self.e) - 1);
        let man = code & ((1 << self.m) - 1);
        let all_ones = (1 << self.e) - 1;
        if self.ieee_specials && exp == all_ones {
            return if man == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
        if self.nan_all_ones && exp == all_ones && man == (1 << self.m) - 1 {
            return f32::NAN;
        }
        let frac = man as f32 / (1u32 << self.m) as f32;
        if exp == 0 {
            sign * frac * 2f32.powi(1 - self.bias)
        } else {
            sign * (1.0 + frac) * 2f32.powi(exp as i32 - self.bias)
        }
    }

    /// Non-negative finite values, in code order (which is also value order).
    pub fn magnitudes(&self) -> Vec<f32> {
        (0..(1u32 << (self.e + self.m)))
            .map(|c| self.decode(c))
            .take_while(|v| v.is_finite())
            .collect()
    }

    pub fn max(&self) -> f32 {
        *self.magnitudes().last().expect("a finite value")
    }

    /// The exponent of the largest normal value (emax_elem in the MX algorithm).
    pub fn emax(&self) -> i32 {
        self.max().log2().floor() as i32
    }

    /// Round to nearest, ties to the even code, saturating; returns the code.
    pub fn encode(&self, x: f32) -> u32 {
        let sign = if x.is_sign_negative() {
            1 << (self.e + self.m)
        } else {
            0
        };
        let a = x.abs();
        let mags = self.magnitudes();
        let top = mags.len() - 1;
        if a.is_nan() || a >= mags[top] {
            return sign | top as u32; // saturate
        }
        // first magnitude >= a
        let hi = mags.partition_point(|&v| v < a);
        if hi == 0 {
            return sign;
        }
        let lo = hi - 1;
        let (dlo, dhi) = (a - mags[lo], mags[hi] - a);
        let c = if dlo < dhi || (dlo == dhi && lo % 2 == 0) {
            lo
        } else {
            hi
        };
        sign | c as u32
    }

    pub fn round(&self, x: f32) -> f32 {
        self.decode(self.encode(x))
    }
}

/// Rounds an f32 to bfloat16 (round to nearest even) and back.
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits((b.wrapping_add(0x7fff + lsb)) & 0xffff_0000)
}

/// An E8M0 scale: 2^(code - 127); the MX formats clamp to that range.
pub fn e8m0(exp: i32) -> f32 {
    2f32.powi(exp.clamp(-127, 127))
}

/// Result of quantizing a tensor: its dequantized values and bits per value.
pub struct Quantized {
    pub values: Vec<f32>,
    pub bits: f64,
    /// Blocks whose largest element had to saturate (MX only).
    pub saturated_blocks: usize,
}

/// FP8 with one FP32 scale for the whole tensor, mapping its largest
/// magnitude to the format's largest value.
pub fn per_tensor(x: &[f32], f: &Minifloat) -> Quantized {
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let s = if amax > 0.0 { amax / f.max() } else { 1.0 };
    Quantized {
        values: x.iter().map(|v| f.round(v / s) * s).collect(),
        bits: f.bits() as f64 + 32.0 / x.len() as f64,
        saturated_blocks: 0,
    }
}

/// An MX format: blocks of 32 share X = 2^(floor(log2(max |v|)) - emax).
pub fn mx(x: &[f32], f: &Minifloat) -> Quantized {
    let mut out = Vec::with_capacity(x.len());
    let mut saturated = 0;
    let top = f.max();
    for b in x.chunks(32) {
        let amax = b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        if amax == 0.0 {
            out.extend(b.iter().map(|_| 0.0));
            continue;
        }
        let scale = e8m0(amax.log2().floor() as i32 - f.emax());
        if amax / scale > top {
            saturated += 1;
        }
        out.extend(b.iter().map(|v| f.round(v / scale) * scale));
    }
    Quantized {
        values: out,
        bits: f.bits() as f64 + 8.0 / 32.0,
        saturated_blocks: saturated,
    }
}

/// NVFP4: E2M1 elements in blocks of 16, each block's scale stored in E4M3
/// after a per-tensor FP32 scale s_enc = 6 * 448 / amax brings it into range.
pub fn nvfp4(x: &[f32]) -> Quantized {
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let s_enc = if amax > 0.0 { 6.0 * 448.0 / amax } else { 1.0 };
    let mut out = Vec::with_capacity(x.len());
    for b in x.chunks(16) {
        let bmax = b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        // the block's decode scale amax_b / 6, stored in E4M3 after scaling by s_enc
        let stored = E4M3.round(bmax / 6.0 * s_enc);
        if stored == 0.0 {
            out.extend(b.iter().map(|_| 0.0));
            continue;
        }
        out.extend(
            b.iter()
                .map(|v| E2M1.round(v * s_enc / stored) * stored / s_enc),
        );
    }
    Quantized {
        values: out,
        bits: 4.0 + 8.0 / 16.0 + 32.0 / x.len() as f64,
        saturated_blocks: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_matches_its_specification() {
        assert_eq!(E4M3.decode(0x7e), 448.0);
        assert!(E4M3.decode(0x7f).is_nan() && E4M3.decode(0xff).is_nan());
        assert_eq!(E4M3.decode(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(E4M3.decode(0x08), 2f32.powi(-6)); // smallest normal
        assert_eq!(E4M3.magnitudes().len(), 127); // 0 and 126 positive values
        for c in 0..256u32 {
            let v = E4M3.decode(c);
            if v.is_finite() {
                assert_eq!(E4M3.encode(v), c, "code {c:#x}");
            }
        }
        assert_eq!(E4M3.encode(1000.0), 0x7e); // saturates
        assert_eq!(E4M3.emax(), 8);
    }

    #[test]
    fn e5m2_matches_its_specification() {
        assert_eq!(E5M2.decode(0x7b), 57344.0);
        assert!(E5M2.decode(0x7c).is_infinite());
        assert!(E5M2.decode(0x7d).is_nan());
        assert_eq!(E5M2.decode(0x04), 2f32.powi(-14)); // smallest normal
        assert_eq!(E5M2.decode(0x01), 2f32.powi(-16)); // smallest subnormal
        for c in 0..256u32 {
            let v = E5M2.decode(c);
            if v.is_finite() {
                assert_eq!(E5M2.encode(v), c, "code {c:#x}");
            }
        }
        assert_eq!(E5M2.emax(), 15);
    }

    #[test]
    fn fp6_and_fp4_grids() {
        assert_eq!(
            E2M1.magnitudes(),
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        );
        assert_eq!(E2M1.emax(), 2);
        assert_eq!(E2M3.max(), 7.5);
        assert_eq!(E3M2.max(), 28.0);
        for f in [E2M3, E3M2, E2M1] {
            for c in 0..(1u32 << f.bits()) {
                assert_eq!(f.encode(f.decode(c)), c, "{} code {c}", f.name);
            }
        }
    }

    #[test]
    fn ties_go_to_the_even_code_and_large_values_saturate() {
        assert_eq!(E2M1.round(2.5), 2.0); // between 2 (code 4) and 3 (code 5)
        assert_eq!(E2M1.round(5.0), 4.0); // between 4 (code 6) and 6 (code 7)
        assert_eq!(E2M1.round(0.25), 0.0);
        assert_eq!(E2M1.round(0.75), 1.0); // between 0.5 (code 1) and 1 (code 2)
        assert_eq!(E2M1.round(-7.0), -6.0);
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0); // tie to even
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn mx_and_nvfp4_keep_grid_values_exact() {
        // A block that already lies on the E2M1 grid at scale 1.
        let block: Vec<f32> = (0..32)
            .map(|i| [0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0][i % 8])
            .collect();
        let q = mx(&block, &E2M1);
        assert_eq!(q.values, block);
        assert_eq!(q.saturated_blocks, 0);
        let q = nvfp4(&block);
        assert_eq!(q.values, block);
        assert!((q.bits - (4.5 + 32.0 / 32.0)).abs() < 1e-12);
    }

    #[test]
    fn the_worked_examples_of_the_chapter() {
        // 0.3 in E4M3: between 0.28125 and 0.3125, nearer the second (code 0x2a).
        assert_eq!(E4M3.encode(0.3), 0x2a);
        assert_eq!(E4M3.decode(0x2a), 0.3125);
        // Chapter 10's eight weights in MXFP4: X = 2^(floor(log2 0.9) - 2) = 1/8.
        let w = [0.9f32, -0.35, 0.12, 0.05, -0.61, 0.28, -0.02, 0.44];
        let q = mx(&w, &E2M1);
        assert_eq!(
            q.values,
            vec![0.75, -0.375, 0.125, 0.0625, -0.5, 0.25, -0.0, 0.5]
        );
        assert_eq!(q.saturated_blocks, 1);
        // ... and in NVFP4: the block scale 0.9 / 6 is stored exactly.
        let q = nvfp4(&w);
        let expect = [0.9f32, -0.3, 0.15, 0.075, -0.6, 0.3, 0.0, 0.45];
        for (a, b) in q.values.iter().zip(expect) {
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
    }

    #[test]
    fn mx_saturates_the_top_of_a_binade() {
        // max 7: floor(log2 7) = 2, X = 1, and 7 > 6 saturates to 6.
        let mut block = vec![0.1f32; 32];
        block[0] = 7.0;
        let q = mx(&block, &E2M1);
        assert_eq!(q.values[0], 6.0);
        assert_eq!(q.saturated_blocks, 1);
    }
}
