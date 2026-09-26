use std::hint::black_box;
use std::time::Instant;

type V = [f32; 4];
#[inline(always)]
fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]] }
#[inline(always)]
fn mul(a: V, b: V) -> V { [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]] }
#[inline(always)]
fn ld(s: &[f32], i: usize) -> V { s[i..i + 4].try_into().unwrap() }

// portable: 16 lanes as four 4-wide values
#[inline(never)]
pub fn dot16(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() / 16 * 16;
    let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    let mut i = 0;
    while i < n {
        s0 = add(s0, mul(ld(a, i), ld(b, i)));
        s1 = add(s1, mul(ld(a, i + 4), ld(b, i + 4)));
        s2 = add(s2, mul(ld(a, i + 8), ld(b, i + 8)));
        s3 = add(s3, mul(ld(a, i + 12), ld(b, i + 12)));
        i += 16;
    }
    let v = add(add(s0, s1), add(s2, s3));
    let mut sum = (v[0] + v[2]) + (v[1] + v[3]);
    for j in n..a.len() { sum += a[j] * b[j]; }
    sum
}

#[inline(never)]
pub fn dot16x4(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = a.len() / 16 * 16;
    let z = [0.0f32; 4];
    let mut s = [[z; 4]; 4];
    let mut i = 0;
    while i < n {
        let w = [ld(a, i), ld(a, i + 4), ld(a, i + 8), ld(a, i + 12)];
        for t in 0..4 {
            let x = b[t];
            s[t][0] = add(s[t][0], mul(w[0], ld(x, i)));
            s[t][1] = add(s[t][1], mul(w[1], ld(x, i + 4)));
            s[t][2] = add(s[t][2], mul(w[2], ld(x, i + 8)));
            s[t][3] = add(s[t][3], mul(w[3], ld(x, i + 12)));
        }
        i += 16;
    }
    let mut out = [0.0f32; 4];
    for t in 0..4 {
        let v = add(add(s[t][0], s[t][1]), add(s[t][2], s[t][3]));
        let mut sum = (v[0] + v[2]) + (v[1] + v[3]);
        for j in n..a.len() { sum += a[j] * b[t][j]; }
        out[t] = sum;
    }
    out
}

#[inline(never)]
pub fn stream16(a: &[f32]) -> f32 {
    let n = a.len() / 16 * 16;
    let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    let mut i = 0;
    while i < n {
        s0 = add(s0, ld(a, i));
        s1 = add(s1, ld(a, i + 4));
        s2 = add(s2, ld(a, i + 8));
        s3 = add(s3, ld(a, i + 12));
        i += 16;
    }
    let v = add(add(s0, s1), add(s2, s3));
    (v[0] + v[2]) + (v[1] + v[3])
}

#[inline(never)]
pub fn stream32(a: &[f32]) -> f32 {
    let n = a.len() / 32 * 32;
    let z = [0.0f32; 4];
    let mut s = [z; 8];
    let mut i = 0;
    while i < n {
        for k in 0..8 { s[k] = add(s[k], ld(a, i + 4 * k)); }
        i += 32;
    }
    let mut v = z;
    for k in 0..8 { v = add(v, s[k]); }
    (v[0] + v[2]) + (v[1] + v[3])
}

fn best<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    (0..runs).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min)
}


#[inline(always)]
fn lo(c: &[f32; 16], k: usize) -> V { [c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]] }
#[inline(always)]
fn fold(v: V) -> f32 { (v[0] + v[1]) + (v[2] + v[3]) }

#[inline(never)]
pub fn dot_a(a: &[f32], b: &[f32]) -> f32 {
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut s = [[0.0f32; 4]; 4];
    for (x, y) in ca.zip(cb) {
        let x: &[f32; 16] = x.try_into().unwrap();
        let y: &[f32; 16] = y.try_into().unwrap();
        for k in 0..4 { s[k] = add(s[k], mul(lo(x, k), lo(y, k))); }
    }
    let mut sum = fold(add(add(s[0], s[1]), add(s[2], s[3])));
    for (x, y) in ra.iter().zip(rb) { sum += x * y; }
    sum
}

#[inline(never)]
pub fn dot4_a(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = a.len() / 16 * 16;
    let mut s = [[[0.0f32; 4]; 4]; 4];
    let (w, x0, x1, x2, x3) = (&a[..n], &b[0][..n], &b[1][..n], &b[2][..n], &b[3][..n]);
    for ((((w, x0), x1), x2), x3) in w.chunks_exact(16).zip(x0.chunks_exact(16)).zip(x1.chunks_exact(16)).zip(x2.chunks_exact(16)).zip(x3.chunks_exact(16)) {
        let w: &[f32; 16] = w.try_into().unwrap();
        let xs: [&[f32; 16]; 4] = [x0.try_into().unwrap(), x1.try_into().unwrap(), x2.try_into().unwrap(), x3.try_into().unwrap()];
        for k in 0..4 {
            let wk = lo(w, k);
            for t in 0..4 { s[t][k] = add(s[t][k], mul(wk, lo(xs[t], k))); }
        }
    }
    let mut out = [0.0f32; 4];
    for t in 0..4 {
        let mut sum = fold(add(add(s[t][0], s[t][1]), add(s[t][2], s[t][3])));
        for (x, y) in a[n..].iter().zip(&b[t][n..]) { sum += x * y; }
        out[t] = sum;
    }
    out
}

#[inline(never)]
pub fn stream_a(a: &[f32]) -> f32 {
    let c = a.chunks_exact(32);
    let mut s = [[0.0f32; 4]; 8];
    for x in c {
        let x: &[f32; 32] = x.try_into().unwrap();
        for k in 0..8 { s[k] = add(s[k], [x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]]); }
    }
    let mut v = [0.0f32; 4];
    for k in 0..8 { v = add(v, s[k]); }
    fold(v)
}

fn main2() {
    let big: Vec<f32> = (0..64usize << 20).map(|i| (i % 7) as f32 * 0.5).collect();
    for _ in 0..2 {
        let t = best(5, || { black_box(stream_a(black_box(&big))); });
        println!("stream_a {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9);
        let t = best(5, || { black_box(dot_a(black_box(&big), black_box(&big))); });
        println!("dot_a(a,a) {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9);
    }
    let (rows, cols, n) = (512usize, 512usize, 64usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
    let x: Vec<f32> = (0..n * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
    let mut y = vec![0.0f32; n * rows];
    let flop = 2.0 * (rows * cols * n) as f64;
    for _ in 0..2 {
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                for (t, xr) in x.chunks_exact(cols).enumerate() { y[t * rows + r] = dot_a(wr, xr); }
            }
            black_box(&y);
        });
        println!("dot_a matmul {:.1} GFLOP/s", flop / t / 1e9);
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                let mut t = 0;
                while t + 4 <= n {
                    let o = dot4_a(wr, [&x[t * cols..(t + 1) * cols], &x[(t + 1) * cols..(t + 2) * cols], &x[(t + 2) * cols..(t + 3) * cols], &x[(t + 3) * cols..(t + 4) * cols]]);
                    for j in 0..4 { y[(t + j) * rows + r] = o[j]; }
                    t += 4;
                }
            }
            black_box(&y);
        });
        println!("dot4_a matmul {:.1} GFLOP/s", flop / t / 1e9);
    }
    // bit identity, odd length too
    for cols in [512usize, 37, 100] {
        let w: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
        let xs: Vec<Vec<f32>> = (0..4).map(|t| (0..cols).map(|i| (((i + t * 7) * 53 % 97) as f32 - 48.0) * 1e-3).collect()).collect();
        let o = dot4_a(&w, [&xs[0], &xs[1], &xs[2], &xs[3]]);
        for j in 0..4 { assert_eq!(o[j].to_bits(), dot_a(&w, &xs[j]).to_bits()); }
    }
    println!("A bit-identical: ok");
}

macro_rules! dot_variant {
    ($name:ident, $fold:expr) => {
        #[inline(never)]
        pub fn $name(a: &[f32], b: &[f32]) -> f32 {
            let ca = a.chunks_exact(16);
            let cb = b.chunks_exact(16);
            let (ra, rb) = (ca.remainder(), cb.remainder());
            let mut s = [[0.0f32; 4]; 4];
            for (x, y) in ca.zip(cb) {
                let x: &[f32; 16] = x.try_into().unwrap();
                let y: &[f32; 16] = y.try_into().unwrap();
                for k in 0..4 { s[k] = add(s[k], mul(lo(x, k), lo(y, k))); }
            }
            let v = add(add(s[0], s[1]), add(s[2], s[3]));
            let f: fn(V) -> f32 = $fold;
            let mut sum = f(v);
            for (x, y) in ra.iter().zip(rb) { sum += x * y; }
            sum
        }
    };
}
dot_variant!(dot_seq, |v| ((v[0] + v[1]) + v[2]) + v[3]);
dot_variant!(dot_lohi, |v| (v[0] + v[2]) + (v[1] + v[3]));

#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    // SAFETY: NEON is part of every aarch64 target; loads stay inside 16-float chunks.
    let mut sum = unsafe {
        let z = vdupq_n_f32(0.0);
        let mut s = [z; 4];
        for (x, y) in ca.zip(cb) {
            for k in 0..4 {
                let (xv, yv) = (vld1q_f32(x.as_ptr().add(4 * k)), vld1q_f32(y.as_ptr().add(4 * k)));
                s[k] = vaddq_f32(s[k], vmulq_f32(xv, yv));
            }
        }
        let v = vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3]));
        let p = vpaddq_f32(v, v);
        vgetq_lane_f32(p, 0) + vgetq_lane_f32(p, 1)
    };
    for (x, y) in ra.iter().zip(rb) { sum += x * y; }
    sum
}

fn main3() {
    let big: Vec<f32> = (0..64usize << 20).map(|i| (i % 7) as f32 * 0.5).collect();
    let (rows, cols, n) = (512usize, 512usize, 64usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
    let x: Vec<f32> = (0..n * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
    let mut y = vec![0.0f32; n * rows];
    let flop = 2.0 * (rows * cols * n) as f64;
    let fs: [(&str, fn(&[f32], &[f32]) -> f32); 4] = [("dot_a", dot_a), ("dot_seq", dot_seq), ("dot_lohi", dot_lohi), ("dot_neon", dot_neon)];
    for (name, f) in fs {
        let t = best(5, || { black_box(f(black_box(&big), black_box(&big))); });
        let bw = (big.len() * 4) as f64 / t / 1e9;
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                for (t, xr) in x.chunks_exact(cols).enumerate() { y[t * rows + r] = f(wr, xr); }
            }
            black_box(&y);
        });
        println!("{name}: stream {bw:.1} GB/s, matmul {:.1} GFLOP/s", flop / t / 1e9);
    }
    for cols in [512usize, 37, 100] {
        let w: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
        let xv: Vec<f32> = (0..cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
        assert_eq!(dot_neon(&w, &xv).to_bits(), dot_a(&w, &xv).to_bits(), "neon vs portable pairwise, cols {cols}");
    }
    println!("neon == portable pairwise: ok");
}

#[inline(always)]
fn fold_lohi(v: V) -> f32 { (v[0] + v[2]) + (v[1] + v[3]) }

#[inline(never)]
pub fn dot4_lohi(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = a.len() / 16 * 16;
    let mut s = [[[0.0f32; 4]; 4]; 4];
    let (w, x0, x1, x2, x3) = (&a[..n], &b[0][..n], &b[1][..n], &b[2][..n], &b[3][..n]);
    for ((((w, x0), x1), x2), x3) in w.chunks_exact(16).zip(x0.chunks_exact(16)).zip(x1.chunks_exact(16)).zip(x2.chunks_exact(16)).zip(x3.chunks_exact(16)) {
        let w: &[f32; 16] = w.try_into().unwrap();
        let xs: [&[f32; 16]; 4] = [x0.try_into().unwrap(), x1.try_into().unwrap(), x2.try_into().unwrap(), x3.try_into().unwrap()];
        for k in 0..4 {
            let wk = lo(w, k);
            for t in 0..4 { s[t][k] = add(s[t][k], mul(wk, lo(xs[t], k))); }
        }
    }
    let mut out = [0.0f32; 4];
    for t in 0..4 {
        let mut sum = fold_lohi(add(add(s[t][0], s[t][1]), add(s[t][2], s[t][3])));
        for (x, y) in a[n..].iter().zip(&b[t][n..]) { sum += x * y; }
        out[t] = sum;
    }
    out
}

fn main4() {
    let (rows, cols, n) = (512usize, 512usize, 64usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
    let x: Vec<f32> = (0..n * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
    let mut y = vec![0.0f32; n * rows];
    let flop = 2.0 * (rows * cols * n) as f64;
    for _ in 0..3 {
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                let mut t = 0;
                while t + 4 <= n {
                    let o = dot4_lohi(wr, [&x[t * cols..(t + 1) * cols], &x[(t + 1) * cols..(t + 2) * cols], &x[(t + 2) * cols..(t + 3) * cols], &x[(t + 3) * cols..(t + 4) * cols]]);
                    for j in 0..4 { y[(t + j) * rows + r] = o[j]; }
                    t += 4;
                }
            }
            black_box(&y);
        });
        println!("dot4_lohi matmul {:.1} GFLOP/s", flop / t / 1e9);
    }
    for cols in [512usize, 37, 100, 1024, 3072] {
        let w: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
        let xs: Vec<Vec<f32>> = (0..4).map(|t| (0..cols).map(|i| (((i + t * 7) * 53 % 97) as f32 - 48.0) * 1e-3).collect()).collect();
        let o = dot4_lohi(&w, [&xs[0], &xs[1], &xs[2], &xs[3]]);
        for j in 0..4 { assert_eq!(o[j].to_bits(), dot_lohi(&w, &xs[j]).to_bits()); }
    }
    println!("dot4_lohi == dot_lohi: ok");
}

#[inline(never)]
pub fn stream_u32(a: &[u32]) -> u32 {
    a.iter().fold(0u32, |s, &x| s.wrapping_add(x))
}

fn main5() {
    let big: Vec<u32> = (0..64u32 << 20).map(|i| i % 7).collect();
    let bigf: Vec<f32> = (0..64usize << 20).map(|i| (i % 7) as f32 * 0.5).collect();
    for _ in 0..3 {
        let t = best(5, || { black_box(stream_u32(black_box(&big))); });
        let t2 = best(5, || { black_box(stream32(black_box(&bigf))); });
        println!("stream_u32 {:.1} GB/s   stream32(f32) {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9, (bigf.len() * 4) as f64 / t2 / 1e9);
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("d") { return main5(); }
    if std::env::args().nth(1).as_deref() == Some("c") { return main4(); }
    if std::env::args().nth(1).as_deref() == Some("b") { return main3(); }
    if std::env::args().nth(1).as_deref() == Some("a") { return main2(); }
    let big: Vec<f32> = (0..64usize << 20).map(|i| (i % 7) as f32 * 0.5).collect();
    for _ in 0..2 {
        let t = best(5, || { black_box(stream16(black_box(&big))); });
        println!("stream16 {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9);
        let t = best(5, || { black_box(stream32(black_box(&big))); });
        println!("stream32 {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9);
        let t = best(5, || { black_box(dot16(black_box(&big), black_box(&big))); });
        println!("dot16(a,a) {:.1} GB/s", (big.len() * 4) as f64 / t / 1e9);
    }
    let (rows, cols, n) = (512usize, 512usize, 64usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3).collect();
    let x: Vec<f32> = (0..n * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3).collect();
    let mut y = vec![0.0f32; n * rows];
    let flop = 2.0 * (rows * cols * n) as f64;
    for _ in 0..2 {
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                for (t, xr) in x.chunks_exact(cols).enumerate() { y[t * rows + r] = dot16(wr, xr); }
            }
            black_box(&y);
        });
        println!("dot16 matmul {:.1} GFLOP/s", flop / t / 1e9);
        let t = best(20, || {
            for (r, wr) in w.chunks_exact(cols).enumerate() {
                let mut t = 0;
                while t + 4 <= n {
                    let o = dot16x4(wr, [&x[t * cols..(t + 1) * cols], &x[(t + 1) * cols..(t + 2) * cols], &x[(t + 2) * cols..(t + 3) * cols], &x[(t + 3) * cols..(t + 4) * cols]]);
                    for j in 0..4 { y[(t + j) * rows + r] = o[j]; }
                    t += 4;
                }
            }
            black_box(&y);
        });
        println!("dot16x4 matmul {:.1} GFLOP/s", flop / t / 1e9);
    }
    // bit identity of dot16x4 vs dot16
    let xs: Vec<&[f32]> = x.chunks_exact(cols).take(4).collect();
    for r in 0..rows {
        let wr = &w[r * cols..(r + 1) * cols];
        let o = dot16x4(wr, [xs[0], xs[1], xs[2], xs[3]]);
        for j in 0..4 { assert_eq!(o[j].to_bits(), dot16(wr, xs[j]).to_bits()); }
    }
    println!("bit-identical: ok; load {}", std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default());
}
