// Chapter 13 lab, GPU extension: the two kernels, compiled at run time by
// launch.m with safe math, so every addition is rounded as on the CPU.
#include <metal_stdlib>
using namespace metal;

// One addition: thread i adds c to element i. The host launches exactly as
// many threads as elements, so there is no bounds check.
kernel void add_one(device float *x [[buffer(0)]],
                    constant float &c [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    x[i] += c;
}

// The fused chain: thread i keeps element i in a register while it applies
// all n additions, in order, then writes it back once.
kernel void add_all(device float *x [[buffer(0)]],
                    constant float *c [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    float v = x[i];
    for (uint k = 0; k < n; ++k) v += c[k];
    x[i] = v;
}
