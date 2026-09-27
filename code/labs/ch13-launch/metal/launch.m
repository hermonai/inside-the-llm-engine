// Chapter 13 lab, GPU extension for Apple silicon: what a Metal dispatch costs.
//
// The Rust lab's chain of n dependent additions, x += c_k, run on the GPU in
// five ways:
//
//   sync     one command buffer per addition, committed and waited for
//   queued   one command buffer per addition, all committed, one wait at the end
//   batched  one command buffer holding all n dispatches in a serial encoder
//   barriers the same, in a concurrent encoder with a memory barrier between
//            dependent dispatches, as llama.cpp encodes a token's graph
//   fused    one dispatch whose kernel applies all n additions
//
// Every result is compared bit for bit with the CPU before anything is timed:
// the kernels (metal/kernels.metal) are compiled with safe math, so the GPU
// performs the same float additions in the same order. The command buffers'
// GPU timestamps separate time spent on the GPU from time spent getting work
// to it and back.
//
// From code/labs/ch13-launch, on a Mac:
//
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework Metal metal/launch.m -o ../target/ch13-metal
//   ../target/ch13-metal [--ops 1000] [--sizes 1024,65536,1048576] [--reps 5]
//       [--kernels metal/kernels.metal]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SYNC, QUEUED, BATCHED, BARRIERS, FUSED, METHODS };
static const char *kNames[METHODS] = {"sync", "queued", "batched", "barriers", "fused"};

static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> addOne, addAll;

static double now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

// The same inputs and constants as the Rust lab.
static void fill(float *x, size_t n, uint64_t seed, float scale) {
    uint64_t s = (seed * 0x9E3779B97F4A7C15ull) | 1;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        x[i] = ((float)(s >> 40) / (float)(1ull << 24) - 0.5f) * scale;
    }
}

static MTLSize group(id<MTLComputePipelineState> p, size_t n) {
    NSUInteger t = MIN(p.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    return MTLSizeMake(MIN(t, (NSUInteger)n), 1, 1);
}

static void encodeAdd(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, size_t n, float c) {
    [enc setComputePipelineState:addOne];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBytes:&c length:sizeof c atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:group(addOne, n)];
}

typedef struct {
    double host, gpu, encode;  // seconds for the whole chain
} Timing;

// Runs the chain once, one of five ways, and waits for it to finish.
static Timing run(int method, id<MTLBuffer> x, size_t n, const float *cs, id<MTLBuffer> csBuf,
                  uint32_t ops) {
    Timing t = {0, 0, 0};
    double t0 = now();
    if (method == SYNC) {
        for (uint32_t k = 0; k < ops; k++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encodeAdd(enc, x, n, cs[k]);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            t.gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
    } else if (method == QUEUED) {
        id<MTLCommandBuffer> first = nil, last = nil;
        for (uint32_t k = 0; k < ops; k++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encodeAdd(enc, x, n, cs[k]);
            [enc endEncoding];
            [cb commit];
            if (!first) first = cb;
            last = cb;
        }
        [last waitUntilCompleted];
        t.gpu = last.GPUEndTime - first.GPUStartTime;
    } else {
        // A serial encoder runs each dispatch after the previous one and shows
        // it that one's writes; a concurrent encoder may overlap dispatches
        // unless a barrier separates them.
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            method == BARRIERS
                ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [cb computeCommandEncoder];
        if (method == BATCHED || method == BARRIERS) {
            for (uint32_t k = 0; k < ops; k++) {
                if (method == BARRIERS && k > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeAdd(enc, x, n, cs[k]);
            }
        } else {
            [enc setComputePipelineState:addAll];
            [enc setBuffer:x offset:0 atIndex:0];
            [enc setBuffer:csBuf offset:0 atIndex:1];
            [enc setBytes:&ops length:sizeof ops atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:group(addAll, n)];
        }
        [enc endEncoding];
        t.encode = now() - t0;
        [cb commit];
        [cb waitUntilCompleted];
        t.gpu = cb.GPUEndTime - cb.GPUStartTime;
    }
    t.host = now() - t0;
    return t;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof *v, cmp);
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

static const char *arg(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return NULL;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *a;
        uint32_t ops = (a = arg(argc, argv, "--ops")) ? (uint32_t)atoi(a) : 1000;
        int reps = (a = arg(argc, argv, "--reps")) ? atoi(a) : 5;
        const char *sizeList = (a = arg(argc, argv, "--sizes")) ? a : "1024,65536,1048576";
        const char *kernels = (a = arg(argc, argv, "--kernels")) ? a : "metal/kernels.metal";

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        queue = [dev newCommandQueue];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opt.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *err = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(kernels)
                                                     encoding:NSUTF8StringEncoding
                                                        error:&err];
        if (!source) {
            fprintf(stderr, "cannot read %s: run from code/labs/ch13-launch or pass --kernels\n",
                    kernels);
            return 1;
        }
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:opt error:&err];
        if (!lib) {
            fprintf(stderr, "shader compile failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        addOne = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_one"] error:&err];
        addAll = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_all"] error:&err];
        if (!addOne || !addAll) {
            fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        float *cs = malloc(ops * sizeof(float));
        for (uint32_t k = 0; k < ops; k++) cs[k] = ((float)(k % 7) - 3.0f) * 1e-3f;
        id<MTLBuffer> csBuf = [dev newBufferWithBytes:cs length:ops * sizeof(float)
                                              options:MTLResourceStorageModeShared];

        printf("Metal on %s: a chain of %u dependent additions x += c\n", dev.name.UTF8String, ops);
        printf("every method checked against the CPU bit for bit, then timed\n");
        printf("microseconds per addition, median of %d runs: host = wall clock around the\n", reps);
        printf("chain, gpu = GPU timestamps, encode = host time to record the command buffer\n\n");
        printf("  %9s  %-8s %9s %9s %9s\n", "elements", "method", "host", "gpu", "encode");

        char *sizes = strdup(sizeList), *save = NULL;
        for (char *tok = strtok_r(sizes, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            size_t n = (size_t)atol(tok);
            float *x0 = malloc(n * sizeof(float)), *want = malloc(n * sizeof(float));
            fill(x0, n, 1, 2.0f);
            for (size_t i = 0; i < n; i++) {
                float v = x0[i];
                for (uint32_t k = 0; k < ops; k++) v += cs[k];
                want[i] = v;
            }
            id<MTLBuffer> x = [dev newBufferWithLength:n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
            for (int m = 0; m < METHODS; m++) {
                memcpy(x.contents, x0, n * sizeof(float));
                run(m, x, n, cs, csBuf, ops);
                if (memcmp(x.contents, want, n * sizeof(float))) {
                    fprintf(stderr, "oracle failed: %s at %zu elements\n", kNames[m], n);
                    return 1;
                }
                double host[reps], gpu[reps], encode[reps];
                for (int r = 0; r < reps; r++) {
                    Timing t = run(m, x, n, cs, csBuf, ops);
                    host[r] = t.host;
                    gpu[r] = t.gpu;
                    encode[r] = t.encode;
                }
                double us = 1e6 / ops;
                if (m >= BATCHED)
                    printf("  %9zu  %-8s %9.3f %9.3f %9.3f\n", n, kNames[m], median(host, reps) * us,
                           median(gpu, reps) * us, median(encode, reps) * us);
                else
                    printf("  %9zu  %-8s %9.3f %9.3f %9s\n", n, kNames[m], median(host, reps) * us,
                           median(gpu, reps) * us, "-");
            }
            free(x0);
            free(want);
        }
        printf("\n  bits: all five methods matched the CPU at every size\n");
        free(sizes);
        free(cs);
    }
    return 0;
}
