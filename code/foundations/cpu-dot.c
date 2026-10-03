/* Native CPU correctness lab. No timings; no alignment promise beyond float. */
#include <stddef.h>
#include <stdio.h>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

float scalar_dot(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float lane_dot(const float *a, const float *b, size_t n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t lanes = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; n - i >= 4; i += 4) {
        lanes = vaddq_f32(lanes, vmulq_f32(vld1q_f32(a+i), vld1q_f32(b+i)));
    }
    float sum = vaddvq_f32(lanes);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    return scalar_dot(a, b, n);
#endif
}

int main(void) {
    float a[258], b[258];
    for (size_t i = 0; i < 258; ++i) {
        a[i] = (float)((int)(i % 13) - 6) / 4.0f;
        b[i] = (float)((int)(i % 7) - 3) / 2.0f;
    }
    size_t checks = 0;
    for (size_t offset = 0; offset < 2; ++offset) {
        for (size_t n = 0; n <= 257; ++n) {
            int numerator = 0;
            for (size_t i = offset; i < offset+n; ++i)
                numerator += ((int)(i % 13)-6) * ((int)(i % 7)-3);
            float expected = (float)numerator / 8.0f;
            if (scalar_dot(a+offset,b+offset,n) != expected ||
                lane_dot(a+offset,b+offset,n) != expected) {
                fprintf(stderr,"mismatch n=%zu offset=%zu\n",n,offset);
                return 1;
            }
            ++checks;
        }
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    printf("path=AArch64 NEON, four F32 lanes; ");
#else
    printf("path=scalar fallback; ");
#endif
    printf("checked=%zu length/offset cases; exact dyadic oracle passed\n",checks);
    return 0;
}
