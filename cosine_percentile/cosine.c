/* cosine.c -- output cosine similarity between reference and quantised
 * weight matmuls.
 *
 * Pure C11. No external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c cosine.c -o cosine.o
 *
 * SIMD strategy
 * -------------
 * The host CPU is detected at compile time via immintrin.h feature-test
 * macros (__AVX512F__, __AVX2__). Three code paths are generated:
 *
 *   1. AVX-512F + BW + DQ + VL  (preferred, 16 floats / lane)
 *   2. AVX2 + FMA               (fallback, 8 floats / lane)
 *   3. scalar                   (last resort, used only when -march=native
 *                               targets a pre-AVX2 chip)
 *
 * All three accumulators (dot, ||a||^2, ||b||^2) live in fp32 SIMD lanes
 * during the inner loop, then are drained to double scalars for the
 * final reduction. This keeps the inner loop tight (3 FMA/iteration)
 * while still letting the final sqrt+divide run in double precision,
 * which is the precision-sensitive step for cosine.
 */

#include "cosine.h"

#include <math.h>
#include <immintrin.h>

/* ======================================================================
 * PRIVATE: horizontal sum helpers
 * ====================================================================== */

#if defined(__AVX512F__)

static inline float hsum_ps_avx512(__m512 v)
{
    /* 16 floats -> 1 float via __m512 reduce_add intrinsic (fast on
     * AVX-512). _mm512_reduce_add_ps is supported on AVX-512F. */
    return _mm512_reduce_add_ps(v);
}

#endif /* __AVX512F__ */

#if defined(__AVX2__) && !defined(__AVX512F__)

static inline float hsum_ps_avx2(__m256 v)
{
    /* 8 floats -> 1 float.
     * v = [a0 a1 a2 a3 a4 a5 a6 a7] (lanes 0..7)
     * Step 1: add high 4 to low 4 via permute2f128.
     * Step 2: hadd twice.
     * Step 3: extract lane 0.
     */
    __m256 shuf = _mm256_permute2f128_ps(v, v, 0x1);
    __m256 sums = _mm256_add_ps(v, shuf);
    sums = _mm256_hadd_ps(sums, sums);
    sums = _mm256_hadd_ps(sums, sums);
    return _mm256_cvtss_f32(sums);
}

#endif /* __AVX2__ && !__AVX512F__ */

/* ======================================================================
 * PUBLIC: cosine_sim
 * ======================================================================
 *
 * Compute cos(a, b) = <a,b> / (||a|| * ||b||) for two flat fp32 arrays
 * of length n.
 *
 * Returns 0.0 if either norm is zero (degenerate input).
 *
 * Implementation: 3 fp32 SIMD accumulators (dot, na2, nb2) over the
 * main aligned run, then a scalar fp64 tail. The final reduction
 * (sqrt, multiply, divide) is done in double to preserve precision.
 */
float cosine_sim(const float *a, const float *b, int n)
{
    if (!a || !b || n <= 0) return 0.0f;

    int i = 0;
    float dot_f = 0.0f, na2_f = 0.0f, nb2_f = 0.0f;

#if defined(__AVX512F__)
    __m512 v_dot = _mm512_setzero_ps();
    __m512 v_na2 = _mm512_setzero_ps();
    __m512 v_nb2 = _mm512_setzero_ps();

    int main_len = n & ~15;
    for (; i < main_len; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        v_dot = _mm512_fmadd_ps(va, vb, v_dot);
        v_na2 = _mm512_fmadd_ps(va, va, v_na2);
        v_nb2 = _mm512_fmadd_ps(vb, vb, v_nb2);
    }
    dot_f = hsum_ps_avx512(v_dot);
    na2_f = hsum_ps_avx512(v_na2);
    nb2_f = hsum_ps_avx512(v_nb2);

#elif defined(__AVX2__)
    __m256 v_dot = _mm256_setzero_ps();
    __m256 v_na2 = _mm256_setzero_ps();
    __m256 v_nb2 = _mm256_setzero_ps();

    int main_len = n & ~7;
    for (; i < main_len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        v_dot = _mm256_fmadd_ps(va, vb, v_dot);
        v_na2 = _mm256_fmadd_ps(va, va, v_na2);
        v_nb2 = _mm256_fmadd_ps(vb, vb, v_nb2);
    }
    dot_f = hsum_ps_avx2(v_dot);
    na2_f = hsum_ps_avx2(v_na2);
    nb2_f = hsum_ps_avx2(v_nb2);
#endif

    /* Scalar tail (handles trailing elements and is the only path
     * when no SIMD is enabled). Accumulate in float to match SIMD
     * precision; the final reduction is in double. */
    for (; i < n; i++) {
        dot_f += a[i] * b[i];
        na2_f += a[i] * a[i];
        nb2_f += b[i] * b[i];
    }

    /* Promote to double for the precision-sensitive reduction. */
    double dot = (double)dot_f;
    double na2 = (double)na2_f;
    double nb2 = (double)nb2_f;

    if (na2 <= 0.0 || nb2 <= 0.0) return 0.0f;
    double denom = sqrt(na2) * sqrt(nb2);
    if (denom <= 0.0) return 0.0f;
    return (float)(dot / denom);
}

/* ======================================================================
 * PUBLIC: output_cosine
 * ======================================================================
 *
 * Compute cos(Y_ref, Y_quant) where Y_ref = X@W^T, Y_quant = X@Wq^T.
 *
 * Strategy:
 *   - For each (sample, out_channel) pair, compute the two dot products
 *     X[s,:]·W[o,:] and X[s,:]·Wq[o,:] simultaneously (interleaved FMAs).
 *   - Accumulate dot(y_ref, y_quant), ||y_ref||^2, ||y_quant||^2 in fp32
 *     SIMD lanes.
 *   - OpenMP parallelise over samples; combine thread-local accumulators
 *     via reduction at the end.
 *   - Final cosine in double precision.
 *
 * Memory access pattern: each thread owns a contiguous range of samples
 * and streams both W and Wq row-by-row. The W/Wq rows for a given
 * out_channel are read by every sample in the parallel region, but the
 * per-sample access pattern reuses them once per channel -- we accept
 * the cache pressure because the alternative (loop-fission into a
 * precomputed Y_ref / Y_quant materialisation) would double memory
 * traffic on the activation side and is not justified at the typical
 * n_samples * out_dim sizes used in palettization calibration.
 */
float output_cosine(
    const float *X,     /* (n_samples, in_dim) */
    const float *W,     /* (out_dim, in_dim)   */
    const float *Wq,    /* (out_dim, in_dim)   */
    int n_samples, int out_dim, int in_dim)
{
    if (!X || !W || !Wq || n_samples <= 0 || out_dim <= 0 || in_dim <= 0)
        return 0.0f;

    /* Thread-local accumulators combined by OpenMP reduction. Using
     * scalar doubles here (not SIMD lanes) because the reduction
     * clause needs a commutative-associative combiner; we keep the
     * per-iteration FMA work in SIMD inside the loop body and just
     * sum the drained lane-totals into these doubles. */
    double g_dot = 0.0, g_na2 = 0.0, g_nb2 = 0.0;

    #pragma omp parallel reduction(+:g_dot, g_na2, g_nb2)
    {
        /* Per-thread accumulators (double, so the cross-sample reduction
         * has plenty of headroom). */
        double t_dot = 0.0, t_na2 = 0.0, t_nb2 = 0.0;

        #pragma omp for schedule(static)
        for (int s = 0; s < n_samples; s++) {
            const float *x = X + (size_t)s * in_dim;

            for (int o = 0; o < out_dim; o++) {
                const float *w  = W  + (size_t)o * in_dim;
                const float *wq = Wq + (size_t)o * in_dim;

                /* Compute y_ref = x . w  and  y_quant = x . wq. */
                float y_ref = 0.0f, y_quant = 0.0f;
                int k = 0;

#if defined(__AVX512F__)
                __m512 acc_r = _mm512_setzero_ps();
                __m512 acc_q = _mm512_setzero_ps();
                int main_len = in_dim & ~15;
                for (; k < main_len; k += 16) {
                    __m512 vx  = _mm512_loadu_ps(x  + k);
                    __m512 vw  = _mm512_loadu_ps(w  + k);
                    __m512 vwq = _mm512_loadu_ps(wq + k);
                    acc_r = _mm512_fmadd_ps(vx, vw,  acc_r);
                    acc_q = _mm512_fmadd_ps(vx, vwq, acc_q);
                }
                y_ref   = hsum_ps_avx512(acc_r);
                y_quant = hsum_ps_avx512(acc_q);

#elif defined(__AVX2__)
                __m256 acc_r = _mm256_setzero_ps();
                __m256 acc_q = _mm256_setzero_ps();
                int main_len = in_dim & ~7;
                for (; k < main_len; k += 8) {
                    __m256 vx  = _mm256_loadu_ps(x  + k);
                    __m256 vw  = _mm256_loadu_ps(w  + k);
                    __m256 vwq = _mm256_loadu_ps(wq + k);
                    acc_r = _mm256_fmadd_ps(vx, vw,  acc_r);
                    acc_q = _mm256_fmadd_ps(vx, vwq, acc_q);
                }
                y_ref   = hsum_ps_avx2(acc_r);
                y_quant = hsum_ps_avx2(acc_q);
#endif

                /* Scalar tail. */
                for (; k < in_dim; k++) {
                    y_ref   += x[k] * w[k];
                    y_quant += x[k] * wq[k];
                }

                /* Accumulate cosine components in double. */
                t_dot += (double)y_ref * (double)y_quant;
                t_na2 += (double)y_ref * (double)y_ref;
                t_nb2 += (double)y_quant * (double)y_quant;
            }
        }

        g_dot += t_dot;
        g_na2 += t_na2;
        g_nb2 += t_nb2;
    }

    if (g_na2 <= 0.0 || g_nb2 <= 0.0) return 0.0f;
    double denom = sqrt(g_na2) * sqrt(g_nb2);
    if (denom <= 0.0) return 0.0f;
    return (float)(g_dot / denom);
}
