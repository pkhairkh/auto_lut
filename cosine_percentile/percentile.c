/* percentile.c -- percentile-based outlier clipping for weight matrices.
 *
 * Pure C11. No external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c percentile.c -o percentile.o
 *
 * Two public functions:
 *   - percentile(sorted_arr, n, pct): linear-interpolated percentile of
 *     an already-sorted array. Identical to numpy.quantile with
 *     interpolation='linear'. pct is a FRACTION in [0, 1].
 *   - clip_outliers(W, out_dim, in_dim): per-channel clip of |W[o,:]|
 *     at the 99.95th percentile (pct = 0.9995 by default).
 *
 * No SIMD: the per-channel work is dominated by the qsort and the clip
 * pass is memory-bound. OpenMP is used to parallelise across channels
 * (each channel's sort+clip is independent).
 */

/* Request posix_memalign() from stdlib.h. Must be defined before any
 * system header is included. C11 guarantees _POSIX_C_SOURCE >= 200809L
 * exposes posix_memalign on conforming platforms. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "percentile.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * PRIVATE: qsort comparator
 * ====================================================================== */

static int cmp_float_asc(const void *pa, const void *pb)
{
    float a = *(const float *)pa;
    float b = *(const float *)pb;
    if (a < b) return -1;
    if (a > b) return  1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* private helper: portable aligned allocation                        */
/* ------------------------------------------------------------------ */
//
// On POSIX we prefer posix_memalign(32) so the scratch buffer is
// 32-byte aligned (good for AVX _mm256_loadu_ps which can use the
// aligned-load fast path on most modern CPUs even with the 'u' suffix).
// On non-POSIX we fall back to plain malloc; correctness is identical,
// only alignment differs.
//
// Declared here (before clip_outliers) so it can be used inline.
// posix_memalign is available because we defined _POSIX_C_SOURCE at
// the top of this file before including stdlib.h.
static void *_aligned_alloc_or_malloc(size_t n)
{
    void *p = NULL;
    if (n == 0) n = 1;  /* posix_memalign rejects 0 */
    if (posix_memalign(&p, 32, n) != 0) return NULL;
    return p;
}

/* ======================================================================
 * PUBLIC: percentile
 * ======================================================================
 *
 * Linear-interpolation percentile of an already-sorted ascending array.
 *
 *   rank  = pct * (n - 1)             (pct in [0, 1])
 *   lo    = (int)floor(rank)
 *   hi    = (int)ceil(rank)
 *   frac  = rank - lo
 *   value = sorted_arr[lo] + frac * (sorted_arr[hi] - sorted_arr[lo])
 *
 * Edge cases:
 *   n <= 0           -> 0.0
 *   n == 1           -> sorted_arr[0] for any pct
 *   pct < 0          -> clamped to 0  (returns sorted_arr[0])
 *   pct > 1          -> clamped to 1  (returns sorted_arr[n-1])
 *   sorted_arr NULL  -> 0.0
 */
float percentile(float *sorted_arr, int n, float pct)
{
    if (!sorted_arr || n <= 0) return 0.0f;
    if (n == 1) return sorted_arr[0];

    /* Clamp pct to [0, 1]. */
    if (pct < 0.0f) pct = 0.0f;
    if (pct > 1.0f) pct = 1.0f;

    /* Linear-interpolation rank in [0, n-1]. */
    double rank = (double)pct * (double)(n - 1);

    /* Clamp rank to valid range (defensive against fp rounding). */
    if (rank < 0.0)            rank = 0.0;
    if (rank > (double)(n-1))  rank = (double)(n - 1);

    int lo = (int)floor(rank);
    int hi = (int)ceil(rank);
    if (lo < 0)       lo = 0;
    if (hi > n - 1)   hi = n - 1;
    if (lo > hi) { int t = lo; lo = hi; hi = t; }

    double frac = rank - (double)lo;
    double v_lo = (double)sorted_arr[lo];
    double v_hi = (double)sorted_arr[hi];
    return (float)(v_lo + frac * (v_hi - v_lo));
}

/* ======================================================================
 * PUBLIC: clip_outliers
 * ======================================================================
 *
 * For each output channel o (0..out_dim-1):
 *   1. Allocate a temp buffer of length in_dim.
 *   2. Fill it with |W[o, i]| for i in 0..in_dim-1.
 *   3. qsort ascending.
 *   4. clip_val = percentile(temp, in_dim, PERCENTILE_CLIP_PCT).
 *   5. For every i: if |W[o, i]| > clip_val, replace W[o, i] with
 *      copysign(clip_val, W[o, i]).
 *   6. Track whether any element was actually clamped; if yes,
 *      increment the channel-clipped counter.
 *
 * Returns: number of output channels with at least one clamped element.
 *
 * Parallelism: channels are independent -> OpenMP parallel for.
 * Each thread allocates its own temp buffer (size in_dim * sizeof(float))
 * once before the loop and reuses it across channels in its range.
 *
 * Edge cases:
 *   W == NULL, out_dim <= 0, or in_dim <= 0 -> return 0 (no-op).
 *   in_dim == 1 -> percentile is just W[o,0]; the clip is a no-op
 *                  (|W[o,0]| > |W[o,0]| is never true); returns 0.
 */
int clip_outliers(float *W, int out_dim, int in_dim)
{
    if (!W || out_dim <= 0 || in_dim <= 0) return 0;

    int clipped_channels = 0;

    #pragma omp parallel reduction(+:clipped_channels)
    {
        /* Per-thread scratch buffer; allocated once, reused per channel.
         * Aligned to 32 bytes for SIMD-friendly access in the copy pass
         * (also fine for qsort which doesn't care about alignment). */
        float *abs_buf = (float *)_aligned_alloc_or_malloc((size_t)in_dim * sizeof(float));

        #pragma omp for schedule(static)
        for (int o = 0; o < out_dim; o++) {
            float *row = W + (size_t)o * in_dim;

            /* Step 1+2: copy |W[o,:]| into abs_buf. */
            for (int i = 0; i < in_dim; i++) {
                float v = row[i];
                abs_buf[i] = fabsf(v);
            }

            /* Step 3: sort ascending. */
            qsort(abs_buf, (size_t)in_dim, sizeof(float), cmp_float_asc);

            /* Step 4: compute clip threshold. */
            float clip_val = percentile(abs_buf, in_dim, PERCENTILE_CLIP_PCT);

            /* Step 5: clamp. We track whether any element was actually
             * modified so the counter is exact (not just "channel had
             * an element >= clip_val"). */
            int channel_clipped = 0;
            for (int i = 0; i < in_dim; i++) {
                float v = row[i];
                float av = fabsf(v);
                if (av > clip_val) {
                    /* copysignf preserves the sign of the original
                     * value while replacing magnitude with clip_val. */
                    row[i] = copysignf(clip_val, v);
                    channel_clipped = 1;
                }
            }

            if (channel_clipped) clipped_channels += 1;
        }

        free(abs_buf);
    }

    return clipped_channels;
}
