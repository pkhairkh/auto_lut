/*
 * lloyd_max.c - Hessian-weighted Lloyd-Max 1D quantizer (C11, AVX2/AVX-512).
 *
 * See lloyd_max.h for the public API contract and algorithm overview.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -std=c11 \
 *       -mavx512f -mavx512dq -mavx512bw -mavx512vl \
 *       lloyd_max.c test_lloyd_max.c -o test_lloyd_max -lm
 *
 * The AVX-512 / AVX2 / scalar paths are selected at compile time via
 * the standard __AVX512F__ / __AVX2__ predefined macros. -march=native
 * on the build host (AMD EPYC 9B45) enables AVX-512.
 */
#include "lloyd_max.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ================================================================== */
/* 1. Paired sort                                                     */
/* ================================================================== */

typedef struct {
    float value;
    float weight;
    int   orig_idx;
} weighted_point;

static int cmp_weighted_point(const void *a, const void *b)
{
    const weighted_point *pa = (const weighted_point *)a;
    const weighted_point *pb = (const weighted_point *)b;

    /* Ascending by value. Ties broken by original index to make the
     * sort a total order (stable for duplicate values). */
    if (pa->value < pb->value) return -1;
    if (pa->value > pb->value) return  1;
    if (pa->orig_idx < pb->orig_idx) return -1;
    if (pa->orig_idx > pb->orig_idx) return  1;
    return 0;
}

/* Sort values+weights together by value. On return:
 *   pts[i].value    == the i-th smallest input value
 *   pts[i].weight   == the corresponding hessian weight
 *   pts[i].orig_idx == position of this element in the original `values` array
 */
static void paired_sort(const float *values,
                        const float *hessian,
                        int N,
                        weighted_point *pts)
{
    for (int i = 0; i < N; i++) {
        pts[i].value    = values[i];
        pts[i].weight   = hessian[i];
        pts[i].orig_idx = i;
    }
    qsort(pts, (size_t)N, sizeof(weighted_point), cmp_weighted_point);
}

/* ================================================================== */
/* 2. SIMD weighted-sum primitive                                     */
/*                                                                    */
/* Computes (sum_w, sum_wv) = (sum w_i, sum w_i * v_i) over n pairs.  */
/* Three paths: AVX-512 (16-wide), AVX2 (8-wide), scalar. Selected at */
/* compile time via __AVX512F__ / __AVX2__ predefined macros.         */
/*                                                                    */
/* We accumulate into double intermediates horizontally to avoid     */
/* catastrophic cancellation on large N (Fisher weights span many   */
/* orders of magnitude). The vector loop uses __m512/__m256 floats   */
/* for throughput; the per-lane partial sums are then promoted to     */
/* double for the final reduction.                                    */
/* ================================================================== */

static void weighted_sum_scalar(const float *v,
                                const float *w,
                                int n,
                                double *sum_w,
                                double *sum_wv)
    __attribute__((unused));
static void weighted_sum_scalar(const float *v,
                                const float *w,
                                int n,
                                double *sum_w,
                                double *sum_wv)
{
    double sw  = 0.0;
    double swv = 0.0;
    for (int i = 0; i < n; i++) {
        sw  += (double)w[i];
        swv += (double)w[i] * (double)v[i];
    }
    *sum_w  = sw;
    *sum_wv = swv;
}

#if defined(__AVX512F__)
static void weighted_sum_avx512(const float *v,
                                const float *w,
                                int n,
                                double *sum_w,
                                double *sum_wv)
{
    __m512 v_acc_w  = _mm512_setzero_ps();  /* lane-wise partial sums of w   */
    __m512 v_acc_wv = _mm512_setzero_ps();  /* lane-wise partial sums of w*v */

    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 vw = _mm512_loadu_ps(w + i);
        v_acc_w  = _mm512_add_ps(v_acc_w,  vw);
        v_acc_wv = _mm512_add_ps(v_acc_wv, _mm512_mul_ps(vv, vw));
    }

    /* Horizontal reduction: one float add per lane, then promote to double. */
    float fw  = _mm512_reduce_add_ps(v_acc_w);
    float fwv = _mm512_reduce_add_ps(v_acc_wv);
    double sw  = (double)fw;
    double swv = (double)fwv;

    /* Scalar tail for the remaining 0..15 elements. */
    for (; i < n; i++) {
        sw  += (double)w[i];
        swv += (double)w[i] * (double)v[i];
    }
    *sum_w  = sw;
    *sum_wv = swv;
}
#elif defined(__AVX2__)
static void weighted_sum_avx2(const float *v,
                              const float *w,
                              int n,
                              double *sum_w,
                              double *sum_wv)
{
    __m256 v_acc_w  = _mm256_setzero_ps();
    __m256 v_acc_wv = _mm256_setzero_ps();

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vv = _mm256_loadu_ps(v + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        v_acc_w  = _mm256_add_ps(v_acc_w,  vw);
        v_acc_wv = _mm256_add_ps(v_acc_wv, _mm256_mul_ps(vv, vw));
    }

    /* Horizontal reduction across 8 lanes. */
    __m128 hi_w  = _mm256_extractf128_ps(v_acc_w,  1);
    __m128 lo_w  = _mm256_castps256_ps128(v_acc_w);
    __m128 shuf  = _mm_movehdup_ps(lo_w);
    __m128 sums  = _mm_add_ps(lo_w, shuf);
    shuf         = _mm_movehl_ps(shuf, sums);
    sums         = _mm_add_ss(sums, shuf);
    float fw     = _mm_cvtss_f32(sums);

    __m128 hi_wv = _mm256_extractf128_ps(v_acc_wv, 1);
    (void)hi_wv;  /* symmetric reduction below */
    __m128 lo_wv = _mm256_castps256_ps128(v_acc_wv);
    __m128 shufv = _mm_movehdup_ps(lo_wv);
    __m128 sumsv = _mm_add_ps(lo_wv, shufv);
    shufv        = _mm_movehl_ps(shufv, sumsv);
    sumsv        = _mm_add_ss(sumsv, shufv);
    float fwv    = _mm_cvtss_f32(sumsv);

    /* For the high 128 bits, accumulate separately to be safe. */
    __m128 shuf_h  = _mm_movehdup_ps(hi_w);
    __m128 sums_h  = _mm_add_ps(hi_w, shuf_h);
    shuf_h         = _mm_movehl_ps(shuf_h, sums_h);
    sums_h         = _mm_add_ss(sums_h, shuf_h);
    fw            += _mm_cvtss_f32(sums_h);

    __m128 shuf_hv = _mm_movehdup_ps(_mm256_extractf128_ps(v_acc_wv, 1));
    __m128 sums_hv = _mm_add_ps(_mm256_extractf128_ps(v_acc_wv, 1), shuf_hv);
    shuf_hv         = _mm_movehl_ps(shuf_hv, sums_hv);
    sums_hv         = _mm_add_ss(sums_hv, shuf_hv);
    fwv            += _mm_cvtss_f32(sums_hv);

    double sw  = (double)fw;
    double swv = (double)fwv;

    for (; i < n; i++) {
        sw  += (double)w[i];
        swv += (double)w[i] * (double)v[i];
    }
    *sum_w  = sw;
    *sum_wv = swv;
}
#endif

static void weighted_sum(const float *v,
                         const float *w,
                         int n,
                         double *sum_w,
                         double *sum_wv)
{
#if defined(__AVX512F__)
    weighted_sum_avx512(v, w, n, sum_w, sum_wv);
#elif defined(__AVX2__)
    weighted_sum_avx2(v, w, n, sum_w, sum_wv);
#else
    weighted_sum_scalar(v, w, n, sum_w, sum_wv);
#endif
}

/* ================================================================== */
/* 3. Weighted quantile initialization                                */
/*                                                                    */
/* Partitions the sorted values into `palette` contiguous groups so  */
/* that each group carries ~1/palette of the total weight, then sets   */
/* each centroid to the weighted mean of its group.                   */
/*                                                                    */
/* Boundaries are found by binary search on the cumulative weight     */
/* prefix sum: cluster k owns indices [bounds[k], bounds[k+1]) where  */
/* bounds[k] = largest i with cum_w[i] <= k * total_w / palette.     */
/* This is O(P log N) for the boundary search + O(N) for the weighted */
/* means.                                                             */
/* ================================================================== */

static void weighted_quantile_init(const float *sorted_v,
                                   const float *sorted_w,
                                   const double *cum_w,    /* length N+1 */
                                   double total_w,
                                   int N,
                                   int palette,
                                   float *centroids)       /* length palette */
{
    int *bounds = (int *)malloc((size_t)(palette + 1) * sizeof(int));
    if (!bounds) {
        /* OOM: fall back to equal-width partitioning. */
        for (int k = 0; k < palette; k++) {
            int idx = (int)((float)k * (float)N / (float)palette);
            if (idx >= N) idx = N - 1;
            centroids[k] = sorted_v[idx];
        }
        return;
    }
    bounds[0] = 0;
    bounds[palette] = N;

    /* Binary search for largest i in [0, N] with cum_w[i] <= target. */
    for (int k = 1; k < palette; k++) {
        double target = (double)k * total_w / (double)palette;
        int lo = 0, hi = N;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (cum_w[mid] <= target) lo = mid;
            else                     hi = mid - 1;
        }
        bounds[k] = lo;
        /* Guard against empty clusters by nudging forward if previous
         * bound is already at this index. */
        if (bounds[k] <= bounds[k - 1]) {
            bounds[k] = bounds[k - 1] + 1;
            if (bounds[k] > N) bounds[k] = N;
        }
    }

    /* Enforce monotonic non-decreasing bounds. */
    for (int k = 1; k < palette; k++) {
        if (bounds[k] < bounds[k - 1]) bounds[k] = bounds[k - 1];
    }
    bounds[palette] = N;

    for (int k = 0; k < palette; k++) {
        int start = bounds[k];
        int end   = bounds[k + 1];
        int n     = end - start;
        if (n <= 0) {
            /* Empty cluster: place centroid at midpoint between neighbors
             * if possible, otherwise reuse the last value. */
            if (k > 0) {
                centroids[k] = 0.5f * (centroids[k - 1] + sorted_v[N - 1]);
            } else {
                centroids[k] = sorted_v[0];
            }
            continue;
        }
        double sw, swv;
        weighted_sum(sorted_v + start, sorted_w + start, n, &sw, &swv);
        if (sw > 0.0) {
            centroids[k] = (float)(swv / sw);
        } else {
            /* Zero total weight in this partition: pick midpoint value. */
            centroids[k] = sorted_v[(start + end) / 2];
        }
    }

    free(bounds);
}

/* ================================================================== */
/* 4. Assignment via midpoint binary search                           */
/*                                                                    */
/* Given `palette` centroids (assumed ascending - which holds for 1D  */
/* Lloyd-Max with sorted input because each cluster is a contiguous   */
/* range of sorted values and the centroid of a contiguous range      */
/* stays within that range), compute the (palette-1) midpoints         */
/* m_k = 0.5 * (c_k + c_{k+1}) and assign each value to the cluster   */
/* whose interval [m_{k-1}, m_k) contains it.                         */
/*                                                                    */
/* For each value v: find smallest k such that v <= m_k; if no such   */
/* k exists, assign to palette-1. Standard binary search:           */
/* O(N log P).                                                        */
/*                                                                    */
/* Since the input is sorted, assignments are non-decreasing; the    */
/* caller (Lloyd-Max loop) exploits this to extract contiguous runs   */
/* for the SIMD weighted-mean update.                                */
/* ================================================================== */

static void assign_midpoint(const float *sorted_v,
                            int N,
                            const float *centroids,
                            int palette,
                            uint8_t *sorted_idx)
{
    /* Midpoints buffer (palette-1 entries, max 255). */
    float midpoints[256];
    int nmid = palette - 1;
    for (int k = 0; k < nmid; k++) {
        midpoints[k] = 0.5f * (centroids[k] + centroids[k + 1]);
    }

    /* If palette == 1, everything goes to cluster 0. */
    if (nmid == 0) {
        memset(sorted_idx, 0, (size_t)N);
        return;
    }

    /* Binary search per element. Since the input is sorted, we could
     * do a single linear pass with a moving pointer, but the binary
     * search keeps the implementation simple and is still O(N log P).
     * For palette <= 256, log P <= 8, so the inner loop is at most 8
     * iterations. */
    for (int i = 0; i < N; i++) {
        float v = sorted_v[i];
        int lo = 0, hi = palette - 1;  /* cluster index in [0, palette-1] */
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (v > midpoints[mid]) lo = mid + 1;
            else                    hi = mid;
        }
        sorted_idx[i] = (uint8_t)lo;
    }
}

/* ================================================================== */
/* 5. Update step: per-cluster weighted mean                          */
/*                                                                    */
/* Walks the assignment array, identifies maximal runs of equal       */
/* cluster index, and computes the weighted mean of each run using    */
/* the SIMD weighted_sum primitive. Output: refreshed centroids.     */
/*                                                                    */
/* Empty clusters retain their previous centroid (Lloyd-Max          */
/* convention); in practice for 1D sorted input with weighted-quantile*/
/* init, empty clusters are rare.                                    */
/* ================================================================== */

static void update_centroids(const float *sorted_v,
                             const float *sorted_w,
                             int N,
                             const uint8_t *sorted_idx,
                             int palette,
                             float *centroids)
{
    /* Per-cluster accumulators. palette <= 256 so a fixed array is fine. */
    double sum_w_arr[256];
    double sum_wv_arr[256];
    for (int k = 0; k < palette; k++) {
        sum_w_arr[k]  = 0.0;
        sum_wv_arr[k] = 0.0;
    }

    /* Single pass: identify runs and accumulate. */
    int i = 0;
    while (i < N) {
        int k = sorted_idx[i];
        int j = i;
        while (j < N && sorted_idx[j] == k) j++;
        int n = j - i;
        double sw, swv;
        weighted_sum(sorted_v + i, sorted_w + i, n, &sw, &swv);
        sum_w_arr[k]  = sw;
        sum_wv_arr[k] = swv;
        i = j;
    }

    /* Refresh centroids where the cluster has positive weight. */
    for (int k = 0; k < palette; k++) {
        if (sum_w_arr[k] > 0.0) {
            centroids[k] = (float)(sum_wv_arr[k] / sum_w_arr[k]);
        }
        /* Else: keep previous centroid (handled by not overwriting). */
    }
}

/* ================================================================== */
/* 6. Public entry point: hessian_lloyd_max()                         */
/*                                                                    */
/* Pipeline:                                                          */
/*   paired_sort  -> weighted_quantile_init                          */
/*                -> Lloyd-Max iteration { assign_midpoint           */
/*                                           update_centroids }      */
/*                -> map assignments back to original order           */
/* ================================================================== */

float hessian_lloyd_max(const float *values,
                        const float *hessian,
                        int N,
                        int palette,
                        int max_iter,
                        float *lut_out,
                        uint8_t *idx_out)
{
    /* ---------------- input validation ---------------- */
    if (!values || !hessian || !lut_out || !idx_out) return -1.0f;
    if (N <= 0 || palette <= 0) return -1.0f;
    if (palette > 256) palette = 256;
    if (max_iter < 1) max_iter = 1;

    /* ---------------- allocations ---------------- */
    weighted_point *pts        = (weighted_point *)malloc((size_t)N * sizeof(weighted_point));
    float           *sorted_v  = (float *)malloc((size_t)N * sizeof(float));
    float           *sorted_w  = (float *)malloc((size_t)N * sizeof(float));
    double          *cum_w     = (double *)malloc((size_t)(N + 1) * sizeof(double));
    float           *old_lut   = (float *)malloc((size_t)palette * sizeof(float));
    uint8_t         *sorted_idx = (uint8_t *)malloc((size_t)N);

    if (!pts || !sorted_v || !sorted_w || !cum_w || !old_lut || !sorted_idx) {
        free(pts); free(sorted_v); free(sorted_w);
        free(cum_w); free(old_lut); free(sorted_idx);
        return -1.0f;
    }

    /* ---------------- stage 1: paired sort ---------------- */
    paired_sort(values, hessian, N, pts);

    for (int i = 0; i < N; i++) {
        sorted_v[i] = pts[i].value;
        sorted_w[i] = pts[i].weight;
    }

    /* Cumulative weight prefix sum (length N+1, cum_w[0] = 0). */
    cum_w[0] = 0.0;
    for (int i = 0; i < N; i++) {
        /* Clamp negative weights to 0 (Fisher diagonal is >= 0, but
         * defend against numerical noise). */
        double w = (sorted_w[i] < 0.0f) ? 0.0 : (double)sorted_w[i];
        cum_w[i + 1] = cum_w[i] + w;
    }
    double total_w = cum_w[N];

    /* If all weights are zero, fall back to uniform weights so the
     * quantizer still produces a sensible LUT. */
    if (total_w <= 0.0) {
        for (int i = 0; i < N; i++) {
            sorted_w[i] = 1.0f;
            cum_w[i + 1] = (double)(i + 1);
        }
        total_w = (double)N;
    }

    /* ---------------- stage 2: weighted quantile init ---------------- */
    weighted_quantile_init(sorted_v, sorted_w, cum_w, total_w,
                            N, palette, lut_out);

    /* ---------------- stage 3: Lloyd-Max iteration ---------------- */
    float last_delta = 0.0f;
    for (int iter = 0; iter < max_iter; iter++) {
        /* Save old levels for convergence check. */
        memcpy(old_lut, lut_out, (size_t)palette * sizeof(float));

        /* Assignment: midpoint binary search. */
        assign_midpoint(sorted_v, N, lut_out, palette, sorted_idx);

        /* Update: weighted mean per cluster. */
        update_centroids(sorted_v, sorted_w, N, sorted_idx, palette, lut_out);

        /* Convergence: max|new - old| < 1e-6. */
        float max_delta = 0.0f;
        for (int k = 0; k < palette; k++) {
            float d = fabsf(lut_out[k] - old_lut[k]);
            if (d > max_delta) max_delta = d;
        }
        last_delta = max_delta;
        if (max_delta < 1e-6f) break;
    }

    /* After the final update, recompute the assignment so that idx_out
     * is consistent with the final lut_out. (The loop above leaves
     * sorted_idx as the assignment for the *previous* centroids.) */
    assign_midpoint(sorted_v, N, lut_out, palette, sorted_idx);

    /* ---------------- stage 4: map back to original order ---------------- */
    for (int i = 0; i < N; i++) {
        idx_out[pts[i].orig_idx] = sorted_idx[i];
    }

    /* ---------------- cleanup ---------------- */
    free(pts);
    free(sorted_v);
    free(sorted_w);
    free(cum_w);
    free(old_lut);
    free(sorted_idx);

    return last_delta;
}
