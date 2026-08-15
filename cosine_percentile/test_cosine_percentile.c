/* test_cosine_percentile.c -- DoD tests for cosine.c + percentile.c.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -std=c11 \
 *       test_cosine_percentile.c cosine.c percentile.c \
 *       -o test_cosine_percentile -lm
 *
 * Run:
 *   ./test_cosine_percentile
 *
 * Exit code: 0 if all tests pass, 1 if any test fails. The test harness
 * prints one line per test case ("PASS" or "FAIL") followed by a final
 * summary. Failures print the expected vs actual values for debugging.
 *
 * Test cases (mapped to the DoD in the task brief):
 *
 *   [cosine_sim]
 *     1. identical vectors     -> cos == 1.0 (within 1e-6)
 *     2. opposite vectors      -> cos == -1.0 (within 1e-6)
 *     3. orthogonal vectors    -> cos == 0.0 (within 1e-6)
 *     4. zero vector           -> cos == 0.0 (degenerate input)
 *     5. random length-N       -> matches naive scalar reference
 *     6. length not multiple of 16 (tail path) -> matches reference
 *
 *   [output_cosine]
 *     7. W == Wq (perfect)     -> cos == 1.0 (within 1e-6)
 *     8. Wq == -W (sign flip)  -> cos == -1.0 (within 1e-6)
 *     9. Wq orthogonal to W    -> cos ~= 0.0 (within 1e-5)
 *    10. known small case      -> matches scalar reference computation
 *
 *   [percentile]                       (pct is a FRACTION in [0, 1])
 *    11. p=0    -> min
 *    12. p=1    -> max
 *    13. p=0.5  -> median (n odd) and interpolated median (n even)
 *    14. p=0.9995 on a known array -> matches numpy.quantile semantics
 *
 *   [clip_outliers]
 *    15. known data with 1 outlier channel: only that channel is clipped,
 *        normal channels are untouched. Returns 1.
 *    16. multi-channel case with 2/3 channels having outliers -> returns 2.
 *    17. all-equal channel -> nothing clipped, returns 0.
 *
 * Note on clip_outliers test design:
 *   With linear-interpolated percentile and a single extreme outlier at
 *   the max position, the percentile rank must land strictly below
 *   index (n-1) so that the linear interpolation does NOT mix the
 *   outlier into clip_val. We need:
 *     pct * (n - 1) <= n - 2
 *   For pct = 0.9995 this requires n >= 2001. We use n = 10000 in the
 *   clip_outliers tests for ample headroom.
 *
 *   We also use *baseline-equal* values for the "normal" elements
 *   (e.g. all 1.0). With all equal values, the percentile equals the
 *   baseline exactly, so no element is strictly greater than clip_val
 *   and the channel is genuinely untouched. This makes the DoD test
 *   "doesn't touch normal channels" meaningful.
 */

#include "cosine.h"
#include "percentile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Tiny test framework                                                 */
/* ------------------------------------------------------------------ */

static int g_pass_count = 0;
static int g_fail_count = 0;

#define EPS_ABS 1e-6f

static int approx_eq(float a, float b, float eps)
{
    float d = fabsf(a - b);
    if (d <= eps) return 1;
    /* Relative tolerance for large magnitudes. */
    float m = fmaxf(fabsf(a), fabsf(b));
    if (m > 1.0f) return d <= eps * m;
    return 0;
}

static void report(const char *name, int ok,
                   float expected, float actual)
{
    if (ok) {
        printf("PASS  %s\n", name);
        g_pass_count++;
    } else {
        printf("FAIL  %s  (expected=%.8g, actual=%.8g)\n",
               name, expected, actual);
        g_fail_count++;
    }
}

static void report_int(const char *name, int ok,
                       int expected, int actual)
{
    if (ok) {
        printf("PASS  %s\n", name);
        g_pass_count++;
    } else {
        printf("FAIL  %s  (expected=%d, actual=%d)\n",
               name, expected, actual);
        g_fail_count++;
    }
}

/* ------------------------------------------------------------------ */
/* Scalar reference implementations (for cross-checking the SIMD path) */
/* ------------------------------------------------------------------ */

static float cosine_sim_scalar(const float *a, const float *b, int n)
{
    if (!a || !b || n <= 0) return 0.0f;
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    if (na2 <= 0.0 || nb2 <= 0.0) return 0.0f;
    double denom = sqrt(na2) * sqrt(nb2);
    return (float)(dot / denom);
}

static float output_cosine_scalar(
    const float *X, const float *W, const float *Wq,
    int n_samples, int out_dim, int in_dim)
{
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (int s = 0; s < n_samples; s++) {
        const float *x = X + (size_t)s * in_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w  = W  + (size_t)o * in_dim;
            const float *wq = Wq + (size_t)o * in_dim;
            double yr = 0.0, yq = 0.0;
            for (int k = 0; k < in_dim; k++) {
                yr += (double)x[k] * w[k];
                yq += (double)x[k] * wq[k];
            }
            dot += yr * yq;
            na2 += yr * yr;
            nb2 += yq * yq;
        }
    }
    if (na2 <= 0.0 || nb2 <= 0.0) return 0.0f;
    double denom = sqrt(na2) * sqrt(nb2);
    return (float)(dot / denom);
}

static float percentile_scalar(float *sorted, int n, float pct)
{
    if (!sorted || n <= 0) return 0.0f;
    if (n == 1) return sorted[0];
    if (pct < 0.0f) pct = 0.0f;
    if (pct > 1.0f)  pct = 1.0f;
    double rank = (double)pct * (double)(n - 1);
    if (rank < 0.0) rank = 0.0;
    if (rank > (double)(n - 1)) rank = (double)(n - 1);
    int lo = (int)floor(rank);
    int hi = (int)ceil(rank);
    if (lo < 0) lo = 0;
    if (hi > n - 1) hi = n - 1;
    double frac = rank - lo;
    return (float)((double)sorted[lo] +
                   frac * ((double)sorted[hi] - (double)sorted[lo]));
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void fill_random(float *a, int n, unsigned int *seed)
{
    for (int i = 0; i < n; i++) {
        /* Range [-1, 1]. */
        a[i] = ((float)rand_r(seed) / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

/* Comparator for qsort ascending by float value. */
static int cmp_test_float_asc(const void *pa, const void *pb)
{
    float a = *(const float *)pa;
    float b = *(const float *)pb;
    if (a < b) return -1;
    if (a > b) return  1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* cosine_sim tests                                                    */
/* ------------------------------------------------------------------ */

static void test_cosine_sim_identical(void)
{
    /* Identical non-zero vectors -> cos = 1.0. Use length 32 so the
     * AVX-512 path runs two full iterations. */
    float a[32];
    unsigned int seed = 42;
    fill_random(a, 32, &seed);
    float r = cosine_sim(a, a, 32);
    report("cosine_sim(identical, n=32) == 1.0",
           approx_eq(r, 1.0f, EPS_ABS), 1.0f, r);
}

static void test_cosine_sim_opposite(void)
{
    /* Opposite vectors -> cos = -1.0. */
    float a[32], b[32];
    unsigned int seed = 7;
    fill_random(a, 32, &seed);
    for (int i = 0; i < 32; i++) b[i] = -a[i];
    float r = cosine_sim(a, b, 32);
    report("cosine_sim(opposite, n=32) == -1.0",
           approx_eq(r, -1.0f, EPS_ABS), -1.0f, r);
}

static void test_cosine_sim_orthogonal(void)
{
    /* Orthogonal: a = [1, 0, 1, 0, ...], b = [0, 1, 0, 1, ...]
     * Length 16 (single AVX-512 iteration). */
    float a[16], b[16];
    for (int i = 0; i < 16; i++) {
        a[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        b[i] = (i % 2 == 0) ? 0.0f : 1.0f;
    }
    float r = cosine_sim(a, b, 16);
    report("cosine_sim(orthogonal, n=16) == 0.0",
           approx_eq(r, 0.0f, EPS_ABS), 0.0f, r);
}

static void test_cosine_sim_zero_vector(void)
{
    /* One vector is all zero -> degenerate, return 0.0. */
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float r = cosine_sim(a, b, 8);
    report("cosine_sim(zero_vector) == 0.0",
           approx_eq(r, 0.0f, EPS_ABS), 0.0f, r);
}

static void test_cosine_sim_random_matches_scalar(void)
{
    /* Random vectors of length 1000 (not multiple of 16 -> exercises
     * both SIMD body and scalar tail). Must match scalar reference. */
    int n = 1000;
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    unsigned int seed = 1234;
    fill_random(a, n, &seed);
    fill_random(b, n, &seed);

    float r_simd  = cosine_sim(a, b, n);
    float r_scalar = cosine_sim_scalar(a, b, n);
    /* Use relative tolerance since values are O(1) but the dot can be
     * large. */
    int ok = approx_eq(r_simd, r_scalar, 1e-5f);
    report("cosine_sim(random, n=1000) matches scalar reference",
           ok, r_scalar, r_simd);
    free(a);
    free(b);
}

static void test_cosine_sim_tail_path(void)
{
    /* Length 17 = 1 SIMD iter (16) + 1 scalar. Verifies the tail loop
     * is reached. */
    float a[17], b[17];
    unsigned int seed = 99;
    fill_random(a, 17, &seed);
    fill_random(b, 17, &seed);
    float r_simd = cosine_sim(a, b, 17);
    float r_sca  = cosine_sim_scalar(a, b, 17);
    int ok = approx_eq(r_simd, r_sca, 1e-5f);
    report("cosine_sim(n=17, tail path) matches scalar reference",
           ok, r_sca, r_simd);
}

/* ------------------------------------------------------------------ */
/* output_cosine tests                                                 */
/* ------------------------------------------------------------------ */

static void test_output_cosine_perfect(void)
{
    /* W == Wq -> cos = 1.0. */
    int n_samples = 4, out_dim = 8, in_dim = 32;
    float *X  = (float *)malloc((size_t)n_samples * in_dim * sizeof(float));
    float *W  = (float *)malloc((size_t)out_dim   * in_dim * sizeof(float));
    float *Wq = (float *)malloc((size_t)out_dim   * in_dim * sizeof(float));
    unsigned int seed = 1;
    fill_random(X, n_samples * in_dim, &seed);
    fill_random(W, out_dim   * in_dim, &seed);
    memcpy(Wq, W, (size_t)out_dim * in_dim * sizeof(float));

    float r = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
    report("output_cosine(W==Wq) == 1.0",
           approx_eq(r, 1.0f, EPS_ABS), 1.0f, r);

    free(X); free(W); free(Wq);
}

static void test_output_cosine_sign_flip(void)
{
    /* Wq == -W -> cos = -1.0. */
    int n_samples = 4, out_dim = 8, in_dim = 32;
    float *X  = (float *)malloc((size_t)n_samples * in_dim * sizeof(float));
    float *W  = (float *)malloc((size_t)out_dim   * in_dim * sizeof(float));
    float *Wq = (float *)malloc((size_t)out_dim   * in_dim * sizeof(float));
    unsigned int seed = 2;
    fill_random(X, n_samples * in_dim, &seed);
    fill_random(W, out_dim   * in_dim, &seed);
    for (int i = 0; i < out_dim * in_dim; i++) Wq[i] = -W[i];

    float r = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
    report("output_cosine(Wq==-W) == -1.0",
           approx_eq(r, -1.0f, EPS_ABS), -1.0f, r);

    free(X); free(W); free(Wq);
}

static void test_output_cosine_orthogonal(void)
{
    /* Constructed orthogonal case:
     *   X = [[1, 1], [0, 1]] (in_dim = 2, n_samples = 2)
     *   W[0]  = [1,  1]
     *   Wq[0] = [1, -1]
     * Then:
     *   Y_ref[0]   = X[0].W[0]  = 2
     *   Y_ref[1]   = X[1].W[0]  = 1
     *   Y_quant[0] = X[0].Wq[0] = 0
     *   Y_quant[1] = X[1].Wq[0] = -1
     * dot = 2*0 + 1*(-1) = -1
     * ||Y_ref||^2   = 4 + 1 = 5
     * ||Y_quant||^2 = 0 + 1 = 1
     * cos = -1 / (sqrt(5) * sqrt(1)) = -1/sqrt(5) ~= -0.4472
     *
     * For a TRUE orthogonal case (cos = 0), use:
     *   X = [[1, 0], [0, 1]]
     *   W[0]  = [1,  1]
     *   Wq[0] = [1, -1]
     * Then:
     *   Y_ref   = [1*1+0*1, 0*1+1*1]  = [1, 1]
     *   Y_quant = [1*1+0*(-1), 0*1+1*(-1)] = [1, -1]
     * dot = 1*1 + 1*(-1) = 0
     * ||Y_ref||^2   = 2
     * ||Y_quant||^2 = 2
     * cos = 0 / (sqrt(2) * sqrt(2)) = 0
     *
     * in_dim = 2 is below AVX-512 width, exercising only the scalar
     * tail. The SIMD path is exercised by other tests.
     */
    int n_samples = 2, out_dim = 1, in_dim = 2;
    float X[4]  = {1, 0, 0, 1};
    float W[2]  = {1, 1};
    float Wq[2] = {1, -1};
    float r = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
    report("output_cosine(orthogonal, small) == 0.0",
           approx_eq(r, 0.0f, EPS_ABS), 0.0f, r);
}

static void test_output_cosine_matches_scalar(void)
{
    /* Random non-trivial case: must match scalar reference. */
    int n_samples = 5, out_dim = 7, in_dim = 33; /* odd in_dim -> tail */
    int nx = n_samples * in_dim;
    int nw = out_dim   * in_dim;
    float *X  = (float *)malloc((size_t)nx * sizeof(float));
    float *W  = (float *)malloc((size_t)nw * sizeof(float));
    float *Wq = (float *)malloc((size_t)nw * sizeof(float));
    unsigned int seed = 55;
    fill_random(X,  nx, &seed);
    fill_random(W,  nw, &seed);
    fill_random(Wq, nw, &seed);

    float r_simd  = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
    float r_scalar = output_cosine_scalar(X, W, Wq, n_samples, out_dim, in_dim);
    int ok = approx_eq(r_simd, r_scalar, 1e-5f);
    report("output_cosine(random, 5x7x33) matches scalar reference",
           ok, r_scalar, r_simd);
    free(X); free(W); free(Wq);
}

/* ------------------------------------------------------------------ */
/* percentile tests                                                    */
/* ------------------------------------------------------------------ */

static void test_percentile_min_max_median(void)
{
    /* Sorted array 0..99 (n=100). */
    int n = 100;
    float *arr = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) arr[i] = (float)i;

    float p0   = percentile(arr, n, 0.0f);
    float p1   = percentile(arr, n, 1.0f);
    float pHalf = percentile(arr, n, 0.5f);

    int ok0    = approx_eq(p0,    0.0f,  EPS_ABS);
    int ok1    = approx_eq(p1,    99.0f, EPS_ABS);
    int okHalf = approx_eq(pHalf, 49.5f, EPS_ABS); /* even n -> midpoint of 49, 50 */

    report("percentile(0..99, p=0)   == 0",    ok0,    0.0f,  p0);
    report("percentile(0..99, p=1)   == 99",   ok1,    99.0f, p1);
    report("percentile(0..99, p=0.5) == 49.5", okHalf, 49.5f, pHalf);
    free(arr);
}

static void test_percentile_odd_median(void)
{
    /* n=5: [10, 20, 30, 40, 50], median = 30 (exact index 2). */
    float arr[5] = {10, 20, 30, 40, 50};
    float p = percentile(arr, 5, 0.5f);
    report("percentile(odd n=5, p=0.5) == 30",
           approx_eq(p, 30.0f, EPS_ABS), 30.0f, p);
}

static void test_percentile_9995_known(void)
{
    /* n=10000 -> p=0.9995 -> rank = 0.9995 * 9999 = 9994.0005
     * -> lo=9994, hi=9995, frac=0.0005
     * -> value = arr[9994] + 0.0005 * (arr[9995] - arr[9994])
     * With arr[i] = i, value = 9994 + 0.0005 * 1 = 9994.0005
     */
    int n = 10000;
    float *arr = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) arr[i] = (float)i;
    float p = percentile(arr, n, 0.9995f);
    float expected = 9994.0f + 0.0005f;
    int ok = approx_eq(p, expected, 1e-3f);
    report("percentile(0..9999, p=0.9995) matches numpy.quantile linear",
           ok, expected, p);
    free(arr);
}

static void test_percentile_matches_scalar_random(void)
{
    int n = 237;
    float *arr = (float *)malloc((size_t)n * sizeof(float));
    unsigned int seed = 31415;
    fill_random(arr, n, &seed);
    /* Sort ascending. */
    qsort(arr, n, sizeof(float), cmp_test_float_asc);

    float pcts[] = {0.0f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99f, 0.9995f, 1.0f};
    int all_ok = 1;
    float last_expected = 0, last_actual = 0;
    for (size_t i = 0; i < sizeof(pcts)/sizeof(pcts[0]); i++) {
        float r   = percentile(arr, n, pcts[i]);
        float ref = percentile_scalar(arr, n, pcts[i]);
        if (!approx_eq(r, ref, 1e-5f)) {
            all_ok = 0;
            last_expected = ref;
            last_actual   = r;
        }
    }
    report("percentile(random, n=237) matches scalar at all pcts",
           all_ok, last_expected, last_actual);
    free(arr);
}

/* ------------------------------------------------------------------ */
/* clip_outliers tests                                                 */
/* ------------------------------------------------------------------ */

static void test_clip_outliers_single_outlier_channel(void)
{
    /* DoD: "clip_outliers clips the outlier channel, doesn't touch
     * normal channels".
     *
     * 3 channels, in_dim = 10000 (>= 2001 so the percentile rank
     * lands below the outlier position; see file-header comment).
     *
     * Channel 0: all values = 1.0, plus one outlier at +1e6 (index 17).
     * Channel 1: all values = 1.0 (no outlier).
     * Channel 2: all values = 1.0, plus one outlier at -1e6 (index 42).
     *
     * For channels 0 and 2:
     *   |W| sorted = [1.0, 1.0, ..., 1.0 (x9999), 1e6]
     *   pct = 0.9995, n = 10000, rank = 9994.0005
     *   lo = 9994, hi = 9995, frac = 0.0005
     *   sorted[9994] = 1.0, sorted[9995] = 1.0 (outlier is at 9999)
     *   value = 1.0 + 0.0005 * (1.0 - 1.0) = 1.0
     *   clip_val = 1.0
     *   outlier (1e6) > 1.0 -> clamped to 1.0 (channel 0) or -1.0 (channel 2)
     *   other 9999 elements (1.0) NOT strictly > 1.0 -> untouched
     *
     * For channel 1:
     *   |W| sorted = all 1.0
     *   percentile = 1.0
     *   no element strictly > 1.0 -> no clipping
     *
     * Returns: 2 (channels 0 and 2 had clipped elements).
     */
    int out_dim = 3, in_dim = 10000;
    float *W = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));

    /* Fill all with baseline 1.0. */
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = 1.0f;
    /* Outliers. */
    W[0 * in_dim + 17] = 1.0e6f;
    W[2 * in_dim + 42] = -1.0e6f;

    int clipped = clip_outliers(W, out_dim, in_dim);

    /* DoD: returns 2 (channels 0 and 2 had outliers). */
    report_int("clip_outliers(1+1 outlier channels) returns 2",
               clipped == 2, 2, clipped);

    /* DoD: channel 1 entirely untouched (all values still 1.0). */
    int ch1_untouched = 1;
    for (int i = 0; i < in_dim; i++) {
        if (W[1 * in_dim + i] != 1.0f) {
            ch1_untouched = 0;
            break;
        }
    }
    report_int("clip_outliers leaves normal channel 1 untouched",
               ch1_untouched, 1, ch1_untouched ? 1 : 0);

    /* DoD: outlier in channel 0 clamped to clip_val = 1.0. */
    float v0 = W[0 * in_dim + 17];
    int ok0 = approx_eq(v0, 1.0f, 1e-6f);
    report("clip_outliers channel 0 outlier clamped to 1.0",
           ok0, 1.0f, v0);

    /* All other elements in channel 0 stay at 1.0. */
    int ch0_others_ok = 1;
    for (int i = 0; i < in_dim; i++) {
        if (i == 17) continue;
        if (W[0 * in_dim + i] != 1.0f) {
            ch0_others_ok = 0;
            break;
        }
    }
    report_int("clip_outliers channel 0 non-outliers stay at 1.0",
               ch0_others_ok, 1, ch0_others_ok ? 1 : 0);

    /* Outlier in channel 2 clamped to -1.0 (sign preserved by copysignf). */
    float v2 = W[2 * in_dim + 42];
    int ok2 = approx_eq(v2, -1.0f, 1e-6f);
    report("clip_outliers channel 2 outlier clamped to -1.0",
           ok2, -1.0f, v2);

    free(W);
}

static void test_clip_outliers_multi_channel(void)
{
    /* 3 channels, in_dim = 10000, all baseline = 1.0.
     * Channel 0 has one outlier (+1e4) at index 5.
     * Channel 1 has one outlier (-1e4) at index 9.
     * Channel 2 has no outlier.
     *
     * Expect: returns 2 (channels 0 and 1 clipped).
     * Channel 2 untouched.
     */
    int out_dim = 3, in_dim = 10000;
    float *W = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = 1.0f;
    W[0 * in_dim + 5] = 1.0e4f;
    W[1 * in_dim + 9] = -1.0e4f;

    int clipped = clip_outliers(W, out_dim, in_dim);
    report_int("clip_outliers(2 of 3 channels with outliers) returns 2",
               clipped == 2, 2, clipped);

    /* Channel 2 (no outlier) untouched. */
    int ch2_untouched = 1;
    for (int i = 0; i < in_dim; i++) {
        if (W[2 * in_dim + i] != 1.0f) { ch2_untouched = 0; break; }
    }
    report_int("clip_outliers multi: channel 2 untouched",
               ch2_untouched, 1, ch2_untouched ? 1 : 0);

    free(W);
}

static void test_clip_outliers_all_equal(void)
{
    /* All-equal channel: |W[o,:]| all = 1.0, so percentile = 1.0,
     * and no element is strictly greater than 1.0 (uses strict >).
     * Returns 0. */
    int out_dim = 2, in_dim = 50;
    float *W = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = 1.0f;

    int clipped = clip_outliers(W, out_dim, in_dim);
    report_int("clip_outliers(all-equal channels) returns 0",
               clipped == 0, 0, clipped);

    /* Verify values are untouched. */
    int untouched = 1;
    for (int i = 0; i < out_dim * in_dim; i++) {
        if (W[i] != 1.0f) { untouched = 0; break; }
    }
    report_int("clip_outliers(all-equal) leaves values at 1.0",
               untouched, 1, untouched ? 1 : 0);

    free(W);
}

static void test_clip_outliers_known_dataset(void)
{
    /* Tight DoD test: 1 channel, 10000 elements, 9999 at 0.5 baseline
     * and 1 at +1e6. The 99.95th percentile lands at:
     *   rank = 0.9995 * 9999 = 9994.0005
     *   lo = 9994, hi = 9995, frac = 0.0005
     *   sorted[9994] = 0.5, sorted[9995] = 0.5 (outlier is at 9999)
     *   value = 0.5 + 0.0005 * (0.5 - 0.5) = 0.5
     *   clip_val = 0.5
     * Outlier (1e6) > 0.5 -> clamped to 0.5.
     * Other 9999 elements (0.5) NOT strictly > 0.5 -> untouched.
     * Returns 1.
     */
    int out_dim = 1, in_dim = 10000;
    float *W = (float *)malloc((size_t)in_dim * sizeof(float));
    for (int i = 0; i < in_dim; i++) W[i] = 0.5f;  /* baseline */
    W[500] = 1.0e6f;  /* outlier */

    int clipped = clip_outliers(W, out_dim, in_dim);
    report_int("clip_outliers(1 outlier in 1 channel) returns 1",
               clipped == 1, 1, clipped);

    /* Verify the outlier is now == clip_val (0.5). */
    int ok = approx_eq(W[500], 0.5f, 1e-6f);
    report("clip_outliers known: outlier clamped to 0.5",
           ok, 0.5f, W[500]);

    /* Verify all other elements are unchanged at 0.5. */
    int others_ok = 1;
    for (int i = 0; i < in_dim; i++) {
        if (i == 500) continue;
        if (W[i] != 0.5f) { others_ok = 0; break; }
    }
    report_int("clip_outliers known: other elements stay 0.5",
               others_ok, 1, others_ok ? 1 : 0);

    free(W);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== cosine_sim tests ===\n");
    test_cosine_sim_identical();
    test_cosine_sim_opposite();
    test_cosine_sim_orthogonal();
    test_cosine_sim_zero_vector();
    test_cosine_sim_random_matches_scalar();
    test_cosine_sim_tail_path();

    printf("\n=== output_cosine tests ===\n");
    test_output_cosine_perfect();
    test_output_cosine_sign_flip();
    test_output_cosine_orthogonal();
    test_output_cosine_matches_scalar();

    printf("\n=== percentile tests ===\n");
    test_percentile_min_max_median();
    test_percentile_odd_median();
    test_percentile_9995_known();
    test_percentile_matches_scalar_random();

    printf("\n=== clip_outliers tests ===\n");
    test_clip_outliers_single_outlier_channel();
    test_clip_outliers_multi_channel();
    test_clip_outliers_all_equal();
    test_clip_outliers_known_dataset();

    printf("\n=== Summary ===\n");
    printf("PASS: %d\n", g_pass_count);
    printf("FAIL: %d\n", g_fail_count);
    printf("Total: %d\n", g_pass_count + g_fail_count);

    return g_fail_count == 0 ? 0 : 1;
}
