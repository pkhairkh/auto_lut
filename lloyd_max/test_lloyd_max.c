/*
 * test_lloyd_max.c - unit tests for the Hessian-weighted Lloyd-Max quantizer.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -std=c11 \
 *       -mavx512f -mavx512dq -mavx512bw -mavx512vl \
 *       lloyd_max.c test_lloyd_max.c -o test_lloyd_max -lm
 *
 * Run:
 *   ./test_lloyd_max
 *
 * Exits 0 on success, non-zero on failure. Each test prints PASS/FAIL.
 */
#include "lloyd_max.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;

static void check_close(const char *label, float got, float expected, float tol)
{
    float d = fabsf(got - expected);
    int ok = (d <= tol);
    printf("  [%s] %s: got=%.6f expected=%.6f (delta=%.6f, tol=%.6f)\n",
           ok ? "PASS" : "FAIL", label, got, expected, d, tol);
    if (!ok) failures++;
}

/* ------------------------------------------------------------------ */
/* Test 1: spec DoD — uniform input [0..99], uniform hessian, palette=4 */
/* Expected centroids: [12, 37, 62, 87] within tolerance 1.0.          */
/* ------------------------------------------------------------------ */
static void test_uniform_dod(void)
{
    const int N = 100;
    const int palette = 4;
    const int max_iter = 20;

    float   *values  = (float *)malloc((size_t)N * sizeof(float));
    float   *hessian = (float *)malloc((size_t)N * sizeof(float));
    float   *lut     = (float *)malloc((size_t)palette * sizeof(float));
    uint8_t *idx     = (uint8_t *)malloc((size_t)N);

    for (int i = 0; i < N; i++) {
        values[i]  = (float)i;
        hessian[i] = 1.0f;
    }

    float delta = hessian_lloyd_max(values, hessian, N, palette,
                                    max_iter, lut, idx);

    printf("test_uniform_dod (N=%d, palette=%d, max_iter=%d)\n",
           N, palette, max_iter);
    printf("  final delta = %.3e\n", delta);

    /* Expected: each cluster is a contiguous block of 25 values, and the
     * weighted mean of an arithmetic progression 0..24 / 25..49 / 50..74 /
     * 75..99 is the midpoint of that range: 12 / 37 / 62 / 87. */
    float expected[4] = {12.0f, 37.0f, 62.0f, 87.0f};
    for (int k = 0; k < palette; k++) {
        char label[32];
        snprintf(label, sizeof(label), "centroid[%d]", k);
        check_close(label, lut[k], expected[k], 1.0f);
    }

    /* Verify convergence (delta should be 0 for this trivial case
     * because the initial quantile partition is already optimal). */
    if (delta >= 1e-6f) {
        printf("  [FAIL] convergence: delta=%.3e >= 1e-6\n", delta);
        failures++;
    } else {
        printf("  [PASS] convergence: delta < 1e-6\n");
    }

    /* Verify assignment monotonicity: since values are sorted, indices
     * must be non-decreasing when read in input order. */
    int mono_ok = 1;
    for (int i = 1; i < N; i++) {
        if (idx[i] < idx[i - 1]) { mono_ok = 0; break; }
    }
    printf("  [%s] assignment monotonicity over sorted input\n",
           mono_ok ? "PASS" : "FAIL");
    if (!mono_ok) failures++;

    /* Verify each cluster covers exactly the expected range. */
    int expected_sizes[4] = {25, 25, 25, 25};
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < N; i++) counts[idx[i]]++;
    for (int k = 0; k < palette; k++) {
        char label[32];
        snprintf(label, sizeof(label), "cluster[%d] size", k);
        if (counts[k] == expected_sizes[k]) {
            printf("  [PASS] %s: %d\n", label, counts[k]);
        } else {
            printf("  [FAIL] %s: got %d, expected %d\n",
                   label, counts[k], expected_sizes[k]);
            failures++;
        }
    }

    free(values); free(hessian); free(lut); free(idx);
}

/* ------------------------------------------------------------------ */
/* Test 2: Hessian-weighted — front half of [0..99] has weight 10x     */
/* the back half. With palette=2, the heavier half should get the     */
/* centroid closer to its own midpoint.                               */
/* ------------------------------------------------------------------ */
static void test_weighted_skew(void)
{
    const int N = 100;
    const int palette = 2;
    const int max_iter = 50;

    float   *values  = (float *)malloc((size_t)N * sizeof(float));
    float   *hessian = (float *)malloc((size_t)N * sizeof(float));
    float   *lut     = (float *)malloc((size_t)palette * sizeof(float));
    uint8_t *idx     = (uint8_t *)malloc((size_t)N);

    for (int i = 0; i < N; i++) {
        values[i]  = (float)i;
        hessian[i] = (i < 50) ? 10.0f : 1.0f;  /* first half 10x heavier */
    }

    float delta = hessian_lloyd_max(values, hessian, N, palette,
                                    max_iter, lut, idx);

    printf("\ntest_weighted_skew (N=%d, palette=%d, max_iter=%d)\n",
           N, palette, max_iter);
    printf("  final delta = %.3e\n", delta);

    /* Total weight = 50*10 + 50*1 = 550. Partition boundary at 275.
     * First cluster: weight 275 -> values 0..27 (27 values with weight 10
     * each = 270, plus value 28 with weight 5 = 275). So first cluster
     * owns indices [0, 28). Centroid = weighted mean of 0..28 = 14. */
    /* Second cluster: indices [28, 100). Weighted mean of 0..99 weighted
     * as 0..27 = 10, 28..49 = 10, 50..99 = 1. */
    /* For simplicity just assert the centroids are increasing and the
     * boundary lies somewhere in the front half (since the front half
     * is 10x heavier, the boundary is much closer to the middle of
     * the front half than to the middle of the whole range). */
    printf("  centroid[0] = %.6f\n", lut[0]);
    printf("  centroid[1] = %.6f\n", lut[1]);

    /* With the front half 10x heavier, the Lloyd-Max optimum pulls the
     * decision boundary INTO the heavy region so cluster 0 is smaller
     * (fewer, but heavier, values) and cluster 1 spans the remainder.
     * Equivalently: the midpoint of the two centroids lies in [0, 50)
     * because the heavier side "deserves" a tighter cluster. */
    float midpoint = 0.5f * (lut[0] + lut[1]);
    int boundary_ok = (midpoint > 0.0f) && (midpoint < 50.0f);
    printf("  [%s] boundary: midpoint=%.4f in heavy region [0, 50)\n",
           boundary_ok ? "PASS" : "FAIL", midpoint);
    if (!boundary_ok) failures++;

    /* Cluster 0 should hold fewer than 50 elements (it's the heavier,
     * tighter cluster). */
    int c0_size = 0;
    for (int i = 0; i < N; i++) if (idx[i] == 0) c0_size++;
    int size_ok = (c0_size > 0) && (c0_size < 50);
    printf("  [%s] cluster 0 size: %d (heavy cluster is tighter, < 50)\n",
           size_ok ? "PASS" : "FAIL", c0_size);
    if (!size_ok) failures++;

    /* Centroids must be ascending. */
    int order_ok = (lut[0] < lut[1]);
    printf("  [%s] centroid ordering: c0 < c1\n", order_ok ? "PASS" : "FAIL");
    if (!order_ok) failures++;

    free(values); free(hessian); free(lut); free(idx);
}

/* ------------------------------------------------------------------ */
/* Test 3: shuffled input — verifies that assignments are correctly    */
/* mapped back to the original (unsorted) order.                       */
/* ------------------------------------------------------------------ */
static void test_shuffled_mapback(void)
{
    const int N = 100;
    const int palette = 4;
    const int max_iter = 20;

    float   *values  = (float *)malloc((size_t)N * sizeof(float));
    float   *hessian = (float *)malloc((size_t)N * sizeof(float));
    float   *lut     = (float *)malloc((size_t)palette * sizeof(float));
    uint8_t *idx     = (uint8_t *)malloc((size_t)N);

    /* Build a deterministic shuffled permutation of [0..99]. */
    for (int i = 0; i < N; i++) values[i] = (float)i;
    /* Fisher-Yates with a fixed LCG seed for reproducibility. */
    unsigned int rng = 12345u;
    for (int i = N - 1; i > 0; i--) {
        rng = rng * 1103515245u + 12345u;
        int j = (int)(rng % (unsigned)(i + 1));
        float t = values[i]; values[i] = values[j]; values[j] = t;
    }
    for (int i = 0; i < N; i++) hessian[i] = 1.0f;

    float delta = hessian_lloyd_max(values, hessian, N, palette,
                                    max_iter, lut, idx);

    printf("\ntest_shuffled_mapback (N=%d, palette=%d)\n", N, palette);
    printf("  final delta = %.3e\n", delta);

    /* Expected centroids are still [12, 37, 62, 87] regardless of input
     * order. */
    float expected[4] = {12.0f, 37.0f, 62.0f, 87.0f};
    for (int k = 0; k < palette; k++) {
        char label[32];
        snprintf(label, sizeof(label), "centroid[%d]", k);
        check_close(label, lut[k], expected[k], 1.0f);
    }

    /* Verify the map-back: each value v should be assigned to the
     * cluster whose centroid is the nearest neighbor (in 1D, this is
     * equivalent to the midpoint partition). */
    int mapback_ok = 1;
    for (int i = 0; i < N; i++) {
        float v = values[i];
        /* Find nearest centroid. */
        int best = 0;
        float best_d = fabsf(v - lut[0]);
        for (int k = 1; k < palette; k++) {
            float d = fabsf(v - lut[k]);
            if (d < best_d) { best_d = d; best = k; }
        }
        if (idx[i] != best) {
            mapback_ok = 0;
            printf("  [FAIL] idx[%d]=%d but nearest centroid is %d (v=%.1f)\n",
                   i, idx[i], best, v);
            break;
        }
    }
    printf("  [%s] map-back: idx_out matches nearest-centroid assignment\n",
           mapback_ok ? "PASS" : "FAIL");
    if (!mapback_ok) failures++;

    free(values); free(hessian); free(lut); free(idx);
}

/* ------------------------------------------------------------------ */
/* Test 4: edge cases — palette=1 (single centroid), N=1, large palette */
/* ------------------------------------------------------------------ */
static void test_edge_cases(void)
{
    printf("\ntest_edge_cases\n");

    /* palette=1: everything in one bucket. */
    {
        float v[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float h[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float lut[1];
        uint8_t idx[5];
        float delta = hessian_lloyd_max(v, h, 5, 1, 10, lut, idx);
        check_close("palette=1 centroid (mean of 1..5)", lut[0], 3.0f, 1e-4f);
        int all_zero = 1;
        for (int i = 0; i < 5; i++) if (idx[i] != 0) all_zero = 0;
        printf("  [%s] palette=1: all idx are 0\n", all_zero ? "PASS" : "FAIL");
        if (!all_zero) failures++;
        (void)delta;
    }

    /* N=1: single element, palette must be 1 (or whatever; only idx 0 valid). */
    {
        float v[1] = {42.0f};
        float h[1] = {1.0f};
        float lut[1];
        uint8_t idx[1];
        float delta = hessian_lloyd_max(v, h, 1, 1, 10, lut, idx);
        check_close("N=1 centroid", lut[0], 42.0f, 1e-6f);
        printf("  [%s] N=1: idx[0]=0 (got %d)\n",
               idx[0] == 0 ? "PASS" : "FAIL", idx[0]);
        if (idx[0] != 0) failures++;
        (void)delta;
    }

    /* palette > N: should still work (some clusters will be empty). */
    {
        float v[3] = {1.0f, 5.0f, 9.0f};
        float h[3] = {1.0f, 1.0f, 1.0f};
        float lut[8];
        uint8_t idx[3];
        float delta = hessian_lloyd_max(v, h, 3, 8, 10, lut, idx);
        /* No specific assertion; just verify it doesn't crash and the
         * idx values are within [0, palette). */
        int in_range = 1;
        for (int i = 0; i < 3; i++) {
            if (idx[i] >= 8) { in_range = 0; break; }
        }
        printf("  [%s] palette>N: all idx in [0, palette)\n",
               in_range ? "PASS" : "FAIL");
        if (!in_range) failures++;
        (void)delta;
    }

    /* NULL inputs. */
    {
        float v[1] = {1.0f}, h[1] = {1.0f}, lut[1];
        uint8_t idx[1];
        float delta1 = hessian_lloyd_max(NULL, h, 1, 1, 1, lut, idx);
        float delta2 = hessian_lloyd_max(v, NULL, 1, 1, 1, lut, idx);
        float delta3 = hessian_lloyd_max(v, h, 0, 1, 1, lut, idx);
        float delta4 = hessian_lloyd_max(v, h, 1, 0, 1, lut, idx);
        int null_ok = (delta1 < 0.0f) && (delta2 < 0.0f) &&
                      (delta3 < 0.0f) && (delta4 < 0.0f);
        printf("  [%s] NULL/invalid input rejection\n",
               null_ok ? "PASS" : "FAIL");
        if (!null_ok) failures++;
    }
}

/* ------------------------------------------------------------------ */
/* Test 5: large-N stress (also exercises SIMD paths).                 */
/* ------------------------------------------------------------------ */
static void test_large_n(void)
{
    const int N = 100000;
    const int palette = 256;
    const int max_iter = 10;

    float   *values  = (float *)malloc((size_t)N * sizeof(float));
    float   *hessian = (float *)malloc((size_t)N * sizeof(float));
    float   *lut     = (float *)malloc((size_t)palette * sizeof(float));
    uint8_t *idx     = (uint8_t *)malloc((size_t)N);

    /* Uniform input [-1, 1). */
    unsigned int rng = 987654321u;
    for (int i = 0; i < N; i++) {
        rng = rng * 1103515245u + 12345u;
        values[i]  = ((float)(rng >> 8) / (float)0x00FFFFFF) * 2.0f - 1.0f;
        hessian[i] = 1.0f;
    }

    float delta = hessian_lloyd_max(values, hessian, N, palette,
                                    max_iter, lut, idx);

    printf("\ntest_large_n (N=%d, palette=%d, max_iter=%d)\n",
           N, palette, max_iter);
    printf("  final delta = %.3e\n", delta);

    /* Centroids should be monotonically non-decreasing. */
    int order_ok = 1;
    for (int k = 1; k < palette; k++) {
        if (lut[k] < lut[k - 1] - 1e-6f) { order_ok = 0; break; }
    }
    printf("  [%s] centroid ordering (256 levels)\n",
           order_ok ? "PASS" : "FAIL");
    if (!order_ok) failures++;

    /* All idx in [0, palette). */
    int in_range = 1;
    for (int i = 0; i < N; i++) {
        if (idx[i] >= palette) { in_range = 0; break; }
    }
    printf("  [%s] all %d indices in [0, %d)\n",
           in_range ? "PASS" : "FAIL", N, palette);
    if (!in_range) failures++;

    /* Centroids should span most of [-1, 1]. */
    int span_ok = (lut[0] < -0.9f) && (lut[palette - 1] > 0.9f);
    printf("  [%s] centroid span: c0=%.4f, c255=%.4f\n",
           span_ok ? "PASS" : "FAIL", lut[0], lut[palette - 1]);
    if (!span_ok) failures++;

    free(values); free(hessian); free(lut); free(idx);
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=== lloyd_max test suite ===\n");
    test_uniform_dod();
    test_weighted_skew();
    test_shuffled_mapback();
    test_edge_cases();
    test_large_n();

    printf("\n=== summary ===\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
}
