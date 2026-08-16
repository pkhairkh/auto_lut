/* kmeans1d.c — 1D k-means via sorting + Lloyd iterations.
 *
 * Uses the fact that in 1D, optimal clusters are contiguous in sorted order.
 * Implements a fast O(n log n + k * n * iters) algorithm:
 *   1. Sort values
 *   2. Initialize centroids at quantiles
 *   3. Lloyd iterations: assign (binary search) + update (weighted mean)
 *
 * This is NOT globally optimal (unlike DP) but converges to a good local
 * optimum in a few iterations. CoreML's kmeans1d library uses the same
 * approach for large inputs.
 */
#include "kmeans1d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { float val; float wt; int orig; } vpt;

static int cmp_vpt(const void *a, const void *b) {
    float va = ((const vpt *)a)->val;
    float vb = ((const vpt *)b)->val;
    return (va > vb) - (va < vb);
}

void kmeans1d_dp(
    const float *values,
    const float *weights,
    int n,
    int k,
    float *centroids,
    uint8_t *assignments
) {
    if (n <= 0 || k <= 0) return;
    if (n <= k) {
        for (int i = 0; i < n; i++) {
            assignments[i] = (uint8_t)i;
            centroids[i] = values[i];
        }
        for (int i = n; i < k; i++) centroids[i] = 0.0f;
        return;
    }

    /* Step 1: Sort */
    vpt *pts = (vpt *)malloc(n * sizeof(vpt));
    for (int i = 0; i < n; i++) {
        pts[i].val = values[i];
        pts[i].wt = weights ? weights[i] : 1.0f;
        pts[i].orig = i;
    }
    qsort(pts, n, sizeof(vpt), cmp_vpt);

    /* Step 2: Initialize centroids at weighted quantiles */
    double total_w = 0;
    for (int i = 0; i < n; i++) total_w += pts[i].wt;
    double cumw = 0;
    int next_q = 0;
    for (int i = 0; i < n && next_q < k; i++) {
        cumw += pts[i].wt;
        while (next_q < k && cumw >= (double)(next_q + 1) * total_w / k) {
            centroids[next_q] = pts[i].val;
            next_q++;
        }
    }
    /* Fill remaining centroids */
    for (int c = next_q; c < k; c++)
        centroids[c] = pts[n - 1].val;

    /* Step 3: Lloyd iterations */
    uint8_t *sorted_assign = (uint8_t *)malloc(n);
    int max_iters = 20;
    for (int iter = 0; iter < max_iters; iter++) {
        /* Assignment: binary search for nearest centroid (centroids are sorted) */
        int changed = 0;
        for (int i = 0; i < n; i++) {
            float v = pts[i].val;
            /* Binary search in sorted centroids */
            int lo = 0, hi = k - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (v > centroids[mid]) lo = mid + 1;
                else hi = mid;
            }
            /* Check if lo-1 is closer */
            if (lo > 0 && fabsf(v - centroids[lo - 1]) < fabsf(v - centroids[lo]))
                lo--;
            if (iter == 0 || sorted_assign[i] != (uint8_t)lo) {
                sorted_assign[i] = (uint8_t)lo;
                changed = 1;
            }
        }
        if (!changed && iter > 0) break;

        /* Update: weighted mean per cluster */
        double sum_w[256], sum_wv[256];
        for (int c = 0; c < k; c++) { sum_w[c] = 0; sum_wv[c] = 0; }
        for (int i = 0; i < n; i++) {
            int c = sorted_assign[i];
            sum_w[c] += pts[i].wt;
            sum_wv[c] += pts[i].wt * pts[i].val;
        }
        for (int c = 0; c < k; c++) {
            if (sum_w[c] > 0)
                centroids[c] = (float)(sum_wv[c] / sum_w[c]);
        }
    }

    /* Map back to original order */
    for (int i = 0; i < n; i++)
        assignments[pts[i].orig] = sorted_assign[i];

    free(pts);
    free(sorted_assign);
}
