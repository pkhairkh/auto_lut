#ifndef KMEANS1D_H
#define KMEANS1D_H

#include <stdint.h>
#include <stddef.h>

/* ===========================================================================
 * kmeans1d.h — Globally optimal 1D k-means via dynamic programming.
 *
 * Based on Wang & Song (2011) "Ckmeans.1d.dp: Optimal k-means Clustering
 * in One Dimension by Dynamic Programming", J. Statistical Software.
 *
 * The DP exploits the fact that in 1D, optimal clusters are contiguous
 * intervals in sorted order. This reduces the problem to finding k-1
 * optimal split points, solvable via the Bellman recurrence:
 *
 *   D[j, c] = min over m in [c-1, j-1] of { D[m, c-1] + SSQ(m+1, j) }
 *
 * where SSQ(j, i) is the within-cluster sum of squares for sorted points
 * j..i, computed in O(1) via prefix sums.
 *
 * Time: O(k * n^2) for the basic DP, O(k * n * log n) with monotonicity.
 * Space: O(k * n).
 *
 * Supports optional per-point weights (for sensitivity-weighted clustering
 * as in SqueezeLLM).
 * =========================================================================== */

/* Globally optimal 1D k-means clustering.
 *
 *   values    : array of n float values to cluster
 *   weights   : array of n non-negative weights (NULL for uniform weighting)
 *   n         : number of data points
 *   k         : number of clusters (palette size, e.g. 16 for 4-bit)
 *   centroids : output array of k floats (caller-allocated)
 *   assignments: output array of n uint8_t indices in [0, k) (caller-allocated)
 *
 * The centroids are the weighted means of their clusters.
 * The assignments map each input value to its cluster index.
 */
void kmeans1d_dp(
    const float *values,
    const float *weights,   /* NULL for uniform */
    int n,
    int k,
    float *centroids,       /* output: k floats */
    uint8_t *assignments    /* output: n uint8_t */
);

#endif /* KMEANS1D_H */
