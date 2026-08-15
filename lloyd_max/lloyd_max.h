/*
 * lloyd_max.h - Hessian-weighted Lloyd-Max 1D quantizer (C11, AVX2/AVX-512).
 *
 * Public API for the auto_lut palettization engine. The quantizer minimizes
 * the *Hessian-weighted* quantization error
 *
 *     J = sum_i  h_i * (v_i - c_{idx_i})^2
 *
 * which corresponds to a second-order (Fisher-diagonal) distortion metric
 * appropriate for preserving the local loss landscape of a neural network
 * when palettizing its weights.
 *
 * The implementation:
 *   1. Sorts values+weights together by value (paired comparator, qsort).
 *   2. Initializes centroids via weighted quantile partitioning using
 *      binary search on the cumulative weight prefix sum.
 *   3. Runs Lloyd-Max iterations:
 *        - Assignment step: midpoint-boundary binary search -> O(N log P)
 *        - Update step:     per-cluster weighted mean accumulated with
 *                           AVX-512 (16 floats) or AVX2 (8 floats) or scalar.
 *        - Convergence:      max|new_levels - old_levels| < 1e-6.
 *   4. Maps assignments back to the original (unsorted) input order.
 *
 * Self-contained: depends only on <stdint.h>, <stddef.h>, and the C11
 * standard library. SIMD intrinsics are pulled in by lloyd_max.c via
 * <immintrin.h> and selected at compile time via -mavx512f / -mavx2 flags.
 *
 * Usage example:
 *
 *     #include "lloyd_max.h"
 *     float   lut[256];
 *     uint8_t idx[N];
 *     float   delta = hessian_lloyd_max(values, hess, N, 256, 20, lut, idx);
 *
 * Return value: the final max centroid delta (>= 0), or -1.0f on bad input.
 */
#ifndef LLOYD_MAX_H
#define LLOYD_MAX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Hessian-weighted Lloyd-Max 1D scalar quantizer.
 *
 *   values   (N,)    input values to quantize (will be sorted internally;
 *                    original order is preserved in idx_out)
 *   hessian  (N,)    per-element weight (Fisher diagonal). Must be >= 0.
 *                    A weight of 0 excludes that element from centroid
 *                    updates but still receives an assignment.
 *   N                number of elements
 *   palette          LUT size (16, 64, or 256). Values > 256 are clamped.
 *   max_iter         iteration cap (20 for final pass, 10 for sweep).
 *   lut_out  (palette,)  output centroids (sorted ascending by construction)
 *   idx_out  (N,)    output assignments in *original* input order, each
 *                    in [0, palette).
 *
 * Returns:
 *   final max|new - old| centroid delta (>= 0) on success,
 *   -1.0f on invalid input (NULL pointers, N <= 0, palette <= 0).
 *
 * Side effects:
 *   Allocates O(N) scratch internally via malloc/free. Thread-safe
 *   (no global state). Safe to call concurrently from multiple OpenMP
 *   threads on disjoint input buffers.
 */
float hessian_lloyd_max(
    const float *values,
    const float *hessian,
    int N,
    int palette,
    int max_iter,
    float *lut_out,
    uint8_t *idx_out
);

#ifdef __cplusplus
}
#endif

#endif /* LLOYD_MAX_H */
