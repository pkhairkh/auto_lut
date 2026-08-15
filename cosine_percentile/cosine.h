#ifndef COSINE_H
#define COSINE_H

/* cosine.h -- output cosine similarity between reference and quantised
 * weight matmuls.
 *
 * Given an input activation block X (n_samples x in_dim) and two weight
 * matrices W (reference, fp32) and Wq (quantised, dequantised back to
 * fp32) both of shape (out_dim x in_dim), output_cosine computes
 *
 *      Y_ref   = X @ W^T     (n_samples x out_dim)
 *      Y_quant = X @ Wq^T    (n_samples x out_dim)
 *
 * flattened into vectors of length n_samples * out_dim, and returns the
 * standard cosine similarity
 *
 *      cos(Y_ref, Y_quant) = <Y_ref, Y_quant>
 *                             / (||Y_ref|| * ||Y_quant||)
 *
 * A return value of 1.0 indicates the quantised weights reproduce the
 * reference outputs exactly; 0.0 indicates orthogonality; -1.0 indicates
 * sign-flip.
 *
 * Pure C11, no external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c cosine.c -o cosine.o
 *
 * The implementation prefers AVX-512 FMA when available and falls back
 * to AVX2 FMA; both paths are selected at compile time via immintrin.h
 * feature-test macros so there is zero runtime dispatch overhead.
 */

/* Cosine similarity between two flat float arrays of length n.
 *
 *   cos(a, b) = <a, b> / (||a|| * ||b||)
 *
 * Returns 0.0 when either vector has zero norm (degenerate input).
 * Numerically: dot and the two squared norms are accumulated in double
 * inside the SIMD lanes to avoid catastrophic cancellation on long
 * vectors, then a single sqrt is taken per norm at the end.
 */
float cosine_sim(const float *a, const float *b, int n);

/* Output cosine between X@W^T and X@Wq^T.
 *
 *   X        : (n_samples, in_dim), row-major, fp32
 *   W        : (out_dim, in_dim), row-major, fp32 reference weights
 *   Wq       : (out_dim, in_dim), row-major, fp32 dequantised weights
 *   n_samples, out_dim, in_dim : dimensions
 *
 * Computes Y_ref = X@W^T and Y_quant = X@Wq^T simultaneously, accumulating
 *   dot(y_ref, y_quant), ||y_ref||^2, ||y_quant||^2
 * over all (sample, out_channel) pairs in a single fused pass, then
 * returns dot / (sqrt(||y_ref||^2) * sqrt(||y_quant||^2)).
 *
 * Returns 0.0 if either accumulated norm is zero.
 *
 * Parallelism: the (sample, out_channel) loop is OpenMP-parallel; each
 * thread accumulates into private double scalars and the reductions are
 * combined at the end. The inner dot products use AVX-512 (16 lanes) or
 * AVX2 (8 lanes) FMA.
 */
float output_cosine(
    const float *X,     /* (n_samples, in_dim) */
    const float *W,     /* (out_dim, in_dim)   */
    const float *Wq,    /* (out_dim, in_dim)   */
    int n_samples, int out_dim, int in_dim
);

#endif /* COSINE_H */
