#ifndef ACCF_H
#define ACCF_H

#include <stdint.h>
#include <stddef.h>

/* ===========================================================================
 * accf.h — Activation-Conditioned Centroid Fine-tuning with joint
 * index reassignment.
 *
 * Two-stage palettization:
 *   1. Initial clustering: kmeans1d (globally optimal 1D k-means)
 *   2. Iterative refinement:
 *      a. Refine palette: activation-weighted mean per cluster
 *      b. Reassign indices: pick palette entry minimizing activation-weighted error
 *
 * Objective per group g:
 *   min_{pal, idx} sum_{o,j} act_sq[j] * (W[o,j] - pal[idx[o,j]])^2
 *
 * where act_sq[j] = sum_s X[s,j]^2 is the per-channel activation energy.
 *
 * This preserves weight fidelity (high weight cosine) while shifting palette
 * entries toward high-activation channels (good output cosine).
 *
 * Pure C11. AVX2/AVX-512 accelerated. No external dependencies.
 * =========================================================================== */

/* ACCF optimization: refine palette and reassign indices.
 *
 *   W            : (out_dim, in_dim) original weights — NOT modified
 *   palette      : (n_groups, palette_size) — MODIFIED in place
 *   indices      : (out_dim, in_dim) — MODIFIED in place
 *   X            : (n_samples, in_dim) calibration activations, or NULL
 *   hessian_diag : (in_dim,) per-channel sensitivity, or NULL
 *   out_dim, in_dim, n_groups, group_size, palette_size
 *   n_iters      : number of ACCF iterations (typically 10-20)
 *
 * Returns 0 on success, -1 on error.
 */
int accf_optimize(
    const float *W,
    float *palette,             /* (n_groups, palette_size) */
    uint8_t *indices,           /* (out_dim, in_dim) */
    const float *X,             /* (n_samples, in_dim) or NULL */
    const float *hessian_diag,  /* (in_dim,) or NULL */
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
);

#endif /* ACCF_H */
