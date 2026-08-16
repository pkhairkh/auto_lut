#ifndef GPTQ_H
#define GPTQ_H

#include <stdint.h>

/* ===========================================================================
 * gptq.h — GPTQ (Optimal Brain Surgeon) weight compensation.
 *
 * Implements the OBS formula (Hassibi & Stork 1993, adapted by Frantar et al.
 * 2022 as GPTQ) for post-training weight quantization:
 *
 *   For each column q (left to right):
 *     1. Quantize w_q to uniform grid: q_hat = round((w_q - wmin) * scale) / scale + wmin
 *     2. Error: delta = w_q - q_hat
 *     3. Compensate remaining columns:
 *        W[:, F] -= delta / [H_inv]_qq * (H_inv)[:, q]
 *     4. Update H_inv via Gaussian elimination:
 *        H_inv -= (H_inv[:, q] * H_inv[q, :]) / [H_inv]_qq
 *
 * After compensation, the weight matrix W is palettized with kmeans1d DP
 * to produce the final LUT + indices.
 *
 * Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization
 * for Generative Pre-trained Transformers", ICLR 2023.
 * =========================================================================== */

/* GPTQ compensation: adjusts weight matrix W to minimize output error
 * when quantized.
 *
 *   W          : (out_dim, in_dim) weight matrix, row-major (MODIFIED IN PLACE)
 *   H          : (in_dim, in_dim) Hessian matrix H = 2 * X^T X
 *   out_dim    : number of output channels (rows of W)
 *   in_dim     : number of input channels (columns of W)
 *   bitwidth   : target bitwidth (4, 6, or 8)
 *   group_size : channels per group for uniform grid computation
 *
 * Returns 0 on success, -1 on error (e.g. Cholesky failure).
 * On success, W contains the GPTQ-compensated weights ready for palettization.
 * On failure, W is left in an intermediate state — caller should fall back
 * to diagonal-only quantization (no GPTQ).
 */
int gptq_compensate(
    float *W,              /* (out_dim, in_dim) — modified in place */
    const float *H,        /* (in_dim, in_dim) Hessian */
    int out_dim,
    int in_dim,
    int bitwidth,
    int group_size
);

#endif /* GPTQ_H */

#ifdef __cplusplus
extern "C" {
#endif
int gptq_compensate_cpu(float *W, const float *H, int out_dim, int in_dim, int bitwidth, int group_size);
int gptq_compensate_gpu(float *W, const float *H, int out_dim, int in_dim, int bitwidth, int group_size);
#ifdef __cplusplus
}
#endif
/* CPU and GPU function declarations for dispatch */
#ifdef __cplusplus
extern "C" {
#endif
int gptq_compensate_cpu(float *W, const float *H, int out_dim, int in_dim, int bitwidth, int group_size);
int gptq_compensate_gpu(float *W, const float *H, int out_dim, int in_dim, int bitwidth, int group_size);
#ifdef __cplusplus
}
#endif
