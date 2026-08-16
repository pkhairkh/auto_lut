/* gptq.c — GPTQ (Optimal Brain Surgeon) weight compensation.
 *
 * Pure C11, no external libs. OpenMP-parallelized.
 *
 * Key optimization: don't compute full H^{-1}. Instead, compute the Cholesky
 * factor L once (O(n³/3)), then for each column q, solve H * x = e_q via
 * forward/backward substitution (O(n²) per column). Total: O(n³/3 + n² * n_cols).
 */
#include "gptq.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Cholesky decomposition: A = L * L^T (in-place, lower triangular) */
static int cholesky_decompose(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum <= 0.0f) return -1;
                A[i * n + j] = sqrtf(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    return 0;
}

/* Solve L * y = b (forward substitution), then L^T * x = y (backward).
 * Gives x = (L * L^T)^{-1} * b = H^{-1} * b.
 * This is the q-th column of H^{-1}. */
static void solve_h_column(const float *L, int n, int q, float *x) {
    /* Forward: L * y = e_q */
    for (int i = 0; i < n; i++) {
        float sum = (i == q) ? 1.0f : 0.0f;
        for (int j = 0; j < i; j++)
            sum -= L[i * n + j] * x[j];
        x[i] = sum / L[i * n + i];
    }
    /* Backward: L^T * x = y (in-place) */
    for (int i = n - 1; i >= 0; i--) {
        float sum = x[i];
        for (int j = i + 1; j < n; j++)
            sum -= L[j * n + i] * x[j];
        x[i] = sum / L[i * n + i];
    }
}

int gptq_compensate_cpu(
    float *W, const float *H, int out_dim, int in_dim,
    int bitwidth, int group_size
) {
    if (!W || !H || in_dim <= 0 || out_dim <= 0)
        return -1;

    int palette = 1 << bitwidth;

    /* Step 1: Cholesky decomposition of H + lambda*I */
    /* Adaptive regularization for rank-deficient Hessians */
    double trace = 0.0;
    for (int i = 0; i < in_dim; i++)
        trace += H[i * in_dim + i];
    float lambda = (float)(0.01 * trace / in_dim);
    if (lambda < 1e-5f) lambda = 1e-5f;

    float *L = (float *)calloc((size_t)in_dim * in_dim, sizeof(float));
    if (!L) return -1;
    memcpy(L, H, (size_t)in_dim * in_dim * sizeof(float));
    for (int i = 0; i < in_dim; i++)
        L[i * in_dim + i] += lambda;

    int cholesky_ok = (cholesky_decompose(L, in_dim) == 0);

    /* If Cholesky failed, use diagonal fallback */
    if (!cholesky_ok) {
        free(L);
        /* Diagonal-only "GPTQ" (no cross-weight compensation) */
        for (int q = 0; q < in_dim; q++) {
            int g = q / group_size;
            int gs = g * group_size;
            int ge = gs + group_size;
            if (ge > in_dim) ge = in_dim;
            float wmin = 1e30f, wmax = -1e30f;
            for (int o = 0; o < out_dim; o++)
                for (int j = gs; j < ge; j++) {
                    float v = W[o * in_dim + j];
                    if (v < wmin) wmin = v;
                    if (v > wmax) wmax = v;
                }
            if (wmax <= wmin) wmax = wmin + 1e-6f;
            float scale = (float)(palette - 1) / (wmax - wmin);
            for (int o = 0; o < out_dim; o++) {
                float w = W[o * in_dim + q];
                W[o * in_dim + q] = roundf((w - wmin) * scale) / scale + wmin;
            }
        }
        return 0;
    }

    /* Step 2: Column-by-column GPTQ using on-the-fly H^{-1} column solve */
    float *h_col = (float *)malloc(in_dim * sizeof(float));
    if (!h_col) { free(L); return -1; }

    for (int q = 0; q < in_dim; q++) {
        /* Solve H * h_col = e_q -> h_col = H^{-1}[:, q] */
        solve_h_column(L, in_dim, q, h_col);

        float h_qq = h_col[q];
        if (fabsf(h_qq) < 1e-12f) h_qq = 1e-12f;

        /* Determine group for uniform grid */
        int g = q / group_size;
        int gs = g * group_size;
        int ge = gs + group_size;
        if (ge > in_dim) ge = in_dim;

        float wmin = 1e30f, wmax = -1e30f;
        for (int o = 0; o < out_dim; o++) {
            for (int j = gs; j < ge; j++) {
                float v = W[o * in_dim + j];
                if (v < wmin) wmin = v;
                if (v > wmax) wmax = v;
            }
        }
        if (wmax <= wmin) wmax = wmin + 1e-6f;
        float scale = (float)(palette - 1) / (wmax - wmin);

        /* Quantize column q and compensate remaining (parallel over rows) */
        #pragma omp parallel for
        for (int o = 0; o < out_dim; o++) {
            float *row = W + (size_t)o * in_dim;
            float w = row[q];
            float wq = roundf((w - wmin) * scale) / scale + wmin;
            float delta = w - wq;
            float coeff = delta / h_qq;
            for (int j = q + 1; j < in_dim; j++) {
                row[j] -= coeff * h_col[j];
            }
            row[q] = wq;
        }
    }

    free(h_col);
    free(L);
    return 0;
}
