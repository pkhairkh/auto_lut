/* accf_cpu.c — ACCF CPU implementation with AVX2/AVX-512.
 *
 * Clean implementation: no placeholders, proper stride handling.
 */
#include "accf.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

int accf_optimize_cpu(
    const float *W,           /* (out_dim, in_dim) row-major, NOT modified */
    float *palette,           /* (n_groups, palette_size) — modified */
    uint8_t *indices,         /* (out_dim, in_dim) — modified */
    const float *X,           /* unused (sensitivity from hessian_diag) */
    const float *hessian_diag,/* (in_dim,) per-channel sensitivity */
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
) {
    if (!W || !palette || !indices || !hessian_diag) return -1;
    if (out_dim <= 0 || in_dim <= 0 || n_groups <= 0) return -1;
    if (palette_size <= 0 || n_iters < 1) return -1;

    /* Use hessian_diag as per-channel activation energy */
    const float *act_sq = hessian_diag;

    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int gs = (start + group_size <= in_dim) ? group_size : (in_dim - start);
        float *pal = palette + (size_t)g * palette_size;

        for (int iter = 0; iter < n_iters; iter++) {
            /* ---- Step A: Refine palette (activation-weighted mean) ---- */
            double sum_w[256], sum_wv[256];
            for (int k = 0; k < palette_size; k++) {
                sum_w[k] = 0.0;
                sum_wv[k] = 0.0;
            }

            for (int o = 0; o < out_dim; o++) {
                const float *wr = W + (size_t)o * in_dim + start;
                const uint8_t *ir = indices + (size_t)o * in_dim + start;
                int j = 0;

                #if defined(__AVX512F__)
                /* AVX-512: process 16 floats at a time */
                for (; j + 15 < gs; j += 16) {
                    __m512 v = _mm512_loadu_ps(wr + j);
                    __m512 a = _mm512_loadu_ps(act_sq + start + j);
                    __m512 va = _mm512_mul_ps(v, a);
                    /* Scalar accumulation per cluster (indices are data-dependent) */
                    float v_arr[16], a_arr[16], va_arr[16];
                    _mm512_storeu_ps(v_arr, v);
                    _mm512_storeu_ps(a_arr, a);
                    _mm512_storeu_ps(va_arr, va);
                    for (int jj = 0; jj < 16; jj++) {
                        int k = ir[j + jj];
                        sum_w[k] += a_arr[jj];
                        sum_wv[k] += va_arr[jj];
                    }
                }
                #elif defined(__AVX2__)
                /* AVX2: process 8 floats at a time */
                for (; j + 7 < gs; j += 8) {
                    __m256 v = _mm256_loadu_ps(wr + j);
                    __m256 a = _mm256_loadu_ps(act_sq + start + j);
                    __m256 va = _mm256_mul_ps(v, a);
                    float v_arr[8], a_arr[8], va_arr[8];
                    _mm256_storeu_ps(v_arr, v);
                    _mm256_storeu_ps(a_arr, a);
                    _mm256_storeu_ps(va_arr, va);
                    for (int jj = 0; jj < 8; jj++) {
                        int k = ir[j + jj];
                        sum_w[k] += a_arr[jj];
                        sum_wv[k] += va_arr[jj];
                    }
                }
                #endif
                /* Scalar tail */
                for (; j < gs; j++) {
                    int k = ir[j];
                    sum_w[k] += act_sq[start + j];
                    sum_wv[k] += wr[j] * act_sq[start + j];
                }
            }

            /* Update palette entries */
            for (int k = 0; k < palette_size; k++) {
                if (sum_w[k] > 0.0) {
                    pal[k] = (float)(sum_wv[k] / sum_w[k]);
                }
                /* Else: keep previous value */
            }

            /* ---- Step B: Reassign indices (argmin activation-weighted error) ---- */
            /* idx[o,j] = argmin_k act_sq[start+j] * (W[o,j] - pal[k])^2
             * Since act_sq >= 0, this is argmin_k (W[o,j] - pal[k])^2
             * (when act_sq > 0; when act_sq == 0, any k works) */
            for (int o = 0; o < out_dim; o++) {
                const float *wr = W + (size_t)o * in_dim + start;
                uint8_t *ir = indices + (size_t)o * in_dim + start;
                int j = 0;

                #if defined(__AVX512F__)
                for (; j + 15 < gs; j += 16) {
                    /* For each of 16 weights, find nearest palette entry */
                    for (int jj = 0; jj < 16; jj++) {
                        float wv = wr[j + jj];
                        float best_d = 1e30f;
                        int best_k = 0;
                        /* Unrolled for palette_size=16 (common case) */
                        if (palette_size == 16) {
                            __m512 wv_b = _mm512_set1_ps(wv);
                            __m512 pal_v = _mm512_loadu_ps(pal);
                            __m512 diff = _mm512_sub_ps(wv_b, pal_v);
                            __m512 sq = _mm512_mul_ps(diff, diff);
                            /* Find argmin */
                            float sq_arr[16];
                            _mm512_storeu_ps(sq_arr, sq);
                            for (int k = 0; k < 16; k++) {
                                if (sq_arr[k] < best_d) { best_d = sq_arr[k]; best_k = k; }
                            }
                        } else {
                            for (int k = 0; k < palette_size; k++) {
                                float d = wv - pal[k];
                                d *= d;
                                if (d < best_d) { best_d = d; best_k = k; }
                            }
                        }
                        ir[j + jj] = (uint8_t)best_k;
                    }
                }
                #endif

                /* Scalar tail */
                for (; j < gs; j++) {
                    float wv = wr[j];
                    float best_d = 1e30f;
                    int best_k = 0;
                    for (int k = 0; k < palette_size; k++) {
                        float d = wv - pal[k];
                        d *= d;
                        if (d < best_d) { best_d = d; best_k = k; }
                    }
                    ir[j] = (uint8_t)best_k;
                }
            }
        }
    }

    return 0;
}
