/* accf_gpu.cu — CUDA implementation of ACCF.
 *
 * Compiled only when CUDA is available (nvcc).
 * Provides GPU-accelerated palette refinement and index reassignment
 * using cuBLAS for matmuls and custom kernels for argmin.
 */
#include "accf.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* CUDA kernel: refine palette entries (activation-weighted mean)      */
/* One block per palette entry, each block processes its assigned      */
/* (out_dim, group_size) elements.                                     */
/* ------------------------------------------------------------------ */
__global__ void refine_palette_kernel(
    const float *W,           /* (out_dim, in_dim) device */
    const float *act_sq,      /* (in_dim,) device */
    const uint8_t *indices,   /* (out_dim, in_dim) device */
    float *palette,           /* (n_groups, palette_size) device — modified */
    int out_dim,
    int in_dim,
    int group_size,
    int n_groups,
    int palette_size
) {
    int g = blockIdx.x;       /* group index */
    int k = blockIdx.y;       /* palette entry index */
    if (g >= n_groups || k >= palette_size) return;

    int start = g * group_size;
    int gs = (start + group_size <= in_dim) ? group_size : (in_dim - start);

    /* Accumulate weighted sum for palette entry k in group g */
    extern __shared__ double smem[];
    double *sum_w = smem;
    double *sum_wv = smem + 1;

    sum_w[0] = 0.0;
    sum_wv[0] = 0.0;
    __syncthreads();

    for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
        for (int j = 0; j < gs; j++) {
            if (indices[(size_t)o * in_dim + start + j] == k) {
                float w = W[(size_t)o * in_dim + start + j];
                float a = act_sq[start + j];
                atomicAdd(sum_w, (double)a);
                atomicAdd(sum_wv, (double)(w * a));
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (sum_w[0] > 0.0) {
            palette[g * palette_size + k] = (float)(sum_wv[0] / sum_w[0]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* CUDA kernel: reassign indices (argmin over palette entries)         */
/* Each thread handles one (o, j) element.                             */
/* ------------------------------------------------------------------ */
__global__ void reassign_indices_kernel(
    const float *W,           /* (out_dim, in_dim) device */
    const float *palette,     /* (n_groups, palette_size) device */
    uint8_t *indices,         /* (out_dim, in_dim) device — modified */
    int out_dim,
    int in_dim,
    int group_size,
    int n_groups,
    int palette_size
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= out_dim || j >= in_dim) return;

    int g = j / group_size;
    if (g >= n_groups) return;

    const float *pal = palette + g * palette_size;
    float wv = W[(size_t)o * in_dim + j];

    float best_dist = 1e30f;
    int best_k = 0;
    for (int k = 0; k < palette_size; k++) {
        float d = wv - pal[k];
        d *= d;
        if (d < best_dist) {
            best_dist = d;
            best_k = k;
        }
    }
    indices[(size_t)o * in_dim + j] = (uint8_t)best_k;
}

/* ------------------------------------------------------------------ */
/* Public API: GPU implementation                                      */
/* ------------------------------------------------------------------ */
int accf_optimize_gpu(
    const float *W_host,
    float *palette_host,
    uint8_t *indices_host,
    const float *hessian_diag_host,
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
) {
    size_t W_size = (size_t)out_dim * in_dim * sizeof(float);
    size_t pal_size = (size_t)n_groups * palette_size * sizeof(float);
    size_t idx_size = (size_t)out_dim * in_dim * sizeof(uint8_t);
    size_t h_size = in_dim * sizeof(float);

    /* Allocate device memory */
    float *d_W, *d_pal, *d_h;
    uint8_t *d_idx;
    CHECK_CUDA(cudaMalloc(&d_W, W_size));
    CHECK_CUDA(cudaMalloc(&d_pal, pal_size));
    CHECK_CUDA(cudaMalloc(&d_idx, idx_size));
    CHECK_CUDA(cudaMalloc(&d_h, h_size));

    /* Copy to device */
    CHECK_CUDA(cudaMemcpy(d_W, W_host, W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pal, palette_host, pal_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_idx, indices_host, idx_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h, hessian_diag_host, h_size, cudaMemcpyHostToDevice));

    /* Kernel launch config */
    dim3 refine_grid(n_groups, palette_size);
    int refine_block = 256;
    size_t smem_size = 2 * sizeof(double);

    dim3 reassign_grid((out_dim + 15) / 16, (in_dim + 15) / 16);
    dim3 reassign_block(16, 16);

    for (int iter = 0; iter < n_iters; iter++) {
        /* Refine palette */
        refine_palette_kernel<<<refine_grid, refine_block, smem_size>>>(
            d_W, d_h, d_idx, d_pal,
            out_dim, in_dim, group_size, n_groups, palette_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Reassign indices */
        reassign_indices_kernel<<<reassign_grid, reassign_block>>>(
            d_W, d_pal, d_idx,
            out_dim, in_dim, group_size, n_groups, palette_size);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Copy results back */
    CHECK_CUDA(cudaMemcpy(palette_host, d_pal, pal_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(indices_host, d_idx, idx_size, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_W);
    cudaFree(d_pal);
    cudaFree(d_idx);
    cudaFree(d_h);

    return 0;
}
