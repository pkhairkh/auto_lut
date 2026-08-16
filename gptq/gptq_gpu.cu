/* gptq_gpu.cu — CUDA implementation of GPTQ compensation.
 *
 * Uses cuSOLVER for Cholesky decomposition and triangular solves,
 * cuBLAS for the column compensation update.
 *
 * Compiled only when CUDA is available (nvcc).
 */
#include "gptq.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>
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

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t err = (call); \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        return -1; \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* CUDA kernel: quantize one column and compensate remaining            */
/* Each thread handles one row (output channel).                       */
/* ------------------------------------------------------------------ */
__global__ void gptq_column_kernel(
    float *W,                  /* (out_dim, in_dim) device — modified */
    const float *h_col,        /* (in_dim,) device — H^{-1}[:, q] */
    int q,                     /* current column index */
    int out_dim,
    int in_dim,
    float wmin,
    float scale
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;

    float *row = W + (size_t)o * in_dim;
    float w = row[q];
    float wq = roundf((w - wmin) * scale) / scale + wmin;
    float delta = w - wq;

    float h_qq = h_col[q];
    if (fabsf(h_qq) < 1e-12f) h_qq = 1e-12f;
    float coeff = delta / h_qq;

    /* Compensate remaining columns */
    for (int j = q + 1; j < in_dim; j++) {
        row[j] -= coeff * h_col[j];
    }
    row[q] = wq;
}

/* ------------------------------------------------------------------ */
/* Public API: GPU implementation                                      */
/* ------------------------------------------------------------------ */
int gptq_compensate_gpu(
    float *W_host,             /* (out_dim, in_dim) — host, modified in place */
    const float *H_host,       /* (in_dim, in_dim) — host */
    int out_dim,
    int in_dim,
    int bitwidth,
    int group_size
) {
    int palette = 1 << bitwidth;
    size_t W_size = (size_t)out_dim * in_dim * sizeof(float);
    size_t H_size = (size_t)in_dim * in_dim * sizeof(float);

    /* Allocate device memory */
    float *d_W, *d_H, *d_L, *d_h_col;
    int *d_info;
    CHECK_CUDA(cudaMalloc(&d_W, W_size));
    CHECK_CUDA(cudaMalloc(&d_H, H_size));
    CHECK_CUDA(cudaMalloc(&d_L, H_size));
    CHECK_CUDA(cudaMalloc(&d_h_col, in_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    /* Copy to device */
    CHECK_CUDA(cudaMemcpy(d_W, W_host, W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H, H_host, H_size, cudaMemcpyHostToDevice));

    /* Step 1: Add regularization to H on device */
    /* Use a simple kernel or cuBLAS axpy */
    /* For simplicity: add lambda * I on host before copy */
    float *H_reg = (float *)malloc(H_size);
    memcpy(H_reg, H_host, H_size);
    double trace = 0.0;
    for (int i = 0; i < in_dim; i++) trace += H_host[i * in_dim + i];
    float lambda = (float)(0.01 * trace / in_dim);
    if (lambda < 1e-5f) lambda = 1e-5f;
    for (int i = 0; i < in_dim; i++)
        H_reg[i * in_dim + i] += lambda;
    CHECK_CUDA(cudaMemcpy(d_L, H_reg, H_size, cudaMemcpyHostToDevice));
    free(H_reg);

    /* Step 2: Cholesky decomposition via cuSOLVER */
    cusolverDnHandle_t cusolver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

    /* Query workspace size */
    int lwork;
    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(
        cusolver_handle, CUBLAS_FILL_MODE_LOWER, in_dim, d_L, in_dim, &lwork));

    float *d_work;
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));

    /* Cholesky: d_L = L (lower triangular) */
    CHECK_CUSOLVER(cusolverDnSpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_LOWER, in_dim,
        d_L, in_dim, d_work, lwork, d_info));

    /* Check result */
    int info;
    CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        /* Cholesky failed — cleanup and return error for CPU fallback */
        cudaFree(d_W); cudaFree(d_H); cudaFree(d_L); cudaFree(d_h_col);
        cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(cusolver_handle);
        return -1;
    }

    /* Step 3: Column-by-column GPTQ */
    float *h_col_host = (float *)malloc(in_dim * sizeof(float));

    for (int q = 0; q < in_dim; q++) {
        /* Solve H * h_col = e_q via L * L^T * h_col = e_q
         * Step a: L * y = e_q (forward solve)
         * Step b: L^T * h_col = y (backward solve) */

        /* Set up e_q on device (unit vector) */
        memset(h_col_host, 0, in_dim * sizeof(float));
        h_col_host[q] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_h_col, h_col_host, in_dim * sizeof(float), cudaMemcpyHostToDevice));

        /* Forward solve: L * y = e_q */
        CHECK_CUSOLVER(cusolverDnSpotrs(
            cusolver_handle, CUBLAS_FILL_MODE_LOWER, in_dim, 1,
            d_L, in_dim, d_h_col, in_dim, d_info));

        /* Now d_h_col contains H^{-1}[:, q] */

        /* Determine group for uniform grid */
        int g = q / group_size;
        int gs = g * group_size;
        int ge = gs + group_size;
        if (ge > in_dim) ge = in_dim;

        /* Compute scale for this group (on host — min/max over group) */
        CHECK_CUDA(cudaMemcpy(h_col_host, d_W, W_size, cudaMemcpyDeviceToHost));
        float wmin = 1e30f, wmax = -1e30f;
        for (int o = 0; o < out_dim; o++) {
            for (int j = gs; j < ge; j++) {
                float v = h_col_host[(size_t)o * in_dim + j];
                if (v < wmin) wmin = v;
                if (v > wmax) wmax = v;
            }
        }
        if (wmax <= wmin) wmax = wmin + 1e-6f;
        float scale = (float)(palette - 1) / (wmax - wmin);

        /* Launch GPTQ kernel for this column */
        int block_size = 256;
        int grid_size = (out_dim + block_size - 1) / block_size;
        gptq_column_kernel<<<grid_size, block_size>>>(
            d_W, d_h_col, q, out_dim, in_dim, wmin, scale);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Copy W back to host */
    CHECK_CUDA(cudaMemcpy(W_host, d_W, W_size, cudaMemcpyDeviceToHost));

    /* Cleanup */
    free(h_col_host);
    cudaFree(d_W); cudaFree(d_H); cudaFree(d_L); cudaFree(d_h_col);
    cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(cusolver_handle);

    return 0;
}
