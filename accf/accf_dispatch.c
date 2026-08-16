/* accf_dispatch.c — Runtime dispatch between CPU and GPU ACCF.
 */
#include "accf.h"
#include <stdio.h>

/* CPU implementation */
int accf_optimize_cpu(
    const float *W,
    float *palette,
    uint8_t *indices,
    const float *X,
    const float *hessian_diag,
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
);

#ifdef HAVE_CUDA
int accf_optimize_gpu(
    const float *W,
    float *palette,
    uint8_t *indices,
    const float *hessian_diag,
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
);
#endif

int accf_optimize(
    const float *W,
    float *palette,
    uint8_t *indices,
    const float *X,
    const float *hessian_diag,
    int out_dim,
    int in_dim,
    int n_groups,
    int group_size,
    int palette_size,
    int n_iters
) {
#ifdef HAVE_CUDA
    int rc = accf_optimize_gpu(W, palette, indices, hessian_diag,
                                out_dim, in_dim, n_groups,
                                group_size, palette_size, n_iters);
    if (rc == 0) return 0;
    fprintf(stderr, "  [accf] GPU failed, falling back to CPU\n");
#endif
    return accf_optimize_cpu(W, palette, indices, X, hessian_diag,
                             out_dim, in_dim, n_groups,
                             group_size, palette_size, n_iters);
}
