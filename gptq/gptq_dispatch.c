/* gptq_dispatch.c — Runtime dispatch between CPU and GPU GPTQ.
 *
 * If compiled with CUDA (HAVE_CUDA defined), tries GPU first.
 * Falls back to CPU on any GPU error.
 */
#include "gptq.h"
#include <stdio.h>

/* CPU implementation (always available) */
int gptq_compensate_cpu(
    float *W, const float *H, int out_dim, int in_dim,
    int bitwidth, int group_size
);

#ifdef HAVE_CUDA
/* GPU implementation (only if CUDA available) */
int gptq_compensate_gpu(
    float *W, const float *H, int out_dim, int in_dim,
    int bitwidth, int group_size
);
#endif

/* Dispatch: try GPU, fall back to CPU */
int gptq_compensate(
    float *W, const float *H, int out_dim, int in_dim,
    int bitwidth, int group_size
) {
#ifdef HAVE_CUDA
    int rc = gptq_compensate_gpu(W, H, out_dim, in_dim, bitwidth, group_size);
    if (rc == 0) return 0;
    /* GPU failed — fall back to CPU */
    fprintf(stderr, "  [gptq] GPU failed, falling back to CPU\n");
#endif
    return gptq_compensate_cpu(W, H, out_dim, in_dim, bitwidth, group_size);
}
