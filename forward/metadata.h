#ifndef METADATA_H
#define METADATA_H

#include "safetensors.h"
#include "forward.h"

/* ===========================================================================
 * metadata.h — Writer for metadata_coreml_pg.json
 *
 * Writes a CoreML-palettization metadata file describing every tensor in
 * the model: its name, shape, dtype, byte size, palettizability, and the
 * activation statistics captured by the forward pass.
 *
 * The downstream CoreML palettization tool consumes this JSON to drive
 * per-tensor bit-width selection and LUT allocation.
 * =========================================================================== */

/* Tensor metadata entry. */
typedef struct {
    char    name[512];
    char    dtype[16];
    int     ndim;
    int     shape[8];
    size_t  n_elements;
    size_t  byte_size;        /* element_size * n_elements (FP16 = 2) */
    int     is_2d;             /* 1 if ndim==2 (palettizable candidate) */
    int     is_palettizable;   /* 1 if 2D and not in the skip list */
    int     n_samples;         /* captured activation rows (0 if none) */
    int     in_dim;            /* width of the matmul (weight's input dim) */
    float   hessian_mean;      /* mean of per-channel hessian diagonal */
    float   hessian_max;
} TensorMeta;

/* Write metadata_coreml_pg.json to `path` describing every tensor in `st`.
 *
 *   st            : loaded SafeTensors
 *   ac            : ActivationCapture from forward_run (may be NULL)
 *   output_dir    : base output dir (used to compute relative paths to .bin)
 *   path          : full path to the metadata JSON file to write
 *
 * Returns 0 on success, -1 on failure. */
int metadata_write(const SafeTensors *st, const ActivationCapture *ac,
                    const char *output_dir, const char *path);

#endif /* METADATA_H */
