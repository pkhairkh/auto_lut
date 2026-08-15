#ifndef METADATA_H
#define METADATA_H

#include "safetensors.h"
#include "forward.h"

/* ===========================================================================
 * metadata.h — Writer for metadata_coreml_pg.json
 *
 * Outputs the CoreML per-grouped-channel palettization manifest in the exact
 * format expected by the downstream CoreML weight extractor.
 * =========================================================================== */

/* Per-tensor palettization result. */
typedef struct {
    char    name[512];          /* original tensor name */
    int     out_dim;            /* dense_shape[0] */
    int     in_dim;             /* dense_shape[1] */
    int     bitwidth;           /* 4, 6, or 8 */
    int     group_size;         /* 64, 128, or 256 */
    int     n_groups;           /* in_dim / group_size */
    int     consumer_transpose_y; /* true for linear weights (y = x @ W^T) */
    char    index_file[600];    /* sanitized name + ".idx4" */
    char    lut_file[600];      /* sanitized name + ".lut_scalar" */
    char   *sha256_idx;         /* hex hash of .idx4 file (owned) */
    char   *sha256_lut;         /* hex hash of .lut_scalar file (owned) */
    size_t  packed_len_bytes;   /* size of packed .idx4 file */
} TensorMeta;

/* Write metadata_coreml_pg.json to `path`.
 *
 *   st            : loaded SafeTensors (for tensor count, unused otherwise)
 *   ac            : ActivationCapture from forward_run (may be NULL)
 *   output_dir    : base output dir (unused, kept for API compat)
 *   path          : full path to the metadata JSON file to write
 *   metas         : array of TensorMeta for each palettized tensor
 *   n_metas       : number of entries in metas
 *
 * Returns 0 on success, -1 on failure. Frees the sha256_idx/sha256_lut
 * strings inside metas (they are owned by this call). */
int metadata_write(
    const SafeTensors *st,
    const ActivationCapture *ac,
    const char *output_dir,
    const char *path,
    const TensorMeta *metas,
    int n_metas
);

#endif /* METADATA_H */
