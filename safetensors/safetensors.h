#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include "json.h"

/* Maximum tensor name length (incl. NUL). 512 is consistent with the
 * safetensors reference Rust implementation's hard cap for tensor names
 * found in real-world models (e.g. dolphin models). */
#define SAFETENSORS_NAME_MAX 512
/* Maximum number of dimensions supported per tensor. */
#define SAFETENSORS_MAX_NDIM 8
/* Maximum dtype string length (e.g. "F16", "BF16", "F32", "F64", "I8",
 * "I16", "I32", "I64", "U8", "U16", "U32", "U64", "BOOL"). */
#define SAFETENSORS_DTYPE_MAX 16

/* Metadata for a single tensor inside a safetensors file.
 *
 *   name         : NUL-terminated tensor name (copy of the JSON key)
 *   dtype        : NUL-terminated dtype string (copy of "dtype" value)
 *   ndim         : number of valid entries in `shape` (0 for scalars)
 *   shape        : per-axis sizes; entries beyond ndim are 0
 *   data_offset  : logical byte offset within the data section as declared
 *                  by the JSON ("data_offsets": [begin, end])
 *   n_elements   : product of shape[0..ndim-1] (1 for scalars)
 *   byte_offset  : absolute byte offset inside the mapped file
 *                  = file->data_start + data_offset[0]
 *   byte_size    : end - begin from data_offsets
 */
typedef struct {
    char    name[SAFETENSORS_NAME_MAX];
    char    dtype[SAFETENSORS_DTYPE_MAX];
    int     ndim;
    int     shape[SAFETENSORS_MAX_NDIM];
    size_t  data_offset;
    size_t  n_elements;
    size_t  byte_offset;
    size_t  byte_size;
} TensorInfo;

/* A loaded safetensors file. mmap'd once; TensorInfo structs reference
 * data inside `mapped` for zero-copy access. */
typedef struct {
    TensorInfo *tensors;
    int         count;
    char       *mapped;       /* MAP_PRIVATE mmap of whole file */
    size_t      file_size;    /* st_size from fstat() */
    size_t      data_start;   /* 8 + header_length */
} SafeTensors;

/* Load a .safetensors file via mmap(MAP_PRIVATE, PROT_READ).
 * Returns NULL on any error (open, fstat, mmap, header parse, JSON parse).
 * On success the caller owns the result and must release with
 * safetensors_free(). */
SafeTensors  *safetensors_load(const char *path);

/* Return a const pointer into the mmap'd region for the given tensor.
 * Returns NULL if st or ti is NULL, or if the computed offset+size
 * exceeds file_size. */
void         *safetensors_get_ptr(SafeTensors *st, TensorInfo *ti);

/* Linear search by tensor name. Returns NULL if not found. */
TensorInfo   *safetensors_find(SafeTensors *st, const char *name);

/* Release all resources held by `st`. munmap()s the file and frees all
 * owned memory. Safe to call with NULL (no-op). */
void          safetensors_free(SafeTensors *st);

#endif /* SAFETENSORS_H */
