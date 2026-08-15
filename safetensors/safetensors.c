/* safetensors.c -- mmap'd loader for the .safetensors binary format.
 *
 * File layout (https://huggingface.co/docs/safetensors/index):
 *
 *   offset 0       : uint64_t header_length   (little-endian)
 *   offset 8       : header_length bytes of UTF-8 JSON
 *   offset 8+N     : raw tensor data blob
 *
 * The JSON top-level object maps tensor_name -> { dtype, shape,
 * data_offsets:[begin, end], ... }. An optional "__metadata__" key
 * carries non-tensor metadata and is skipped by this loader.
 *
 * This loader mmap()s the file MAP_PRIVATE|PROT_READ once and produces
 * TensorInfo records that point back into the mapping for zero-copy
 * access.
 */

#include "safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static uint64_t read_u64_le(const unsigned char *p)
{
    return  ((uint64_t)p[0])        | ((uint64_t)p[1] << 8)
          | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
          | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40)
          | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

/* Copy a JSON tensor-name (key) into ti->name with truncation. */
static void set_tensor_name(TensorInfo *ti, const char *key)
{
    if (!key) { ti->name[0] = '\0'; return; }
    size_t n = strlen(key);
    if (n >= sizeof(ti->name)) n = sizeof(ti->name) - 1;
    memcpy(ti->name, key, n);
    ti->name[n] = '\0';
}

/* Copy a JSON dtype string into ti->dtype. */
static void set_tensor_dtype(TensorInfo *ti, const char *s)
{
    if (!s) { ti->dtype[0] = '\0'; return; }
    size_t n = strlen(s);
    if (n >= sizeof(ti->dtype)) n = sizeof(ti->dtype) - 1;
    memcpy(ti->dtype, s, n);
    ti->dtype[n] = '\0';
}

/* Populate one TensorInfo from its JSON value object. Returns 0 on
 * success, -1 on malformed input. */
static int fill_tensor_info(TensorInfo *ti, const JsonValue *v, size_t data_start)
{
    memset(ti, 0, sizeof(*ti));

    /* dtype (string) */
    const JsonValue *dt = json_object_get(v, "dtype");
    char tmp_dt[32];
    if (json_as_string(dt, tmp_dt, sizeof tmp_dt) < 0) return -1;
    set_tensor_dtype(ti, tmp_dt);

    /* shape (array of ints) */
    const JsonValue *sh = json_object_get(v, "shape");
    if (sh && sh->type == JSON_ARRAY) {
        if (sh->v.array.count > SAFETENSORS_MAX_NDIM) return -1;
        ti->ndim = (int)sh->v.array.count;
        size_t n_elem = 1;
        for (size_t i = 0; i < sh->v.array.count; i++) {
            long d = json_as_long(json_array_get(sh, i));
            if (d < 0) return -1;
            ti->shape[i] = (int)d;
            n_elem *= (size_t)d;
        }
        ti->n_elements = (ti->ndim == 0) ? 1 : n_elem;
    } else {
        /* scalar */
        ti->ndim = 0;
        ti->n_elements = 1;
    }

    /* data_offsets: [begin, end] */
    const JsonValue *off = json_object_get(v, "data_offsets");
    if (!off || off->type != JSON_ARRAY || off->v.array.count != 2)
        return -1;
    long begin = json_as_long(json_array_get(off, 0));
    long end   = json_as_long(json_array_get(off, 1));
    if (begin < 0 || end < begin) return -1;
    ti->data_offset = (size_t)begin;
    ti->byte_size   = (size_t)(end - begin);
    ti->byte_offset = data_start + (size_t)begin;
    return 0;
}

/* ------------------------------------------------------------------ */
/* safetensors_load                                                    */
/* ------------------------------------------------------------------ */

SafeTensors *safetensors_load(const char *path)
{
    if (!path) {
        fprintf(stderr, "safetensors_load: NULL path\n");
        return NULL;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "safetensors_load: open(%s) failed: %s\n",
                path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "safetensors_load: fstat(%s) failed: %s\n",
                path, strerror(errno));
        close(fd);
        return NULL;
    }
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "safetensors_load: %s is not a regular file\n", path);
        close(fd);
        return NULL;
    }
    if (st.st_size < 8) {
        fprintf(stderr, "safetensors_load: %s too small (%lld bytes)\n",
                path, (long long)st.st_size);
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    char *mapped = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        mapped = NULL;
        fprintf(stderr, "safetensors_load: mmap(%s) failed: %s\n",
                path, strerror(errno));
        close(fd);
        return NULL;
    }
    /* fd can be closed after mmap; mapping persists until munmap(). */
    close(fd);

    /* Read 8-byte little-endian header length. */
    uint64_t header_len = read_u64_le((const unsigned char *)mapped);

    /* Sanity check: header must fit inside the file, leaving at least
     * a single byte of trailing data (typical files have much more).
     * Header itself can be up to ~100 MB for very large models. */
    if (header_len == 0 || header_len > file_size - 8) {
        fprintf(stderr, "safetensors_load: bad header_length=%llu (file_size=%zu)\n",
                (unsigned long long)header_len, file_size);
        munmap(mapped, file_size);
        return NULL;
    }
    size_t data_start = 8 + (size_t)header_len;
    if (data_start > file_size) {
        fprintf(stderr, "safetensors_load: data_start exceeds file_size\n");
        munmap(mapped, file_size);
        return NULL;
    }

    /* Parse the JSON header. */
    JsonValue *root = json_parse_n(mapped + 8, (size_t)header_len);
    if (!root || root->type != JSON_OBJECT) {
        fprintf(stderr, "safetensors_load: header is not a JSON object\n");
        if (root) json_free(root);
        munmap(mapped, file_size);
        return NULL;
    }

    /* Allocate TensorInfo array. Count = top-level keys minus
     * __metadata__ (if present). */
    size_t n_keys = root->v.object.count;
    int    has_meta = (json_object_get(root, "__metadata__") != NULL);
    int    n_tensors = (int)n_keys - (has_meta ? 1 : 0);
    if (n_tensors < 0) n_tensors = 0;

    TensorInfo *tensors = NULL;
    if (n_tensors > 0) {
        tensors = (TensorInfo *)calloc((size_t)n_tensors, sizeof(*tensors));
        if (!tensors) {
            fprintf(stderr, "safetensors_load: OOM allocating %d tensors\n",
                    n_tensors);
            json_free(root);
            munmap(mapped, file_size);
            return NULL;
        }
    }

    SafeTensors *result = (SafeTensors *)calloc(1, sizeof(*result));
    if (!result) {
        fprintf(stderr, "safetensors_load: OOM allocating SafeTensors\n");
        free(tensors);
        json_free(root);
        munmap(mapped, file_size);
        return NULL;
    }
    result->tensors    = tensors;
    result->count      = 0;
    result->mapped     = mapped;
    result->file_size  = file_size;
    result->data_start = data_start;

    /* Walk top-level keys, skip __metadata__, fill TensorInfo. */
    int idx = 0;
    for (size_t i = 0; i < n_keys; i++) {
        const char *key = root->v.object.entries[i].key;
        const JsonValue *val = root->v.object.entries[i].value;
        if (strcmp(key, "__metadata__") == 0) continue;
        if (val->type != JSON_OBJECT) {
            /* Skip non-object entries (shouldn't happen in valid files). */
            continue;
        }
        if (idx >= n_tensors) {
            /* Shouldn't happen; defensive. */
            break;
        }
        TensorInfo *ti = &tensors[idx];
        if (fill_tensor_info(ti, val, data_start) != 0) {
            fprintf(stderr, "safetensors_load: malformed tensor info for '%s'\n",
                    key);
            safetensors_free(result);
            json_free(root);
            return NULL;
        }
        /* set name AFTER fill_tensor_info, because fill_tensor_info
         * memsets the struct (which would otherwise clobber the name). */
        set_tensor_name(ti, key);
        /* Bounds-check the tensor's bytes against the mapping. */
        if (ti->byte_offset + ti->byte_size > file_size) {
            fprintf(stderr, "safetensors_load: tensor '%s' bytes exceed file_size\n",
                    key);
            safetensors_free(result);
            json_free(root);
            return NULL;
        }
        idx++;
    }
    result->count = idx;

    /* JSON tree is no longer needed; TensorInfo values are copies. */
    json_free(root);

    return result;
}

/* ------------------------------------------------------------------ */
/* accessors                                                           */
/* ------------------------------------------------------------------ */

void *safetensors_get_ptr(SafeTensors *st, TensorInfo *ti)
{
    if (!st || !ti || !st->mapped) return NULL;
    if (ti->byte_size == 0) return NULL;
    if (ti->byte_offset + ti->byte_size > st->file_size) return NULL;
    return (void *)(st->mapped + ti->byte_offset);
}

TensorInfo *safetensors_find(SafeTensors *st, const char *name)
{
    if (!st || !name) return NULL;
    for (int i = 0; i < st->count; i++) {
        if (strcmp(st->tensors[i].name, name) == 0)
            return &st->tensors[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* free                                                                */
/* ------------------------------------------------------------------ */

void safetensors_free(SafeTensors *st)
{
    if (!st) return;
    if (st->mapped && st->file_size > 0) {
        munmap(st->mapped, st->file_size);
    }
    free(st->tensors);
    free(st);
}
