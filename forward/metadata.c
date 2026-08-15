/* metadata.c — Writer for metadata_coreml_pg.json
 *
 * Walks every tensor in a SafeTensors archive, classifies it as palettizable
 * (2D F16 weight) or non-palettizable (1D norm/bias, I64 index tables, etc.),
 * and emits a JSON manifest with shape/dtype/size + captured activation
 * statistics. */
#include "metadata.h"
#include "fp16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Find an Activation by name (linear scan; counts are small). */
static const Activation *find_act(const ActivationCapture *ac, const char *name) {
    if (!ac) return NULL;
    for (int i = 0; i < ac->n_acts; i++) {
        if (strncmp(ac->acts[i].name, name, 512) == 0) return &ac->acts[i];
    }
    return NULL;
}

/* Compute mean and max of the per-channel Hessian diagonal. */
static void hessian_stats(const Activation *a, float *mean, float *max) {
    *mean = 0.0f; *max = 0.0f;
    if (!a || !a->hessian || a->in_dim <= 0) return;
    double sum = 0.0;
    float mx = -1e30f;
    for (int i = 0; i < a->in_dim; i++) {
        float v = a->hessian[i];
        sum += v;
        if (v > mx) mx = v;
    }
    *mean = (float)(sum / a->in_dim);
    *max  = (mx == -1e30f) ? 0.0f : mx;
}

/* Determine if a 2D tensor is palettizable. Excludes:
 *   - relative_position_bias_table (small, special semantics)
 *   - relative_position_index (I64, not a weight)
 *   - embed_positions (would palettize but huge; skip to save time)
 * Actually we palettize ALL 2D F16 tensors — CoreML palettization handles
 * them all. Non-palettizable = anything not 2D F16. */
static int is_palettizable(const TensorInfo *ti) {
    if (ti->ndim != 2) return 0;
    if (strcmp(ti->dtype, "F16") != 0 && strcmp(ti->dtype, "BF16") != 0 &&
        strcmp(ti->dtype, "F32") != 0) return 0;
    /* skip relative_position_index (I64) and other non-weights */
    if (strstr(ti->name, "relative_position_index")) return 0;
    return 1;
}

/* Write a JSON string with proper escaping. */
static void write_json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        switch (*s) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f);  break;
            case '\r': fputs("\\r", f);  break;
            case '\t': fputs("\\t", f);  break;
            default:
                if ((unsigned char)*s < 0x20) {
                    fprintf(f, "\\u%04x", (unsigned char)*s);
                } else {
                    fputc(*s, f);
                }
        }
    }
    fputc('"', f);
}

int metadata_write(const SafeTensors *st, const ActivationCapture *ac,
                    const char *output_dir, const char *path) {
    if (!st || !path) return -1;
    (void)output_dir;

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "metadata_write: cannot open %s\n", path);
        return -1;
    }

    int n_pal = 0, n_nonpal = 0, n_with_act = 0;
    for (int i = 0; i < st->count; i++) {
        if (is_palettizable(&st->tensors[i])) n_pal++;
        else n_nonpal++;
        if (ac && find_act(ac, st->tensors[i].name)) n_with_act++;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"auto_lut_coreml_pg_v1\",\n");
    fprintf(f, "  \"tensor_count\": %d,\n", st->count);
    fprintf(f, "  \"palettizable_count\": %d,\n", n_pal);
    fprintf(f, "  \"non_palettizable_count\": %d,\n", n_nonpal);
    fprintf(f, "  \"captured_count\": %d,\n", n_with_act);
    fprintf(f, "  \"tensors\": [\n");

    for (int i = 0; i < st->count; i++) {
        const TensorInfo *ti = &st->tensors[i];
        int pal = is_palettizable(ti);
        /* element size in bytes */
        int elem_size = 2;  /* F16/BF16 */
        if (strcmp(ti->dtype, "F32") == 0 || strcmp(ti->dtype, "I32") == 0 ||
            strcmp(ti->dtype, "U32") == 0) elem_size = 4;
        else if (strcmp(ti->dtype, "F64") == 0 || strcmp(ti->dtype, "I64") == 0 ||
                 strcmp(ti->dtype, "U64") == 0) elem_size = 8;
        else if (strcmp(ti->dtype, "I8") == 0 || strcmp(ti->dtype, "U8") == 0 ||
                 strcmp(ti->dtype, "BOOL") == 0) elem_size = 1;
        else if (strcmp(ti->dtype, "I16") == 0 || strcmp(ti->dtype, "U16") == 0) elem_size = 2;

        size_t byte_size = ti->n_elements * elem_size;

        const Activation *a = ac ? find_act(ac, ti->name) : NULL;
        float h_mean = 0.0f, h_max = 0.0f;
        int n_samples = 0, in_dim = 0;
        if (a) {
            hessian_stats(a, &h_mean, &h_max);
            n_samples = a->n_samples;
            in_dim = a->in_dim;
        }

        fprintf(f, "    {");
        fprintf(f, "\"name\": ");
        write_json_string(f, ti->name);
        fprintf(f, ", \"dtype\": \"%s\"", ti->dtype);
        fprintf(f, ", \"ndim\": %d", ti->ndim);
        fprintf(f, ", \"shape\": [");
        for (int d = 0; d < ti->ndim; d++) {
            fprintf(f, "%d%s", ti->shape[d], (d+1<ti->ndim) ? ", " : "");
        }
        fprintf(f, "]");
        fprintf(f, ", \"n_elements\": %zu", ti->n_elements);
        fprintf(f, ", \"byte_size\": %zu", byte_size);
        fprintf(f, ", \"is_2d\": %s", ti->ndim == 2 ? "true" : "false");
        fprintf(f, ", \"is_palettizable\": %s", pal ? "true" : "false");
        if (a) {
            fprintf(f, ", \"n_samples\": %d", n_samples);
            fprintf(f, ", \"in_dim\": %d", in_dim);
            fprintf(f, ", \"hessian_mean\": %.6e", h_mean);
            fprintf(f, ", \"hessian_max\": %.6e", h_max);
        }
        fprintf(f, "}%s\n", (i+1 < st->count) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}
