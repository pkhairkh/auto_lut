/* metadata.c — Writer for metadata_coreml_pg.json
 *
 * Outputs the EXACT format expected by the CoreML palettization pipeline:
 *
 * {
 *   "bitwidth": -1,                          // -1 = per-tensor mixed
 *   "cluster_dim": 1,
 *   "group_axis": 1,
 *   "group_size": 128,                       // default group_size
 *   "nibble_order": "LSB_FIRST",
 *   "indices_layout": "D0D1",
 *   "quantization_type": "palettizer_per_grouped_channel",
 *   "teacher_id": "<model_name>",
 *   "tensors": {
 *     "<tensor_name>": {
 *       "var": "<tensor_name>",
 *       "dense_shape": [out_dim, in_dim],
 *       "indices_shape": [out_dim, in_dim],
 *       "bitwidth": 4,                       // per-tensor: 4, 6, or 8
 *       "groups": in_dim / group_size,
 *       "group_axis": 1,
 *       "group_size": 128,                   // per-tensor group_size
 *       "nibble_order": "LSB_FIRST",
 *       "indices_layout": "D0D1",
 *       "consumer_transpose_y": false,       // true for linear weights (y = x @ W^T)
 *       "index_file": "<sanitized>.idx4",
 *       "lut_file": "<sanitized>.lut_scalar",
 *       "sha256_idx": "...",
 *       "sha256_lut": "...",
 *       "packed_len_bytes": ...,
 *       "idx_payload_offset_used": 0,        // our C tool writes flat files, offset=0
 *       "lut_payload_offset_used": 0
 *     },
 *     ...
 *   }
 * }
 *
 * Only palettized (2D weight) tensors appear in the "tensors" dict.
 */
#include "metadata.h"
#include "fp16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Sanitize a tensor name into a filesystem-safe filename component. */
static void sanitize_name(const char *src, char *dst, size_t cap) {
    size_t i = 0;
    for (; i + 1 < cap && src[i]; i++) {
        char c = src[i];
        if (c == '/' || c == '\\' || c == '.' || c == ' ' || c == ':') c = '_';
        dst[i] = c;
    }
    dst[i] = 0;
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

int metadata_write(
    const SafeTensors *st,
    const ActivationCapture *ac,
    const char *output_dir,
    const char *path,
    const TensorMeta *metas,
    int n_metas
) {
    if (!st || !path) return -1;

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "metadata_write: cannot open %s\n", path);
        return -1;
    }

    /* Determine default group_size (most common among palettized tensors) */
    int default_gs = 128;
    if (n_metas > 0) {
        int gs_counts[3] = {0, 0, 0};  /* 64, 128, 256 */
        for (int i = 0; i < n_metas; i++) {
            if (metas[i].group_size == 64)  gs_counts[0]++;
            else if (metas[i].group_size == 128) gs_counts[1]++;
            else if (metas[i].group_size == 256) gs_counts[2]++;
        }
        int mx = 0;
        if (gs_counts[0] > mx) { mx = gs_counts[0]; default_gs = 64; }
        if (gs_counts[1] > mx) { mx = gs_counts[1]; default_gs = 128; }
        if (gs_counts[2] > mx) { mx = gs_counts[2]; default_gs = 256; }
    }

    /* Top-level fields */
    fprintf(f, "{\n");
    fprintf(f, "  \"bitwidth\": -1,\n");
    fprintf(f, "  \"cluster_dim\": 1,\n");
    fprintf(f, "  \"group_axis\": 1,\n");
    fprintf(f, "  \"group_size\": %d,\n", default_gs);
    fprintf(f, "  \"nibble_order\": \"LSB_FIRST\",\n");
    fprintf(f, "  \"indices_layout\": \"D0D1\",\n");
    fprintf(f, "  \"quantization_type\": \"palettizer_per_grouped_channel\",\n");
    fprintf(f, "  \"teacher_id\": \"dolphin\",\n");
    fprintf(f, "  \"tensors\": {");

    /* Write each palettized tensor as a key-value pair */
    for (int i = 0; i < n_metas; i++) {
        const TensorMeta *m = &metas[i];
        if (i > 0) fputc(',', f);
        fprintf(f, "\n    ");
        write_json_string(f, m->name);
        fprintf(f, ": {\n");
        fprintf(f, "      \"var\": ");
        write_json_string(f, m->name);
        fprintf(f, ",\n");
        fprintf(f, "      \"dense_shape\": [%d, %d],\n", m->out_dim, m->in_dim);
        fprintf(f, "      \"indices_shape\": [%d, %d],\n", m->out_dim, m->in_dim);
        fprintf(f, "      \"bitwidth\": %d,\n", m->bitwidth);
        fprintf(f, "      \"groups\": %d,\n", m->n_groups);
        fprintf(f, "      \"group_axis\": 1,\n");
        fprintf(f, "      \"group_size\": %d,\n", m->group_size);
        fprintf(f, "      \"nibble_order\": \"LSB_FIRST\",\n");
        fprintf(f, "      \"indices_layout\": \"D0D1\",\n");
        fprintf(f, "      \"consumer_transpose_y\": %s,\n",
                m->consumer_transpose_y ? "true" : "false");
        fprintf(f, "      \"index_file\": ");
        write_json_string(f, m->index_file);
        fprintf(f, ",\n");
        fprintf(f, "      \"lut_file\": ");
        write_json_string(f, m->lut_file);
        fprintf(f, ",\n");
        fprintf(f, "      \"sha256_idx\": ");
        write_json_string(f, m->sha256_idx ? m->sha256_idx : "");
        fprintf(f, ",\n");
        fprintf(f, "      \"sha256_lut\": ");
        write_json_string(f, m->sha256_lut ? m->sha256_lut : "");
        fprintf(f, ",\n");
        fprintf(f, "      \"packed_len_bytes\": %zu,\n", m->packed_len_bytes);
        fprintf(f, "      \"idx_payload_offset_used\": 0,\n");
        fprintf(f, "      \"lut_payload_offset_used\": 0\n");
        fprintf(f, "    }");
    }

    fprintf(f, "\n  }\n}\n");
    fclose(f);

    /* Free sha256 strings */
    for (int i = 0; i < n_metas; i++) {
        free((void*)metas[i].sha256_idx);
        free((void*)metas[i].sha256_lut);
    }

    (void)ac;
    (void)output_dir;
    return 0;
}
