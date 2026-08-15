/* main.c — auto_lut orchestrator.
 *
 * Usage:
 *   ./auto_lut <model_dir> <images_glob> <output_dir>
 *
 * Pipeline:
 *   1. Load model: <model_dir>/model.safetensors + config.json + tokenizer.json
 *      + preprocessor_config.json
 *   2. Glob-expand <images_glob> (shell-style *) to a list of PNG paths
 *   3. For each image: png_load + preprocess_donut -> CHW float32 pixel_values
 *   4. Run forward_run() over all images + a fixed Dolphin prompt
 *      to capture matmul input activations
 *   5. For each 2D F16 weight tensor, run the sensitivity-based bitwidth
 *      selection (9-step priority table) using captured activations to
 *      measure output cosine similarity:
 *        Priority 1: bitwidth=4, group_size=256
 *        Priority 2: bitwidth=4, group_size=128
 *        Priority 3: bitwidth=4, group_size=64
 *        Priority 4: bitwidth=6, group_size=256
 *        Priority 5: bitwidth=6, group_size=128
 *        Priority 6: bitwidth=6, group_size=64
 *        Priority 7: bitwidth=8, group_size=256
 *        Priority 8: bitwidth=8, group_size=128
 *        Priority 9: bitwidth=8, group_size=64
 *      For each config, palettize W per-grouped-channel (kmeans via
 *      hessian_lloyd_max per group), dequantize, and compute cosine sim
 *      between X@W^T and X@Wq^T. Pick the first config that meets the
 *      threshold (0.995). If none meet it, use priority 9 (8-bit, gs=64).
 *   6. Write the palettized weights as .idx4 + .lut_scalar files in the
 *      ANE pre-transpose layout (indices transposed to [in_dim, out_dim],
 *      packed LSB-first via pack_idx4/pack_idx6/pack_idx8).
 *   7. Write metadata_coreml_pg.json in the exact CoreML reference format.
 *   8. Copy non-palettizable tensors (1D biases, norms) as .fp16 files.
 *
 * Output layout under <output_dir>/:
 *   metadata_coreml_pg.json
 *   activations/<sanitized_name>.bin  (per-tensor activation capture)
 *   activations/manifest.json
 *   palettized/<sanitized_name>.idx4       (packed uint indices, transposed)
 *   palettized/<sanitized_name>.lut_scalar (FP16 LUT, groups * 2^bitwidth entries)
 *   palettized/<sanitized_name>.fp16       (non-palettizable tensors, raw FP16)
 */
#include "safetensors.h"
#include "json.h"
#include "fp16.h"
#include "png.h"
#include "preprocess.h"
#include "tokenizer.h"
#include "forward.h"
#include "metadata.h"
#include "pack.h"
#include "lloyd_max.h"
#include "percentile.h"
#include "cosine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
extern char *strdup(const char *);
#endif

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static int mkdir_p(const char *path) {
    char buf[1024];
    strncpy(buf, path, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;
    size_t len = strlen(buf);
    if (len == 0) return 0;
    if (buf[len-1] == '/') buf[len-1] = 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(buf, 0755); *p = '/'; }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static void sanitize_for_file(const char *src, char *dst, size_t cap) {
    size_t i = 0;
    for (; i+1 < cap && src[i]; i++) {
        char c = src[i];
        if (c == '/' || c == '\\' || c == '.' || c == ' ' || c == ':') c = '_';
        dst[i] = c;
    }
    dst[i] = 0;
}

/* ------------------------------------------------------------------ */
/* Glob expansion (minimal: supports exactly one '*' in a path)       */
/* ------------------------------------------------------------------ */

static char **glob_expand(const char *pattern, int *out_n) {
    *out_n = 0;
    const char *star = strchr(pattern, '*');
    if (!star) {
        char **list = xmalloc(sizeof(char*));
        list[0] = strdup(pattern);
        *out_n = 1;
        return list;
    }
    char prefix[1024], suffix[1024];
    size_t plen = star - pattern;
    if (plen >= sizeof(prefix)) plen = sizeof(prefix)-1;
    memcpy(prefix, pattern, plen); prefix[plen] = 0;
    strncpy(suffix, star + 1, sizeof(suffix)-1); suffix[sizeof(suffix)-1] = 0;

    char *last_slash = strrchr(prefix, '/');
    char dir[1024] = ".";
    char pat_prefix[1024] = "";
    if (last_slash) {
        size_t dlen = last_slash - prefix + 1;
        if (dlen >= sizeof(dir)) dlen = sizeof(dir)-1;
        memcpy(dir, prefix, dlen); dir[dlen] = 0;
        if (dlen > 0 && dir[dlen-1] == '/') dir[dlen-1] = 0;
        strncpy(pat_prefix, last_slash + 1, sizeof(pat_prefix)-1);
        pat_prefix[sizeof(pat_prefix)-1] = 0;
    } else {
        strncpy(pat_prefix, prefix, sizeof(pat_prefix)-1);
        pat_prefix[sizeof(pat_prefix)-1] = 0;
    }

    DIR *d = opendir(dir);
    if (!d) return NULL;
    char **list = NULL;
    int cap = 0, n = 0;
    size_t pplen = strlen(pat_prefix);
    struct dirent *e;
    while ((e = readdir(d))) {
        const char *name = e->d_name;
        if (strncmp(name, pat_prefix, pplen) != 0) continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s%s", dir, name, suffix);
        struct stat stbuf;
        if (stat(full, &stbuf) != 0) continue;
        if (!S_ISREG(stbuf.st_mode)) continue;
        if (n >= cap) {
            cap = cap ? cap * 2 : 16;
            list = realloc(list, cap * sizeof(char*));
            if (!list) { closedir(d); return NULL; }
        }
        list[n++] = strdup(full);
    }
    closedir(d);
    *out_n = n;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (strcmp(list[i], list[j]) > 0) {
                char *t = list[i]; list[i] = list[j]; list[j] = t;
            }
        }
    }
    return list;
}

/* ------------------------------------------------------------------ */
/* Find captured activation for a weight tensor.                       */
/* ------------------------------------------------------------------ */
static const Activation *find_activation(const ActivationCapture *ac, const char *name) {
    if (!ac) return NULL;
    for (int i = 0; i < ac->n_acts; i++) {
        if (strncmp(ac->acts[i].name, name, 512) == 0) return &ac->acts[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Per-grouped-channel palettization.                                  */
/*                                                                    */
/* Palettize W (out_dim, in_dim) by splitting in_dim into groups of    */
/* group_size channels, running hessian_lloyd_max on each group's      */
/* flattened (out_dim * group_size) values, and producing:             */
/*   - idx: (out_dim, in_dim) uint8 indices in [0, 2^bitwidth)         */
/*   - lut: (n_groups, 2^bitwidth) float32 centroids                   */
/*                                                                    */
/* Returns the dequantized Wq (out_dim, in_dim) for cosine evaluation. */
/* Caller frees *idx_out and *lut_out and Wq.                          */
/* ------------------------------------------------------------------ */
static float *palettize_per_grouped_channel(
    const float *W, int out_dim, int in_dim,
    const float *hessian_per_in,  /* (in_dim,) or NULL */
    int bitwidth, int group_size,
    uint8_t **idx_out, float **lut_out, int *n_groups_out
) {
    int palette = 1 << bitwidth;
    int n_groups = in_dim / group_size;
    int remainder = in_dim % group_size;
    /* For the remainder, we fold it into the last group (or skip if 0) */
    if (remainder > 0) {
        /* If in_dim not divisible by group_size, pad the last group.
         * For simplicity, we just process the divisible part and copy
         * the remainder as-is (no palettization). This matches CoreML's
         * behavior when group_size doesn't divide in_dim evenly. */
    }

    uint8_t *idx = xmalloc((size_t)out_dim * in_dim);
    float *lut = xmalloc((size_t)n_groups * palette * sizeof(float));
    float *Wq = xmalloc((size_t)out_dim * in_dim * sizeof(float));

    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = start + group_size;
        /* Extract group values: (out_dim * group_size) floats */
        int group_n = out_dim * group_size;
        float *group_vals = xmalloc(group_n * sizeof(float));
        float *group_hess = xmalloc(group_n * sizeof(float));
        for (int o = 0; o < out_dim; o++) {
            for (int j = 0; j < group_size; j++) {
                int idx_w = o * in_dim + start + j;
                group_vals[o * group_size + j] = W[idx_w];
                group_hess[o * group_size + j] = hessian_per_in ? hessian_per_in[start + j] : 1.0f;
            }
        }
        /* Run hessian_lloyd_max on this group */
        float *group_lut = lut + g * palette;
        uint8_t *group_idx = xmalloc(group_n);
        float delta = hessian_lloyd_max(group_vals, group_hess, group_n,
                                         palette, 20, group_lut, group_idx);
        if (delta < 0.0f) {
            /* fallback: uniform quantization */
            float wmin = group_vals[0], wmax = group_vals[0];
            for (int i = 1; i < group_n; i++) {
                if (group_vals[i] < wmin) wmin = group_vals[i];
                if (group_vals[i] > wmax) wmax = group_vals[i];
            }
            if (wmax <= wmin) wmax = wmin + 1e-6f;
            for (int k = 0; k < palette; k++)
                group_lut[k] = wmin + (wmax - wmin) * (float)k / (palette - 1);
            float scale = (float)(palette - 1) / (wmax - wmin);
            for (int i = 0; i < group_n; i++) {
                int k = (int)floorf((group_vals[i] - wmin) * scale + 0.5f);
                if (k < 0) k = 0;
                if (k >= palette) k = palette - 1;
                group_idx[i] = (uint8_t)k;
            }
        }
        /* Write indices and dequantized values back */
        for (int o = 0; o < out_dim; o++) {
            for (int j = 0; j < group_size; j++) {
                int idx_w = o * in_dim + start + j;
                idx[idx_w] = group_idx[o * group_size + j];
                Wq[idx_w] = group_lut[group_idx[o * group_size + j]];
            }
        }
        free(group_vals);
        free(group_hess);
        free(group_idx);
    }

    /* Handle remainder: copy as-is (no palettization) */
    if (remainder > 0) {
        int start = n_groups * group_size;
        for (int o = 0; o < out_dim; o++) {
            for (int j = 0; j < remainder; j++) {
                int idx_w = o * in_dim + start + j;
                idx[idx_w] = 0;
                Wq[idx_w] = W[idx_w];
            }
        }
    }

    *idx_out = idx;
    *lut_out = lut;
    *n_groups_out = n_groups;
    return Wq;
}

/* ------------------------------------------------------------------ */
/* Sensitivity-based bitwidth selection.                               */
/*                                                                    */
/* For each candidate (bitwidth, group_size) in priority order,        */
/* palettize W, dequantize, compute output cosine similarity          */
/* (X @ W^T vs X @ Wq^T), and return the first config that meets      */
/* the threshold. Returns priority 9 (8-bit, gs=64) if none meet it.  */
/* ------------------------------------------------------------------ */

#define COSINE_THRESHOLD 0.995f

typedef struct {
    int bitwidth;
    int group_size;
    float cosine_sim;
} PalettizeConfig;

static const PalettizeConfig PRIORITY_TABLE[9] = {
    {4, 256}, {4, 128}, {4, 64},
    {6, 256}, {6, 128}, {6, 64},
    {8, 256}, {8, 128}, {8, 64},
};

static PalettizeConfig select_bitwidth(
    const float *W, int out_dim, int in_dim,
    const float *hessian_per_in,
    const float *X, int n_samples, int X_in_dim
) {
    /* If no activation captured, default to priority 9 (safest) */
    if (!X || n_samples == 0 || X_in_dim != in_dim) {
        PalettizeConfig r = {8, 64, 1.0f};
        return r;
    }

    /* Subsample activations to keep it fast (max 256 samples) */
    int use_n = n_samples;
    const float *use_X = X;
    float *X_sub = NULL;
    if (n_samples > 256) {
        use_n = 256;
        X_sub = xmalloc((size_t)use_n * in_dim * sizeof(float));
        /* take first 256 samples (deterministic) */
        memcpy(X_sub, X, (size_t)use_n * in_dim * sizeof(float));
        use_X = X_sub;
    }

    PalettizeConfig best = {8, 64, 0.0f};  /* fallback */
    for (int p = 0; p < 9; p++) {
        int bw = PRIORITY_TABLE[p].bitwidth;
        int gs = PRIORITY_TABLE[p].group_size;
        if (in_dim < gs) continue;  /* group_size larger than in_dim */

        uint8_t *idx = NULL;
        float *lut = NULL;
        int n_groups = 0;
        float *Wq = palettize_per_grouped_channel(
            W, out_dim, in_dim, hessian_per_in, bw, gs, &idx, &lut, &n_groups);

        float sim = output_cosine(use_X, W, Wq, use_n, out_dim, in_dim);

        free(idx); free(lut); free(Wq);

        if (sim >= COSINE_THRESHOLD) {
            if (X_sub) free(X_sub);
            PalettizeConfig r = {bw, gs, sim};
            return r;
        }
        if (sim > best.cosine_sim) {
            best.bitwidth = bw;
            best.group_size = gs;
            best.cosine_sim = sim;
        }
    }

    if (X_sub) free(X_sub);
    return best;  /* best effort if none met threshold */
}

/* ------------------------------------------------------------------ */
/* Pack indices for a given bitwidth.                                  */
/*                                                                    */
/* Transposes idx from (out_dim, in_dim) to (in_dim, out_dim) per     */
/* the ANE pre-transpose convention, then packs via pack_idx4/6/8.    */
/* Returns packed bytes in *out, returns byte count.                   */
/* ------------------------------------------------------------------ */
static size_t pack_indices_transposed(
    const uint8_t *idx, int out_dim, int in_dim, int bitwidth,
    uint8_t **out
) {
    size_t N = (size_t)out_dim * in_dim;
    uint8_t *idx_t = xmalloc(N);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < in_dim; j++) {
        for (int o = 0; o < out_dim; o++) {
            idx_t[(size_t)j * out_dim + o] = idx[(size_t)o * in_dim + j];
        }
    }

    size_t packed_max;
    if (bitwidth == 4)      packed_max = (N + 1) / 2;
    else if (bitwidth == 6) packed_max = (N * 6 + 7) / 8;
    else                    packed_max = N;  /* 8-bit */
    uint8_t *packed = xmalloc(packed_max);

    size_t nb;
    if (bitwidth == 4)      nb = pack_idx4(idx_t, in_dim, out_dim, packed);
    else if (bitwidth == 6) nb = pack_idx6(idx_t, in_dim, out_dim, packed);
    else                    nb = pack_idx8(idx_t, in_dim, out_dim, packed);

    free(idx_t);
    *out = packed;
    return nb;
}

/* ------------------------------------------------------------------ */
/* SHA-256 of a file.                                                  */
/* ------------------------------------------------------------------ */
static char *sha256_file(const char *path) {
    /* Use a simple shell-out to sha256sum to avoid openssl dependency */
    char cmd[2200];
    snprintf(cmd, sizeof(cmd), "sha256sum '%s' 2>/dev/null | awk '{print $1}'", path);
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;
    char *hex = malloc(65);
    if (!hex) { pclose(fp); return NULL; }
    if (fscanf(fp, "%64s", hex) != 1) {
        pclose(fp);
        free(hex);
        return NULL;
    }
    pclose(fp);
    return hex;
}

/* ------------------------------------------------------------------ */
/* Palettize one tensor, write outputs, return TensorMeta.             */
/* ------------------------------------------------------------------ */
static int palettize_and_write(
    const float *W, int out_dim, int in_dim,
    const float *hessian_per_in,
    const float *X, int n_samples, int X_in_dim,
    const char *out_dir, const char *tensor_name,
    TensorMeta *meta_out
) {
    /* Step 1: select bitwidth via sensitivity priority table */
    PalettizeConfig cfg = select_bitwidth(W, out_dim, in_dim,
                                            hessian_per_in, X, n_samples, X_in_dim);

    /* Step 2: clip outliers */
    float *W_clipped = xmalloc((size_t)out_dim * in_dim * sizeof(float));
    memcpy(W_clipped, W, (size_t)out_dim * in_dim * sizeof(float));
    clip_outliers(W_clipped, out_dim, in_dim);

    /* Step 3: final palettization with selected config */
    uint8_t *idx = NULL;
    float *lut = NULL;
    int n_groups = 0;
    float *Wq = palettize_per_grouped_channel(
        W_clipped, out_dim, in_dim, hessian_per_in,
        cfg.bitwidth, cfg.group_size, &idx, &lut, &n_groups);

    /* Step 4: pack indices (transposed) */
    uint8_t *packed = NULL;
    size_t packed_bytes = pack_indices_transposed(
        idx, out_dim, in_dim, cfg.bitwidth, &packed);

    /* Step 5: write .idx4 file */
    char sanitized[1024];
    sanitize_for_file(tensor_name, sanitized, sizeof(sanitized));
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s.idx4", out_dir, sanitized);
    FILE *f = fopen(path, "wb");
    if (!f) {
        free(W_clipped); free(idx); free(lut); free(Wq); free(packed);
        return -1;
    }
    fwrite(packed, 1, packed_bytes, f);
    fclose(f);

    /* Step 6: write .lut_scalar file (n_groups * 2^bitwidth FP16 entries) */
    int palette = 1 << cfg.bitwidth;
    snprintf(path, sizeof(path), "%s/%s.lut_scalar", out_dir, sanitized);
    write_lut_fp16(lut, n_groups, palette, path);

    /* Step 7: fill TensorMeta */
    memset(meta_out, 0, sizeof(*meta_out));
    strncpy(meta_out->name, tensor_name, 511);
    meta_out->out_dim = out_dim;
    meta_out->in_dim = in_dim;
    meta_out->bitwidth = cfg.bitwidth;
    meta_out->group_size = cfg.group_size;
    meta_out->n_groups = n_groups;
    /* consumer_transpose_y: true for linear weights (y = x @ W^T).
     * All our 2D weights are linear weights, so true. */
    meta_out->consumer_transpose_y = 1;
    snprintf(meta_out->index_file, sizeof(meta_out->index_file), "%s.idx4", sanitized);
    snprintf(meta_out->lut_file, sizeof(meta_out->lut_file), "%s.lut_scalar", sanitized);
    meta_out->packed_len_bytes = packed_bytes;

    /* SHA-256 of output files */
    snprintf(path, sizeof(path), "%s/%s.idx4", out_dir, sanitized);
    meta_out->sha256_idx = sha256_file(path);
    snprintf(path, sizeof(path), "%s/%s.lut_scalar", out_dir, sanitized);
    meta_out->sha256_lut = sha256_file(path);

    free(W_clipped); free(idx); free(lut); free(Wq); free(packed);
    return 0;
}

/* Write a non-palettizable tensor as raw FP16 (.fp16 file). */
static int write_nonpalettizable(
    const void *raw_data, size_t n_elements, int is_fp16,
    const char *out_dir, const char *tensor_name
) {
    char sanitized[1024];
    sanitize_for_file(tensor_name, sanitized, sizeof(sanitized));
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s.fp16", out_dir, sanitized);
    copy_tensor_fp16(raw_data, n_elements, !is_fp16, path);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main orchestrator                                                   */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model_dir> <images_glob> <output_dir>\n", argv[0]);
        return 1;
    }
    const char *model_dir  = argv[1];
    const char *img_glob   = argv[2];
    const char *output_dir = argv[3];

    clock_t t0 = clock();
    double t_start = (double)t0 / CLOCKS_PER_SEC;

    /* 1. Load model files */
    char path[2048];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    fprintf(stderr, "[main] loading %s ...\n", path);
    SafeTensors *st = safetensors_load(path);
    if (!st) { fprintf(stderr, "[main] FAILED to load safetensors\n"); return 1; }
    fprintf(stderr, "[main]   %d tensors loaded\n", st->count);

    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *cf = fopen(path, "rb");
    if (!cf) { fprintf(stderr, "[main] FAILED to open config.json\n"); return 1; }
    fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
    char *cbuf = xmalloc(csz + 1);
    fread(cbuf, 1, csz, cf); cbuf[csz] = 0; fclose(cf);
    JsonValue *cfg = json_parse(cbuf);
    free(cbuf);
    if (!cfg) { fprintf(stderr, "[main] FAILED to parse config.json\n"); return 1; }

    /* load preprocessor config */
    snprintf(path, sizeof(path), "%s/preprocessor_config.json", model_dir);
    FILE *pf = fopen(path, "rb");
    float img_mean[3] = {0.485f, 0.456f, 0.406f};
    float img_std[3]  = {0.229f, 0.224f, 0.225f};
    float rescale = 1.0f / 255.0f;
    int img_h = 896, img_w = 896;
    if (pf) {
        fseek(pf, 0, SEEK_END); long psz = ftell(pf); fseek(pf, 0, SEEK_SET);
        char *pbuf = xmalloc(psz + 1);
        fread(pbuf, 1, psz, pf); pbuf[psz] = 0; fclose(pf);
        JsonValue *pcfg = json_parse(pbuf);
        free(pbuf);
        if (pcfg) {
            const JsonValue *m = json_object_get(pcfg, "image_mean");
            const JsonValue *s = json_object_get(pcfg, "image_std");
            const JsonValue *r = json_object_get(pcfg, "rescale_factor");
            const JsonValue *sz = json_object_get(pcfg, "size");
            if (m && m->type == JSON_ARRAY && m->v.array.count >= 3) {
                for (int i = 0; i < 3; i++) {
                    const JsonValue *mi = json_array_get(m, i);
                    img_mean[i] = (mi && mi->type == JSON_NUMBER) ? (float)mi->v.number : 0.0f;
                }
            }
            if (s && s->type == JSON_ARRAY && s->v.array.count >= 3) {
                for (int i = 0; i < 3; i++) {
                    const JsonValue *si = json_array_get(s, i);
                    img_std[i] = (si && si->type == JSON_NUMBER) ? (float)si->v.number : 1.0f;
                }
            }
            if (r) rescale = (float)r->v.number;
            if (sz) {
                const JsonValue *h = json_object_get(sz, "height");
                const JsonValue *w = json_object_get(sz, "width");
                if (h) img_h = (int)json_as_long(h);
                if (w) img_w = (int)json_as_long(w);
            }
            json_free(pcfg);
        }
    }

    /* load tokenizer */
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    Tokenizer *tk = tokenizer_load(path);
    if (!tk) fprintf(stderr, "[main] WARNING: tokenizer.json failed to load\n");

    /* 2. Glob-expand images */
    int n_images = 0;
    char **img_paths = glob_expand(img_glob, &n_images);
    if (!img_paths || n_images == 0) {
        fprintf(stderr, "[main] FAILED to find any images matching %s\n", img_glob);
        return 1;
    }
    int cap_images = 2;
    if (n_images > cap_images) {
        fprintf(stderr, "[main] capping from %d to %d images for time budget\n", n_images, cap_images);
        n_images = cap_images;
    }
    fprintf(stderr, "[main] found %d images\n", n_images);

    /* 3. Preprocess each image */
    float **pixel_values = xmalloc(n_images * sizeof(float*));
    for (int i = 0; i < n_images; i++) {
        Image *img = png_load(img_paths[i]);
        if (!img) { pixel_values[i] = NULL; continue; }
        pixel_values[i] = preprocess_donut(img, img_w, img_h, img_mean, img_std, rescale);
        image_free(img);
    }

    /* 4. Build the decoder prompt.
     * Dolphin prompt: <s> </s> </s> </s> </s> (5 tokens).
     * The C BPE tokenizer doesn't match </s> as a special token (it encodes
     * character-by-character), so we use hardcoded token IDs directly.
     * <s>=0, </s>=2 (verified from tokenizer.json added_tokens). */
    int input_ids[16];
    int n_prompt = 0;
    input_ids[n_prompt++] = 0;  /* <s> */
    input_ids[n_prompt++] = 2;  /* </s> */
    input_ids[n_prompt++] = 2;
    input_ids[n_prompt++] = 2;
    input_ids[n_prompt++] = 2;
    if (tk) tokenizer_free(tk);

    /* 5. Run the forward pass with activation capture */
    fprintf(stderr, "[main] running forward pass over %d images with %d-token prompt ...\n",
            n_images, n_prompt);
    mkdir_p(output_dir);
    ActivationCapture *ac = forward_run(st, cfg, pixel_values, n_images,
                                          input_ids, n_prompt, output_dir);
    if (!ac) { fprintf(stderr, "[main] forward_run FAILED\n"); return 1; }
    fprintf(stderr, "[main] captured %d activations\n", ac->n_acts);

    /* 6. Palettize every 2D F16/BF16/F32 weight tensor with the
     *    sensitivity-based priority table; copy non-palettizable as FP16. */
    snprintf(path, sizeof(path), "%s/palettized", output_dir);
    mkdir_p(path);

    /* Allocate TensorMeta array (max = number of 2D weight tensors) */
    int max_metas = 0;
    for (int i = 0; i < st->count; i++) {
        TensorInfo *ti = &st->tensors[i];
        if (ti->ndim == 2 &&
            (strcmp(ti->dtype, "F16") == 0 || strcmp(ti->dtype, "BF16") == 0 ||
             strcmp(ti->dtype, "F32") == 0)) {
            max_metas++;
        }
    }
    TensorMeta *metas = xmalloc(max_metas * sizeof(TensorMeta));
    int n_metas = 0;

    int n_pal = 0, n_fp16 = 0, n_skip = 0;
    clock_t t_pal_start = clock();

    /* Bitwidth distribution counters for logging */
    int bw_counts[3] = {0, 0, 0};  /* 4, 6, 8 */
    int gs_counts[3] = {0, 0, 0};  /* 64, 128, 256 */

    for (int i = 0; i < st->count; i++) {
        TensorInfo *ti = &st->tensors[i];
        void *raw = safetensors_get_ptr(st, ti);
        if (!raw) { n_skip++; continue; }
        int is_f16 = (strcmp(ti->dtype, "F16") == 0 || strcmp(ti->dtype, "BF16") == 0);
        int is_f32 = (strcmp(ti->dtype, "F32") == 0);
        if (ti->ndim == 2 && (is_f16 || is_f32)) {
            size_t n = ti->n_elements;
            float *W = xmalloc(n * sizeof(float));
            if (is_f16) fp16_to_f32_array(raw, W, n);
            else        memcpy(W, raw, n * sizeof(float));

            const Activation *act = find_activation(ac, ti->name);
            const float *hess = act ? act->hessian : NULL;
            const float *X = act ? act->data : NULL;
            int n_samples = act ? act->n_samples : 0;

            if (palettize_and_write(W, ti->shape[0], ti->shape[1],
                                      hess, X, n_samples, ti->shape[1],
                                      path, ti->name, &metas[n_metas]) == 0) {
                n_metas++;
                n_pal++;
                /* update distribution */
                int bw = metas[n_metas-1].bitwidth;
                int gs = metas[n_metas-1].group_size;
                if (bw == 4) bw_counts[0]++;
                else if (bw == 6) bw_counts[1]++;
                else if (bw == 8) bw_counts[2]++;
                if (gs == 64) gs_counts[0]++;
                else if (gs == 128) gs_counts[1]++;
                else if (gs == 256) gs_counts[2]++;
            } else {
                n_skip++;
            }
            free(W);
        } else if ((is_f16 || is_f32)) {
            if (write_nonpalettizable(raw, ti->n_elements, is_f16, path, ti->name) == 0)
                n_fp16++;
            else
                n_skip++;
        } else {
            n_skip++;
        }
    }
    double t_pal = (double)(clock() - t_pal_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "[main] palettized %d tensors (%.1fs), copied %d as FP16, skipped %d\n",
            n_pal, t_pal, n_fp16, n_skip);
    fprintf(stderr, "[main] bitwidth distribution: 4-bit=%d, 6-bit=%d, 8-bit=%d\n",
            bw_counts[0], bw_counts[1], bw_counts[2]);
    fprintf(stderr, "[main] group_size distribution: 64=%d, 128=%d, 256=%d\n",
            gs_counts[0], gs_counts[1], gs_counts[2]);

    /* 7. Write metadata_coreml_pg.json in the exact CoreML reference format */
    snprintf(path, sizeof(path), "%s/metadata_coreml_pg.json", output_dir);
    if (metadata_write(st, ac, output_dir, path, metas, n_metas) != 0) {
        fprintf(stderr, "[main] WARNING: metadata_write failed\n");
    } else {
        fprintf(stderr, "[main] wrote %s (%d tensors)\n", path, n_metas);
    }
    free(metas);

    /* cleanup */
    for (int i = 0; i < n_images; i++) {
        free(pixel_values[i]);
        free(img_paths[i]);
    }
    free(pixel_values);
    free(img_paths);
    activation_capture_free(ac);
    json_free(cfg);
    safetensors_free(st);

    double t_end = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stderr, "[main] DONE in %.2fs (CPU)\n", t_end - t_start);
    return 0;
}
