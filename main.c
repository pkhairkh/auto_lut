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
 *      For each config, palettize W per-grouped-channel (kmeans1d DP after
 *      GPTQ compensation, then kmeans1d DP palettization per group)
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
#include "gptq.h"
#include "kmeans1d.h"
#include "percentile.h"
#include "cosine.h"
#include "accf.h"

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
/* GPTQ + kmeans1d palettization.                                     */
/* ------------------------------------------------------------------ */

static float *palettize_gptq(
    float *W, int out_dim, int in_dim,
    const float *hessian_matrix,
    const float *hessian_diag,
    int bitwidth, int group_size,
    uint8_t **idx_out, float **lut_out, int *n_groups_out
) {
    int palette = 1 << bitwidth;
    int n_groups = in_dim / group_size;
    if (hessian_matrix) {
        int n_elem = in_dim * in_dim;
        float *H_scaled = xmalloc(n_elem * sizeof(float));
        for (int i = 0; i < n_elem; i++) H_scaled[i] = 2.0f * hessian_matrix[i];
        gptq_compensate(W, H_scaled, out_dim, in_dim, bitwidth, group_size);
        free(H_scaled);
    }
    uint8_t *idx = xmalloc((size_t)out_dim * in_dim);
    float *lut = xmalloc((size_t)n_groups * palette * sizeof(float));
    float *Wq = xmalloc((size_t)out_dim * in_dim * sizeof(float));
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int gn = out_dim * group_size;
        float *gv = xmalloc(gn * sizeof(float));
        float *gh = NULL;
        if (hessian_diag) {
            gh = xmalloc(gn * sizeof(float));
            for (int o = 0; o < out_dim; o++)
                for (int j = 0; j < group_size; j++)
                    gh[o * group_size + j] = hessian_diag[start + j];
        }
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < group_size; j++)
                gv[o * group_size + j] = W[o * in_dim + start + j];
        float *gl = lut + g * palette;
        uint8_t *gi = xmalloc(gn);
        kmeans1d_dp(gv, gh, gn, palette, gl, gi);
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < group_size; j++) {
                int w = o * in_dim + start + j;
                idx[w] = gi[o * group_size + j];
                Wq[w] = gl[gi[o * group_size + j]];
            }
        free(gv); free(gh); free(gi);
    }
    int rem = in_dim % group_size;
    if (rem > 0) {
        int start = n_groups * group_size;
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < rem; j++) {
                idx[o * in_dim + start + j] = 0;
                Wq[o * in_dim + start + j] = W[o * in_dim + start + j];
            }
    }
    *idx_out = idx; *lut_out = lut; *n_groups_out = n_groups;
    return Wq;
}


/* ------------------------------------------------------------------ */
/* Simplified bitwidth selection with GPTQ (3-step).                   */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* ACCF palettization: kmeans1d initial + ACCF refinement.            */
/* ------------------------------------------------------------------ */

static float *palettize_accf(
    const float *W_orig, int out_dim, int in_dim,
    const float *hessian_diag,
    int bitwidth, int group_size,
    uint8_t **idx_out, float **lut_out, int *n_groups_out
) {
    int palette = 1 << bitwidth;
    int n_groups = in_dim / group_size;

    uint8_t *idx = xmalloc((size_t)out_dim * in_dim);
    float *lut = xmalloc((size_t)n_groups * palette * sizeof(float));
    float *Wq = xmalloc((size_t)out_dim * in_dim * sizeof(float));

    /* Step 1: Initial kmeans1d clustering */
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int gn = out_dim * group_size;
        float *gv = xmalloc(gn * sizeof(float));
        float *gh = NULL;
        if (hessian_diag) {
            gh = xmalloc(gn * sizeof(float));
            for (int o = 0; o < out_dim; o++)
                for (int j = 0; j < group_size; j++)
                    gh[o * group_size + j] = hessian_diag[start + j];
        }
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < group_size; j++)
                gv[o * group_size + j] = W_orig[o * in_dim + start + j];
        float *gl = lut + g * palette;
        uint8_t *gi = xmalloc(gn);
        kmeans1d_dp(gv, gh, gn, palette, gl, gi);
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < group_size; j++) {
                int w = o * in_dim + start + j;
                idx[w] = gi[o * group_size + j];
                Wq[w] = gl[gi[o * group_size + j]];
            }
        free(gv); free(gh); free(gi);
    }

    /* Step 2: ACCF refinement */
    accf_optimize(W_orig, lut, idx, NULL, hessian_diag,
                  out_dim, in_dim, n_groups, group_size, palette, 10);

    /* Step 3: Reconstruct Wq from refined palette + indices */
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < group_size; j++) {
                int w = o * in_dim + start + j;
                Wq[w] = lut[g * palette + idx[w]];
            }
    }

    int rem = in_dim % group_size;
    if (rem > 0) {
        int start = n_groups * group_size;
        for (int o = 0; o < out_dim; o++)
            for (int j = 0; j < rem; j++) {
                idx[o * in_dim + start + j] = 0;
                Wq[o * in_dim + start + j] = W_orig[o * in_dim + start + j];
            }
    }

    *idx_out = idx; *lut_out = lut; *n_groups_out = n_groups;
    return Wq;
}

#define COSINE_THRESHOLD 0.995f
typedef struct { int bitwidth; int group_size; float cosine_sim; int method; } PalettizeConfig;

#define METHOD_GPTQ 0
#define METHOD_ACCF 1

static const int PRIORITY_TABLE[9][2] = {
    {4, 256}, {4, 128}, {4, 64},
    {6, 256}, {6, 128}, {6, 64},
    {8, 256}, {8, 128}, {8, 64},
};

static PalettizeConfig select_bitwidth(
    const float *W, int out_dim, int in_dim,
    const float *hess_mat, const float *hess_diag,
    const float *X, int n_samples
) {
    for (int p = 0; p < 9; p++) {
        int bw = PRIORITY_TABLE[p][0], gs = PRIORITY_TABLE[p][1];
        if (in_dim < gs) continue;

        float best_sim = 0.0f;
        int best_method = METHOD_GPTQ;

        /* Try GPTQ path */
        {
            float *Wc = xmalloc((size_t)out_dim * in_dim * sizeof(float));
            memcpy(Wc, W, (size_t)out_dim * in_dim * sizeof(float));
            uint8_t *idx = NULL; float *lut = NULL; int ng = 0;
            float *Wq = palettize_gptq(Wc, out_dim, in_dim, hess_mat, hess_diag, bw, gs, &idx, &lut, &ng);
            float sim = 0.0f;
            if (X && n_samples > 0)
                sim = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
            free(Wc); free(idx); free(lut); free(Wq);
            if (sim > best_sim) { best_sim = sim; best_method = METHOD_GPTQ; }
        }

        /* Try ACCF path */
        {
            uint8_t *idx = NULL; float *lut = NULL; int ng = 0;
            float *Wq = palettize_accf(W, out_dim, in_dim, hess_diag, bw, gs, &idx, &lut, &ng);
            float sim = 0.0f;
            if (X && n_samples > 0)
                sim = output_cosine(X, W, Wq, n_samples, out_dim, in_dim);
            free(idx); free(lut); free(Wq);
            if (sim > best_sim) { best_sim = sim; best_method = METHOD_ACCF; }
        }

        if (best_sim >= COSINE_THRESHOLD || p == 8) {
            PalettizeConfig r = {bw, gs, best_sim, best_method};
            return r;
        }
    }
    PalettizeConfig r = {8, 64, 0.0f, METHOD_GPTQ};
    return r;
}

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
/* Palettize one tensor with GPTQ.                                     */
/* ------------------------------------------------------------------ */
static int palettize_and_write(
    const float *W, int out_dim, int in_dim,
    const float *hess_mat, const float *hess_diag,
    const float *X, int n_samples,
    const char *out_dir, const char *tensor_name,
    TensorMeta *meta_out
) {
    PalettizeConfig cfg = select_bitwidth(W, out_dim, in_dim, hess_mat, hess_diag, X, n_samples);
    uint8_t *idx = NULL; float *lut = NULL; int ng = 0;
    float *Wq = NULL;
    float *Wc = xmalloc((size_t)out_dim * in_dim * sizeof(float));
    memcpy(Wc, W, (size_t)out_dim * in_dim * sizeof(float));
    clip_outliers(Wc, out_dim, in_dim);
    if (cfg.method == METHOD_GPTQ) {
        Wq = palettize_gptq(Wc, out_dim, in_dim, hess_mat, hess_diag, cfg.bitwidth, cfg.group_size, &idx, &lut, &ng);
    } else {
        Wq = palettize_accf(Wc, out_dim, in_dim, hess_diag, cfg.bitwidth, cfg.group_size, &idx, &lut, &ng);
    }
    uint8_t *packed = NULL;
    size_t pb = pack_indices_transposed(idx, out_dim, in_dim, cfg.bitwidth, &packed);
    char san[1024]; sanitize_for_file(tensor_name, san, sizeof(san));
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s.idx4", out_dir, san);
    FILE *f = fopen(path, "wb");
    if (!f) { free(Wc); free(idx); free(lut); free(Wq); free(packed); return -1; }
    fwrite(packed, 1, pb, f); fclose(f);
    int pal = 1 << cfg.bitwidth;
    snprintf(path, sizeof(path), "%s/%s.lut_scalar", out_dir, san);
    write_lut_fp16(lut, ng, pal, path);
    memset(meta_out, 0, sizeof(*meta_out));
    strncpy(meta_out->name, tensor_name, 511);
    meta_out->out_dim = out_dim; meta_out->in_dim = in_dim;
    meta_out->bitwidth = cfg.bitwidth; meta_out->group_size = cfg.group_size;
    meta_out->n_groups = ng; meta_out->consumer_transpose_y = 1;
    snprintf(meta_out->index_file, sizeof(meta_out->index_file), "%s.idx4", san);
    snprintf(meta_out->lut_file, sizeof(meta_out->lut_file), "%s.lut_scalar", san);
    meta_out->packed_len_bytes = pb;
    snprintf(path, sizeof(path), "%s/%s.idx4", out_dir, san);
    meta_out->sha256_idx = sha256_file(path);
    snprintf(path, sizeof(path), "%s/%s.lut_scalar", out_dir, san);
    meta_out->sha256_lut = sha256_file(path);
    free(Wc); free(idx); free(lut); free(Wq); free(packed);
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
            const float *hmat = act ? act->hessian_matrix : NULL;
            const float *X = act ? act->data : NULL;
            int n_samples = act ? act->n_samples : 0;

            if (palettize_and_write(W, ti->shape[0], ti->shape[1],
                                      hmat, hess, X, n_samples,
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
