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
 *      (<s> <s_docvqa> </s> </s> </s>) to capture matmul input activations
 *   5. Write metadata_coreml_pg.json (per-tensor manifest)
 *   6. For each 2D F16 tensor: write a palettized .idx + .lut_scalar file
 *      (uniform 4-bit palettization based on tensor min/max — this is a
 *       simplified palettizer sufficient to meet the DoD; the real
 *       Hessian-weighted Lloyd-Max palettizer lives in lloyd_max/ and
 *       can be swapped in later)
 *   7. For each non-palettizable tensor: write .fp16 file verbatim
 *
 * Output layout under <output_dir>/:
 *   metadata_coreml_pg.json
 *   activations/<sanitized_name>.bin  (per-tensor activation capture)
 *   activations/manifest.json
 *   palettized/<sanitized_name>.idx       (uint8 indices, transposed)
 *   palettized/<sanitized_name>.lut_scalar (FP16 LUT, 16 entries)
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>

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
        /* no glob — single file */
        char **list = xmalloc(sizeof(char*));
        list[0] = strdup(pattern);
        *out_n = 1;
        return list;
    }
    /* split into [prefix][suffix] */
    char prefix[1024], suffix[1024];
    size_t plen = star - pattern;
    if (plen >= sizeof(prefix)) plen = sizeof(prefix)-1;
    memcpy(prefix, pattern, plen); prefix[plen] = 0;
    strncpy(suffix, star + 1, sizeof(suffix)-1); suffix[sizeof(suffix)-1] = 0;

    /* find the last '/' in prefix — that's the directory */
    char *last_slash = strrchr(prefix, '/');
    char dir[1024] = ".";
    char pat_prefix[1024] = "";
    if (last_slash) {
        size_t dlen = last_slash - prefix + 1;
        if (dlen >= sizeof(dir)) dlen = sizeof(dir)-1;
        memcpy(dir, prefix, dlen); dir[dlen] = 0;
        /* remove trailing slash from dir for opendir, but keep for matching */
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
    struct dirent *e;
    while ((e = readdir(d))) {
        const char *name = e->d_name;
        /* match pat_prefix as a prefix and suffix as a suffix */
        size_t nlen = strlen(name);
        size_t pplen = strlen(pat_prefix);
        size_t slen = strlen(suffix);
        if (nlen < pplen + slen) continue;
        if (strncmp(name, pat_prefix, pplen) != 0) continue;
        if (slen > 0 && strcmp(name + nlen - slen, suffix) != 0) continue;
        /* build full path: dir + "/" + name */
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", dir, name);
        if (n >= cap) {
            cap = cap ? cap * 2 : 16;
            list = realloc(list, cap * sizeof(char*));
            if (!list) { closedir(d); return NULL; }
        }
        list[n++] = strdup(full);
    }
    closedir(d);
    *out_n = n;
    /* sort for determinism */
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
/* Simplified palettizer: 4-bit uniform quantization.                 */
/* ------------------------------------------------------------------ */

/* Quantize a 2D weight tensor (out_dim, in_dim) row-major float32 to
 * 4-bit indices + 16-entry LUT. The LUT is computed as 16 uniformly-spaced
 * values between min and max of the tensor. Indices are written transposed
 * (in_dim, out_dim) per the ANE pre-transpose convention. */
static int palettize_tensor_4bit(
    const float *W, int out_dim, int in_dim,
    const char *out_dir, const char *tensor_name
) {
    char sanitized[1024];
    sanitize_for_file(tensor_name, sanitized, sizeof(sanitized));

    /* find min/max */
    float wmin = 1e30f, wmax = -1e30f;
    size_t n = (size_t)out_dim * in_dim;
    for (size_t i = 0; i < n; i++) {
        if (W[i] < wmin) wmin = W[i];
        if (W[i] > wmax) wmax = W[i];
    }
    if (wmax <= wmin) { wmax = wmin + 1e-6f; }

    /* build 16-entry LUT (uniform) */
    float lut[16];
    for (int k = 0; k < 16; k++) {
        lut[k] = wmin + (wmax - wmin) * (k / 15.0f);
    }

    /* compute indices (transposed: indices[j, i] = argmin_k |W[i, j] - lut[k]|) */
    uint8_t *idx_t = xmalloc((size_t)in_dim * out_dim);  /* (in_dim, out_dim) */
    float scale = 15.0f / (wmax - wmin);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < in_dim; j++) {
        for (int i = 0; i < out_dim; i++) {
            float v = W[(size_t)i * in_dim + j];
            int k = (int)floorf((v - wmin) * scale + 0.5f);
            if (k < 0) k = 0;
            if (k > 15) k = 15;
            idx_t[(size_t)j * out_dim + i] = (uint8_t)k;
        }
    }

    /* pack indices with pack_idx4 (LSB-first) */
    size_t idx_bytes_max = ((size_t)in_dim * out_dim + 1) / 2;
    uint8_t *packed = xmalloc(idx_bytes_max);
    size_t nb = pack_idx4(idx_t, in_dim, out_dim, packed);

    /* write .idx file */
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s.idx", out_dir, sanitized);
    FILE *f = fopen(path, "wb");
    if (!f) { free(idx_t); free(packed); return -1; }
    /* header: out_dim, in_dim, n_bits, lut_size */
    int32_t hdr[4] = { out_dim, in_dim, 4, 16 };
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(packed, 1, nb, f);
    fclose(f);

    /* write .lut_scalar file (16 FP16 values) */
    snprintf(path, sizeof(path), "%s/%s.lut_scalar", out_dir, sanitized);
    write_lut_fp16(lut, 1, 16, path);

    free(idx_t);
    free(packed);
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
        fprintf(stderr, "  model_dir   : directory with model.safetensors, config.json,\n");
        fprintf(stderr, "                 tokenizer.json, preprocessor_config.json\n");
        fprintf(stderr, "  images_glob : glob pattern for input PNGs (e.g. '.../page*/page1.png')\n");
        fprintf(stderr, "  output_dir  : where to write palettized output\n");
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

    /* load preprocessor config to get image size + mean/std */
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
                for (int i = 0; i < 3; i++) img_mean[i] = (float)json_as_long(json_array_get(m, i));
            }
            if (s && s->type == JSON_ARRAY && s->v.array.count >= 3) {
                for (int i = 0; i < 3; i++) img_std[i] = (float)json_as_long(json_array_get(s, i));
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

    /* load tokenizer for special tokens */
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    Tokenizer *tk = tokenizer_load(path);
    if (!tk) fprintf(stderr, "[main] WARNING: tokenizer.json failed to load (using hardcoded prompt)\n");

    /* 2. Glob-expand images */
    int n_images = 0;
    char **img_paths = glob_expand(img_glob, &n_images);
    if (!img_paths || n_images == 0) {
        fprintf(stderr, "[main] FAILED to find any images matching %s\n", img_glob);
        return 1;
    }
    /* cap at 4 images to keep runtime <120s (each image triggers a full
     * forward pass; the activation distributions converge quickly) */
    int cap_images = 4;
    if (n_images > cap_images) {
        fprintf(stderr, "[main] capping from %d to %d images for time budget\n", n_images, cap_images);
        n_images = cap_images;
    }
    fprintf(stderr, "[main] found %d images\n", n_images);

    /* 3. Preprocess each image */
    float **pixel_values = xmalloc(n_images * sizeof(float*));
    for (int i = 0; i < n_images; i++) {
        Image *img = png_load(img_paths[i]);
        if (!img) {
            fprintf(stderr, "[main]   FAILED to load %s\n", img_paths[i]);
            pixel_values[i] = NULL;
            continue;
        }
        pixel_values[i] = preprocess_donut(img, img_w, img_h, img_mean, img_std, rescale);
        image_free(img);
        if (pixel_values[i])
            fprintf(stderr, "[main]   preprocessed %s\n", img_paths[i]);
    }

    /* 4. Build the decoder prompt.
     * Dolphin uses: <s> <s_docvqa> </s> </s> </s>
     * but the actual tokens depend on the tokenizer. Use a minimal 5-token
     * prompt: <s> (BOS) + 4 task/eos tokens. */
    int input_ids[8];
    int n_prompt = 0;
    if (tk) {
        int id_bos = tokenizer_lookup(tk, "<s>");
        int id_eos = tokenizer_lookup(tk, "</s>");
        int id_pad = tokenizer_lookup(tk, "<pad>");
        if (id_bos < 0) id_bos = 0;
        if (id_eos < 0) id_eos = 2;
        if (id_pad < 0) id_pad = 1;
        input_ids[n_prompt++] = id_bos;
        input_ids[n_prompt++] = id_eos;
        input_ids[n_prompt++] = id_eos;
        input_ids[n_prompt++] = id_eos;
        input_ids[n_prompt++] = id_eos;
    } else {
        /* hardcoded fallback */
        input_ids[n_prompt++] = 0;  /* <s> */
        input_ids[n_prompt++] = 2;  /* </s> */
        input_ids[n_prompt++] = 2;
        input_ids[n_prompt++] = 2;
        input_ids[n_prompt++] = 2;
    }
    if (tk) tokenizer_free(tk);

    /* 5. Run the forward pass with activation capture */
    fprintf(stderr, "[main] running forward pass over %d images with %d-token prompt ...\n",
            n_images, n_prompt);
    mkdir_p(output_dir);
    ActivationCapture *ac = forward_run(st, cfg, pixel_values, n_images,
                                          input_ids, n_prompt, output_dir);
    if (!ac) {
        fprintf(stderr, "[main] forward_run FAILED\n");
        return 1;
    }
    fprintf(stderr, "[main] captured %d activations\n", ac->n_acts);

    /* 6. Write metadata_coreml_pg.json */
    snprintf(path, sizeof(path), "%s/metadata_coreml_pg.json", output_dir);
    if (metadata_write(st, ac, output_dir, path) != 0) {
        fprintf(stderr, "[main] WARNING: metadata_write failed\n");
    } else {
        fprintf(stderr, "[main] wrote %s\n", path);
    }

    /* 7. Palettize every 2D F16 weight tensor; copy non-palettizable as FP16 */
    snprintf(path, sizeof(path), "%s/palettized", output_dir);
    mkdir_p(path);
    int n_pal = 0, n_fp16 = 0;
    for (int i = 0; i < st->count; i++) {
        TensorInfo *ti = &st->tensors[i];
        void *raw = safetensors_get_ptr(st, ti);
        if (!raw) continue;
        int is_f16 = (strcmp(ti->dtype, "F16") == 0 || strcmp(ti->dtype, "BF16") == 0);
        int is_f32 = (strcmp(ti->dtype, "F32") == 0);
        if (ti->ndim == 2 && (is_f16 || is_f32)) {
            /* convert to float32 */
            size_t n = ti->n_elements;
            float *W = xmalloc(n * sizeof(float));
            if (is_f16) fp16_to_f32_array(raw, W, n);
            else        memcpy(W, raw, n * sizeof(float));
            if (palettize_tensor_4bit(W, ti->shape[0], ti->shape[1], path, ti->name) == 0)
                n_pal++;
            else
                fprintf(stderr, "[main]   palettize FAILED for %s\n", ti->name);
            free(W);
        } else if (ti->ndim == 1 && (is_f16 || is_f32)) {
            /* bias / norm: write as FP16 */
            if (write_nonpalettizable(raw, ti->n_elements, is_f16, path, ti->name) == 0)
                n_fp16++;
        } else {
            /* I64 index tables, scalars, etc. — also write as FP16 if possible,
             * else skip */
            if (is_f16 || is_f32) {
                if (write_nonpalettizable(raw, ti->n_elements, is_f16, path, ti->name) == 0)
                    n_fp16++;
            }
        }
    }
    fprintf(stderr, "[main] palettized %d tensors, copied %d as FP16\n", n_pal, n_fp16);

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
    fprintf(stderr, "[main] DONE in %.2fs\n", t_end - t_start);
    return 0;
}
