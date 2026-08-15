/*
 * preprocess.c - Donut-style image preprocessor for auto_lut.
 *
 * Pipeline (mirrors HuggingFace DonutImageProcessor):
 *
 *   Image (HWC uint8)  --bilinear resize-->  (th, tw, 3) uint8
 *                       --rescale *1/255-->   float32
 *                       --normalise (-mean/std per channel)-->  float32
 *                       --transpose HWC->CHW-->  float32
 *                       --cast FP16-->  PixelValues (1,3,th,tw) FP16
 *
 * Config is read from a preprocessor_config.json file using the JSON
 * parser shared with the safetensors module (../safetensors/json.{h,c}).
 */
#include "preprocess.h"
#include "fp16.h"
#include "../safetensors/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int   target_h;
    int   target_w;
    float rescale_factor;   /* typically 1/255 */
    float mean[3];          /* per-channel RGB mean */
    float std[3];           /* per-channel RGB std  */
} PreprocessConfig;

/*
 * Parse a HuggingFace-style preprocessor_config.json into a
 * PreprocessConfig. Accepts both the "size": {"height","width"} object
 * form and a legacy "image_size": {"height","width"} key. Falls back to
 * square 896x896 if size is absent. Mean/std default to the standard
 * ImageNet constants; rescale defaults to 1/255.
 *
 * Returns 0 on success, -1 on failure (bad path / parse error).
 */
static int parse_config(const char *path, PreprocessConfig *cfg) {
    /* Sensible ImageNet defaults so a partial config still works. */
    cfg->target_h = 896;
    cfg->target_w = 896;
    cfg->rescale_factor = 1.0f / 255.0f;
    const float def_mean[3] = {0.485f, 0.456f, 0.406f};
    const float def_std[3]  = {0.229f, 0.224f, 0.225f};
    memcpy(cfg->mean, def_mean, sizeof(def_mean));
    memcpy(cfg->std,  def_std,  sizeof(def_std));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) { fclose(f); return -1; }
    char *text = (char *)malloc((size_t)sz + 1);
    if (!text) { fclose(f); return -1; }
    size_t rd = fread(text, 1, (size_t)sz, f);
    fclose(f);
    text[rd] = '\0';

    JsonValue *root = json_parse(text);
    free(text);
    if (!root || root->type != JSON_OBJECT) { json_free(root); return -1; }

    /* size: {"height":..,"width":..}  or  image_size: {...} */
    const JsonValue *size = json_object_get(root, "size");
    if (!size) size = json_object_get(root, "image_size");
    if (size && size->type == JSON_OBJECT) {
        const JsonValue *h = json_object_get(size, "height");
        const JsonValue *w = json_object_get(size, "width");
        if (h) cfg->target_h = (int)json_as_long(h);
        if (w) cfg->target_w = (int)json_as_long(w);
        /* Some configs use "shortest_edge" only; treat as square. */
        if (!h && !w) {
            const JsonValue *se = json_object_get(size, "shortest_edge");
            if (se) { cfg->target_h = cfg->target_w = (int)json_as_long(se); }
        }
    }

    const JsonValue *rf = json_object_get(root, "rescale_factor");
    if (rf && rf->type == JSON_NUMBER) cfg->rescale_factor = (float)rf->v.number;

    const JsonValue *mean = json_object_get(root, "image_mean");
    if (mean && mean->type == JSON_ARRAY && mean->v.array.count >= 3) {
        for (int i = 0; i < 3; i++) {
            const JsonValue *m = json_array_get(mean, (size_t)i);
            if (m && m->type == JSON_NUMBER) cfg->mean[i] = (float)m->v.number;
        }
    }

    const JsonValue *std = json_object_get(root, "image_std");
    if (std && std->type == JSON_ARRAY && std->v.array.count >= 3) {
        for (int i = 0; i < 3; i++) {
            const JsonValue *s = json_array_get(std, (size_t)i);
            if (s && s->type == JSON_NUMBER) cfg->std[i] = (float)s->v.number;
        }
    }

    /* Guard against zero std which would divide by zero later. */
    for (int i = 0; i < 3; i++)
        if (cfg->std[i] == 0.0f) cfg->std[i] = 1.0f;

    json_free(root);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Bilinear resize (Wave 4)                                            */
/* ------------------------------------------------------------------ */

/*
 * Bilinearly resize an interleaved HWC uint8 RGB image to
 * (target_w, target_h). Uses the half-pixel-center coordinate mapping
 * that matches PIL/HuggingFace resample=BILINEAR (resample=2):
 *
 *   src = (dst + 0.5) * (src_size / dst_size) - 0.5
 *
 * Coordinates are clamped to the valid source range so borders never
 * read out of bounds. Returns a malloc'd buffer of
 * target_w*target_h*3 bytes (always 3 channels) or NULL on OOM.
 */
static uint8_t *bilinear_resize(const Image *img, int target_w, int target_h) {
    int src_w = img->width;
    int src_h = img->height;
    int src_c = img->channels;
    if (src_c < 3) src_c = 3;                 /* promote grayscale by replication */

    uint8_t *out = (uint8_t *)malloc((size_t)target_w * target_h * 3);
    if (!out) return NULL;

    const float sx_ratio = (float)src_w / (float)target_w;
    const float sy_ratio = (float)src_h / (float)target_h;

    for (int dy = 0; dy < target_h; dy++) {
        float sy = ((float)dy + 0.5f) * sy_ratio - 0.5f;
        int   y0 = (int)floorf(sy);
        int   y1 = y0 + 1;
        float fy = sy - (float)y0;
        if (y0 < 0) { y0 = 0; fy = 0.0f; }
        if (y1 < 0) y1 = 0;
        if (y0 > src_h - 1) { y0 = src_h - 1; fy = 0.0f; }
        if (y1 > src_h - 1) y1 = src_h - 1;

        for (int dx = 0; dx < target_w; dx++) {
            float sx = ((float)dx + 0.5f) * sx_ratio - 0.5f;
            int   x0 = (int)floorf(sx);
            int   x1 = x0 + 1;
            float fx = sx - (float)x0;
            if (x0 < 0) { x0 = 0; fx = 0.0f; }
            if (x1 < 0) x1 = 0;
            if (x0 > src_w - 1) { x0 = src_w - 1; fx = 0.0f; }
            if (x1 > src_w - 1) x1 = src_w - 1;

            const uint8_t *p00 = img->data + ((size_t)y0 * src_w + x0) * img->channels;
            const uint8_t *p01 = img->data + ((size_t)y0 * src_w + x1) * img->channels;
            const uint8_t *p10 = img->data + ((size_t)y1 * src_w + x0) * img->channels;
            const uint8_t *p11 = img->data + ((size_t)y1 * src_w + x1) * img->channels;
            uint8_t *o = out + ((size_t)dy * target_w + dx) * 3;

            for (int c = 0; c < 3; c++) {
                float v00 = (float)p00[c];
                float v01 = (float)p01[c];
                float v10 = (float)p10[c];
                float v11 = (float)p11[c];
                /* Bilinear: top row, bottom row, then vertical blend. */
                float top  = v00 + (v01 - v00) * fx;
                float bot  = v10 + (v11 - v10) * fx;
                float val  = top + (bot - top) * fy;
                int   ival = (int)(val + 0.5f);
                if (ival < 0)   ival = 0;
                if (ival > 255) ival = 255;
                o[c] = (uint8_t)ival;
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Transform: rescale + normalise + transpose + FP16 (Wave 5)          */
/* ------------------------------------------------------------------ */

/*
 * Convert an interleaved HWC uint8 RGB buffer into a contiguous NCHW
 * FP16 buffer. For each pixel and channel the transform is:
 *
 *   fp16 = (uint8 * rescale_factor - mean[c]) / std[c]
 *
 * The HWC->CHW transpose is performed on the fly: the destination is
 * laid out as [channel][y][x] so dst[c*H*W + y*W + x] holds the value
 * for channel c at spatial position (y,x). This matches the layout
 * expected by the downstream inference / palettization stages.
 *
 * The loop is written to be auto-vectorisable: the inner x-loop walks
 * contiguous source memory and writes contiguous destination memory per
 * channel, so -O3 -march=native can emit AVX2/AVX-512 for the float
 * arithmetic and the fp32->fp16 conversion.
 */
static void transform_to_fp16(const uint8_t *src, int h, int w,
                              const PreprocessConfig *cfg, uint16_t *dst) {
    const float rf = cfg->rescale_factor;
    const size_t plane = (size_t)h * (size_t)w;   /* samples per channel */

    for (int c = 0; c < 3; c++) {
        float mean = cfg->mean[c];
        float inv_std = 1.0f / cfg->std[c];
        uint16_t *out_plane = dst + c * plane;
        const uint8_t *in = src + c;                /* HWC: channel c */
        for (size_t i = 0; i < plane; i++) {
            float v = (float)in[i * 3] * rf;        /* rescale 1/255 */
            v = (v - mean) * inv_std;               /* normalise */
            out_plane[i] = fp32_to_fp16(v);         /* cast FP16 */
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public API (Wave 6)                                                 */
/* ------------------------------------------------------------------ */

/*
 * Run the full preprocessing pipeline on a decoded Image:
 *
 *   1. Parse the preprocessor config (size / mean / std / rescale).
 *   2. Bilinearly resize to (target_w, target_h).
 *   3. Rescale by 1/255, normalise per channel, transpose HWC->CHW,
 *      and cast to FP16.
 *
 * The returned PixelValues owns a single contiguous FP16 buffer of
 * shape (1, 3, target_h, target_w). Returns NULL on any failure
 * (bad config, OOM, NULL input image). The caller must release the
 * result with pixel_values_free().
 */
PixelValues *preprocess_image(Image *img, const char *preprocessor_config_path) {
    if (!img || !img->data || !preprocessor_config_path) return NULL;

    PreprocessConfig cfg;
    if (parse_config(preprocessor_config_path, &cfg) != 0) return NULL;
    if (cfg.target_w <= 0 || cfg.target_h <= 0) return NULL;

    /* Resize: if the source is already the target size we still run the
     * resize pass (it is a cheap identity copy) so the output buffer is
     * always a fresh, owned, 3-channel HWC buffer. */
    uint8_t *resized = bilinear_resize(img, cfg.target_w, cfg.target_h);
    if (!resized) return NULL;

    PixelValues *pv = (PixelValues *)malloc(sizeof(PixelValues));
    if (!pv) { free(resized); return NULL; }
    pv->batch    = 1;
    pv->channels = 3;
    pv->height   = cfg.target_h;
    pv->width    = cfg.target_w;

    size_t n = (size_t)pv->batch * pv->channels * pv->height * pv->width;
    pv->data = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!pv->data) { free(resized); free(pv); return NULL; }

    transform_to_fp16(resized, cfg.target_h, cfg.target_w, &cfg, pv->data);

    free(resized);
    return pv;
}

/*
 * Release a PixelValues previously returned by preprocess_image().
 * NULL-tolerant: calling on NULL is a no-op.
 */
void pixel_values_free(PixelValues *pv) {
    if (!pv) return;
    free(pv->data);
    free(pv);
}
