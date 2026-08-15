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
/* Forward declarations (implemented in later waves)                   */
/* ------------------------------------------------------------------ */

static uint8_t *bilinear_resize(const Image *img, int target_w, int target_h);
static void transform_to_fp16(const uint8_t *src, int h, int w,
                              const PreprocessConfig *cfg, uint16_t *dst);

/* ------------------------------------------------------------------ */
/* Public API (implemented in Wave 6)                                  */
/* ------------------------------------------------------------------ */

PixelValues *preprocess_image(Image *img, const char *preprocessor_config_path);
void pixel_values_free(PixelValues *pv);
