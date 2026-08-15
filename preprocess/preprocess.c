/* preprocess.c — Donut-style image preprocessor for auto_lut.
 *
 * Implements resize (nearest), rescale (1/255), normalize (per-channel mean/std).
 * Output is CHW float32 (3 * H * W). Uses the parallel-agent Image API. */
#include "preprocess.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* nearest-neighbor resize of an 8-bit RGB(A) image to (target_w, target_h).
 * Returns a malloc'd uint8 buffer of target_w*target_h*3 bytes (always RGB). */
static uint8_t *resize_rgb(const Image *img, int target_w, int target_h) {
    uint8_t *out = (uint8_t *)malloc((size_t)target_w * target_h * 3);
    if (!out) return NULL;
    int src_ch = img->channels;
    for (int y = 0; y < target_h; y++) {
        int sy = (int)floorf((float)y * img->height / target_h);
        if (sy >= img->height) sy = img->height - 1;
        for (int x = 0; x < target_w; x++) {
            int sx = (int)floorf((float)x * img->width / target_w);
            if (sx >= img->width) sx = img->width - 1;
            const uint8_t *sp = img->pixels + ((size_t)sy * img->width + sx) * src_ch;
            uint8_t *dp = out + ((size_t)y * target_w + x) * 3;
            if (src_ch >= 3) {
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            } else if (src_ch == 1) {
                dp[0] = dp[1] = dp[2] = sp[0];
            } else {
                dp[0] = dp[1] = dp[2] = 0;
            }
        }
    }
    return out;
}

float *preprocess_donut(const Image *img, int target_w, int target_h,
                         const float mean[3], const float std[3],
                         float rescale) {
    if (!img || target_w <= 0 || target_h <= 0) return NULL;

    /* Step 1: resize to target (or copy if already correct size & RGB) */
    uint8_t *rgb;
    int owns_rgb = 0;
    if (img->width == target_w && img->height == target_h && img->channels == 3) {
        rgb = img->pixels;
        owns_rgb = 0;
    } else {
        rgb = resize_rgb(img, target_w, target_h);
        if (!rgb) return NULL;
        owns_rgb = 1;
    }

    /* Step 2+3: rescale + normalize, layout CHW */
    size_t npix = (size_t)target_w * target_h;
    float *out = (float *)malloc(npix * 3 * sizeof(float));
    if (!out) { if (owns_rgb) free(rgb); return NULL; }
    for (int c = 0; c < 3; c++) {
        float *op = out + c * npix;
        for (size_t i = 0; i < npix; i++) {
            float v = (float)rgb[i * 3 + c] * rescale;
            op[i] = (v - mean[c]) / std[c];
        }
    }
    if (owns_rgb) free(rgb);
    return out;
}
