/*
 * test_preprocess.c - DoD verification for the preprocess module.
 *
 * Loads a real Dolphin calibration PNG, runs the full preprocessing
 * pipeline, and asserts the Definition-of-Done criteria:
 *
 *   1. Output shape is (1, 3, 896, 896) FP16.
 *   2. No element is NaN or Inf.
 *   3. At least one element is non-zero (buffer is not all-zero).
 *   4. All elements lie within a reasonable range [-5, 5].
 *
 * Exits 0 on success, non-zero on any failure.
 */
#include "preprocess.h"
#include "fp16.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CONFIG_PATH "/root/dolphin/hf_model_main/preprocessor_config.json"

int main(int argc, char **argv) {
    const char *png_path = (argc > 1) ? argv[1]
        : "/root/dolphin/dolphin-palettize/test_50_pages/page00_2401_04088/page1.png";
    const char *cfg_path = (argc > 2) ? argv[2] : CONFIG_PATH;

    /* --- Load + preprocess --- */
    Image *img = png_load(png_path);
    if (!img) {
        fprintf(stderr, "FAIL: png_load(%s) returned NULL\n", png_path);
        return 1;
    }
    printf("loaded PNG: %dx%d c=%d\n", img->width, img->height, img->channels);

    PixelValues *pv = preprocess_image(img, cfg_path);
    image_free(img);
    if (!pv) {
        fprintf(stderr, "FAIL: preprocess_image returned NULL\n");
        return 1;
    }

    /* --- DoD 1: shape (1, 3, 896, 896) --- */
    int ok_shape = (pv->batch == 1 && pv->channels == 3 &&
                    pv->height == 896 && pv->width == 896);
    printf("shape: (%d, %d, %d, %d)  %s\n",
           pv->batch, pv->channels, pv->height, pv->width,
           ok_shape ? "OK" : "FAIL");
    if (!ok_shape) { pixel_values_free(pv); return 1; }

    /* --- Scan the FP16 buffer --- */
    size_t n = (size_t)pv->batch * pv->channels * pv->height * pv->width;
    int    has_nan = 0, has_inf = 0, has_nonzero = 0, out_of_range = 0;
    float  vmin = 1e30f, vmax = -1e30f;
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        float v = fp16_to_fp32(pv->data[i]);
        if (isnan(v)) { has_nan = 1; continue; }
        if (isinf(v)) { has_inf = 1; continue; }
        if (v != 0.0f) has_nonzero = 1;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        if (v < -5.0f || v > 5.0f) out_of_range++;
        sum += (double)v;
    }

    /* --- DoD 2: no NaN / Inf --- */
    printf("nan=%d inf=%d  %s\n", has_nan, has_inf,
           (!has_nan && !has_inf) ? "OK" : "FAIL");

    /* --- DoD 3: not all zero --- */
    printf("has_nonzero=%d  %s\n", has_nonzero, has_nonzero ? "OK" : "FAIL");

    /* --- DoD 4: reasonable range [-5, 5] --- */
    printf("range: [%.4f, %.4f]  out_of_range=%zu  %s\n",
           vmin, vmax, (size_t)out_of_range,
           (out_of_range == 0) ? "OK" : "FAIL");

    printf("mean=%.6f  n=%zu\n", sum / (double)n, n);

    /* Spot-check: channel 0, top-left pixel. For a white (255,255,255)
     * document background the normalised value is
     * (1.0 - 0.485) / 0.229 = 2.249. */
    float c0 = fp16_to_fp32(pv->data[0]);
    float c1 = fp16_to_fp32(pv->data[(size_t)896 * 896]);
    float c2 = fp16_to_fp32(pv->data[2 * (size_t)896 * 896]);
    printf("pixel(0,0): ch0=%.4f ch1=%.4f ch2=%.4f\n", c0, c1, c2);

    int passed = ok_shape && !has_nan && !has_inf && has_nonzero && (out_of_range == 0);
    pixel_values_free(pv);

    printf("\n=== DoD: %s ===\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
