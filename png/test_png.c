/*
 * test_png.c - Smoke test for the auto_lut PNG decoder.
 *
 * Loads one or more PNG files (default: the first page of the Dolphin
 * calibration corpus) and prints width / height / channels plus a
 * handful of pixel statistics that make it easy to eyeball whether
 * the decoder produced sane output:
 *
 *   - pixel (0,0) raw bytes
 *   - per-channel min / max / mean over the whole image
 *   - count of non-zero pixels (sanity check: should be >> 0)
 *
 * Exit code is 0 if every requested file loaded and pixel (0,0) is
 * not all-zero; otherwise 1.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra \
 *       png/png.c png/test_png.c -o png/test_png
 *
 * Usage:
 *   ./png/test_png [path1.png] [path2.png] ...
 */
#include "png.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_PATH \
    "/root/dolphin/dolphin-palettize/test_50_pages/page01_2401_04100/page1.png"

static int stats_for_image(const Image *img) {
    size_t n = (size_t)img->width * (size_t)img->height * (size_t)img->channels;
    if (n == 0) {
        fprintf(stderr, "  empty image\n");
        return -1;
    }

    /* pixel (0,0) */
    printf("  pixel(0,0): ");
    for (int c = 0; c < img->channels; c++) {
        printf("%3d ", img->pixels[c]);
    }
    printf("\n");

    int all_zero_at_origin = 1;
    for (int c = 0; c < img->channels; c++) {
        if (img->pixels[c] != 0) { all_zero_at_origin = 0; break; }
    }
    if (all_zero_at_origin) {
        fprintf(stderr, "  FAIL: pixel(0,0) is all-zero\n");
        return -1;
    }

    /* per-channel min/max/mean */
    int minv[4] = {255, 255, 255, 255};
    int maxv[4] = {0, 0, 0, 0};
    double sum[4] = {0, 0, 0, 0};
    size_t nonzero_pixels = 0;
    for (size_t i = 0; i < (size_t)img->width * (size_t)img->height; i++) {
        int nz = 0;
        for (int c = 0; c < img->channels; c++) {
            uint8_t v = img->pixels[i * img->channels + c];
            if (v < minv[c]) minv[c] = v;
            if (v > maxv[c]) maxv[c] = v;
            sum[c] += v;
            if (v != 0) nz = 1;
        }
        if (nz) nonzero_pixels++;
    }
    double total_pixels = (double)img->width * (double)img->height;
    for (int c = 0; c < img->channels; c++) {
        printf("  ch%d: min=%3d max=%3d mean=%6.2f\n",
               c, minv[c], maxv[c], sum[c] / total_pixels);
    }
    printf("  non-zero pixels: %zu / %u (%.2f%%)\n",
           nonzero_pixels, img->width * img->height,
           100.0 * nonzero_pixels / total_pixels);

    /* a few samples from the middle to sanity-check spatial layout */
    printf("  pixel(w/2,h/2): ");
    int mid = (img->height / 2 * img->width + img->width / 2) * img->channels;
    for (int c = 0; c < img->channels; c++) {
        printf("%3d ", img->pixels[mid + c]);
    }
    printf("\n");
    printf("  pixel(w-1,h-1): ");
    int corner = ((img->height - 1) * img->width + (img->width - 1)) * img->channels;
    for (int c = 0; c < img->channels; c++) {
        printf("%3d ", img->pixels[corner + c]);
    }
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    const char **paths;
    int n_paths;
    const char *default_path = DEFAULT_PATH;
    if (argc >= 2) {
        paths = (const char **)argv + 1;
        n_paths = argc - 1;
    } else {
        paths = &default_path;
        n_paths = 1;
    }

    int failures = 0;
    for (int i = 0; i < n_paths; i++) {
        printf("[png] loading %s\n", paths[i]);
        Image *img = png_load(paths[i]);
        if (!img) {
            fprintf(stderr, "  FAIL: png_load returned NULL (err=%d)\n",
                    png_last_error());
            failures++;
            continue;
        }
        printf("  width=%d height=%d channels=%d\n",
               img->width, img->height, img->channels);
        if (stats_for_image(img) != 0) {
            failures++;
        }
        image_free(img);
    }

    if (failures) {
        printf("[png] %d/%d file(s) failed\n", failures, n_paths);
        return 1;
    }
    printf("[png] all %d file(s) OK\n", n_paths);
    return 0;
}
