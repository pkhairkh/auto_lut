#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "png.h"

/* ===========================================================================
 * preprocess.h — Donut-style image preprocessor for auto_lut.
 *
 * Mirrors HuggingFace DonutImageProcessor behavior for the case where the
 * input PNG is already at the target (896x896) size (which is true for
 * the Dolphin test corpus — pages are pre-rendered at 896x896 RGB).
 *
 * Pipeline (matches preprocessor_config.json):
 *   1. do_resize:    if image size != (target_w, target_h), nearest resize
 *   2. do_rescale:   x = x * rescale_factor  (1/255)
 *   3. do_normalize: x = (x - mean[c]) / std[c]   per channel
 *
 * Output layout: CHW float32, channels=3, H=target_h, W=target_w.
 *
 * Uses the parallel-agent `Image` API from png.h (fields: width, height,
 * channels, pixels; free with image_free()).
 * =========================================================================== */

/* Preprocess a PNG into a float32 CHW buffer (3 * target_h * target_w floats).
 *
 *   img          : input Image (any size, will be resized if needed)
 *   target_w/h   : output dimensions (896, 896 for Dolphin)
 *   mean[3]      : per-channel mean (e.g. {0.485, 0.456, 0.406})
 *   std[3]       : per-channel std  (e.g. {0.229, 0.224, 0.225})
 *   rescale      : rescale factor (1/255 for 8-bit PNGs)
 *
 * Returns a malloc'd float buffer of size 3*target_h*target_w, or NULL.
 * Caller owns the result. If img has alpha (channels==4) the alpha channel
 * is dropped. If img is grayscale (channels==1) it is replicated to 3 chans.
 */
float *preprocess_donut(const Image *img, int target_w, int target_h,
                         const float mean[3], const float std[3],
                         float rescale);

#endif /* PREPROCESS_H */
