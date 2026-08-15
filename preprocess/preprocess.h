#ifndef PREPROCESS_H
#define PREPROCESS_H
/*
 * preprocess.h - HuggingFace-style image preprocessor for auto_lut.
 *
 * Mirrors the DonutImageProcessor pipeline described by
 * preprocessor_config.json:
 *
 *   PNG (HWC uint8)  -->  resize (bilinear)  -->  rescale (*1/255)
 *                     -->  normalise (-mean / std)  -->  transpose (HWC->CHW)
 *                     -->  cast to FP16  -->  PixelValues (NCHW FP16)
 *
 * The output tensor is always 4-D NCHW with batch=1 and is laid out
 * contiguously in row-major order so it can be fed directly to the
 * downstream inference / palettization stages.
 */
#include "png.h"
#include <stdint.h>

/*
 * Preprocessed image tensor. data is a heap-allocated FP16 buffer of
 * length batch*channels*height*width owned by the struct; release it
 * with pixel_values_free().
 */
typedef struct {
    int batch;          /* always 1 for single-image preprocessing */
    int channels;       /* 3 for RGB */
    int height;         /* target height from preprocessor_config */
    int width;          /* target width from preprocessor_config */
    uint16_t *data;     /* NCHW FP16, row-major */
} PixelValues;

/*
 * Preprocess a decoded Image using the config stored at
 * preprocessor_config_path. Returns a heap-allocated PixelValues on
 * success or NULL on failure (bad path, unsupported config, OOM, ...).
 */
PixelValues *preprocess_image(Image *img, const char *preprocessor_config_path);

/* Release a PixelValues previously returned by preprocess_image. */
void pixel_values_free(PixelValues *pv);

#endif /* PREPROCESS_H */
