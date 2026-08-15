#ifndef PNG_H
#define PNG_H
/*
 * png.h - Minimal PNG decoder for the auto_lut project.
 *
 * Exposes a tiny Image struct (interleaved HWC uint8 RGB) and a single
 * loader function. The implementation lives in png.c and uses only the
 * system zlib (libz) for DEFLATE inflate; no third-party imaging
 * libraries are required.
 *
 * Supported subset (sufficient for the Dolphin calibration corpus):
 *   - 8-bit truecolour RGB (PNG colour type 2)
 *   - non-interlaced storage (interlace method 0)
 *
 * Other colour types / interlacing raise PNG_ERR_UNSUPPORTED. The
 * decoder is intentionally small so it can be reused as the image
 * front-end of the preprocess pipeline without pulling in libpng.
 */
#include <stdint.h>

/* Opaque-ish decoded image. channels is always 3 for the supported
 * RGB path. data is row-major HWC: pixel (y,x,c) at
 * index (y*width + x)*channels + c. */
typedef struct {
    int width;
    int height;
    int channels;       /* always 3 for colour type 2 */
    uint8_t *data;      /* heap-allocated, owned by the Image */
} Image;

/* Error codes returned by png_load_into / reported via Image == NULL. */
#define PNG_OK               0
#define PNG_ERR_IO          -1   /* file open / read failure */
#define PNG_ERR_SIGNATURE   -2   /* not a PNG file */
#define PNG_ERR_TRUNCATED   -3   /* unexpected EOF / bad chunk */
#define PNG_ERR_UNSUPPORTED -4   /* bit depth / colour type / interlace */
#define PNG_ERR_INFLATE     -5   /* zlib decompression failure */
#define PNG_ERR_FILTER      -6   /* bad scanline filter byte */
#define PNG_ERR_CRC         -7   /* chunk CRC mismatch */
#define PNG_ERR_MEMORY      -8   /* allocation failure */

/*
 * Load a PNG file into a freshly allocated Image. Returns NULL on
 * failure; on success the caller owns the Image and must release it
 * with image_free(). The function is the only public entry point
 * required by the preprocess module.
 */
Image *png_load(const char *path);

/* Release an Image previously returned by png_load. NULL-tolerant. */
void image_free(Image *img);

#endif /* PNG_H */
