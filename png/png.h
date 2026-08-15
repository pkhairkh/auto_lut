#ifndef PNG_H
#define PNG_H
/*
 * png.h - Minimal PNG decoder for the auto_lut project.
 *
 * Pure C11, no external libraries. The implementation in png.c
 * contains a self-contained DEFLATE/zlib inflate (no libz) so the
 * decoder can be dropped into any toolchain without third-party
 * dependencies.
 *
 * Supported subset (sufficient for the Dolphin calibration corpus):
 *   - PNG signature (8-byte) validation
 *   - Colour types: 0 (grayscale), 2 (RGB), 6 (RGBA)
 *   - Bit depth: 8 (the depth used by the calibration PNGs)
 *   - Non-interlaced storage (interlace method 0)
 *   - All five PNG scanline filters (None, Sub, Up, Average, Paeth)
 *
 * Unsupported features raise PNG_ERR_UNSUPPORTED:
 *   - bit depths != 8
 *   - colour types 3 (palette), 4 (grayscale+alpha) -> easy to add later
 *   - interlace methods != 0 (Adam7 not implemented)
 *
 * The decoded image is row-major HWC interleaved uint8:
 *   pixel (y, x, c) lives at index (y*width + x)*channels + c
 */
#include <stdint.h>

/* Decoded image. channels is 1 (grayscale), 3 (RGB) or 4 (RGBA).
 * pixels is heap-allocated and owned by the Image; release with
 * image_free(). */
typedef struct {
    int width;
    int height;
    int channels;
    uint8_t *pixels;
} Image;

/* Public error codes (informative; png_load just returns NULL on
 * failure, but png_last_error() exposes the reason for diagnostics). */
#define PNG_OK                0
#define PNG_ERR_IO           -1   /* file open / read failure */
#define PNG_ERR_SIGNATURE    -2   /* not a PNG file */
#define PNG_ERR_TRUNCATED    -3   /* unexpected EOF / short chunk */
#define PNG_ERR_UNSUPPORTED  -4   /* bit depth / colour type / interlace */
#define PNG_ERR_INFLATE      -5   /* zlib / DEFLATE decompression failure */
#define PNG_ERR_FILTER       -6   /* bad scanline filter byte */
#define PNG_ERR_CRC          -7   /* chunk CRC mismatch */
#define PNG_ERR_MEMORY       -8   /* allocation failure */

/*
 * Load a PNG file into a freshly allocated Image. Returns NULL on
 * failure; on success the caller owns the Image and must release it
 * with image_free().
 *
 * On failure, png_last_error() returns the specific PNG_ERR_* code
 * for diagnostics (thread-local: safe to call immediately after the
 * failing png_load() in the same thread).
 */
Image *png_load(const char *path);

/* Release an Image previously returned by png_load. NULL-tolerant. */
void image_free(Image *img);

/* Return the last error code recorded by png_load in the calling
 * thread. PNG_OK means no error has been recorded (or the last call
 * succeeded). */
int png_last_error(void);

#endif /* PNG_H */
