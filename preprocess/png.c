/*
 * png.c - Minimal PNG decoder for the auto_lut preprocess module.
 *
 * Decodes the subset of PNG required by the Dolphin calibration corpus:
 *   - 8-bit truecolour RGB (colour type 2)
 *   - non-interlaced (interlace method 0)
 *
 * The decoder is deliberately small and dependency-free apart from the
 * system zlib (libz), which is used for DEFLATE inflate. It walks the
 * PNG chunk stream, validates the signature and CRCs, concatenates all
 * IDAT chunks, inflates them, then reverses the per-scanline filtering
 * applied by the encoder (None / Sub / Up / Average / Paeth).
 *
 * The output is a heap-allocated Image in interleaved HWC uint8 layout
 * which is exactly what the downstream resize / normalise stages
 * expect.
 */
#include "png.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* ---- Big-endian helpers -------------------------------------------------- */

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

/* PNG CRC-32 (polynomial 0xEDB88320). The table is computed lazily on
 * first use so the binary stays small and dependency-free. */
static uint32_t png_crc_table[256];
static int png_crc_table_ready = 0;

static void png_crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
        png_crc_table[n] = c;
    }
    png_crc_table_ready = 1;
}

static uint32_t png_crc(const uint8_t *buf, size_t len) {
    if (!png_crc_table_ready) png_crc_init();
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < len; i++)
        c = png_crc_table[(c ^ buf[i]) & 0xffu] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

/* ---- Paeth predictor ----------------------------------------------------- */

static int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p;
    int pb = p > b ? p - b : b - p;
    int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/* ---- File slurp ---------------------------------------------------------- */

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    *out_len = (size_t)sz;
    return buf;
}

/* ---- Chunk walk + IDAT collection --------------------------------------- */

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t  bit_depth;
    uint8_t  colour_type;
    uint8_t  compression;
    uint8_t  filter;
    uint8_t  interlace;
} ihdr_t;

#define PNG_SIG_LEN 8
static const uint8_t PNG_SIG[PNG_SIG_LEN] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
};

/*
 * Walk the chunk stream starting after the signature. Fills the IHDR
 * fields and aggregates every IDAT chunk into a single growable buffer.
 * Returns PNG_OK or a negative error code.
 */
static int parse_chunks(const uint8_t *buf, size_t len,
                        ihdr_t *ihdr, uint8_t **idat_out,
                        size_t *idat_len_out) {
    size_t off = PNG_SIG_LEN;
    int have_ihdr = 0;
    uint8_t *idat = NULL;
    size_t idat_len = 0;

    while (off + 8 <= len) {
        uint32_t clen = be32(buf + off);
        const uint8_t *ctype = buf + off + 4;
        size_t data_off = off + 8;
        if (data_off + clen + 4 > len) return PNG_ERR_TRUNCATED;

        /* CRC covers chunk type + data. */
        uint32_t want = be32(buf + data_off + clen);
        uint32_t got = png_crc(buf + off + 4, clen + 4);
        if (want != got) return PNG_ERR_CRC;

        if (memcmp(ctype, "IHDR", 4) == 0) {
            if (clen != 13) return PNG_ERR_TRUNCATED;
            ihdr->width       = be32(buf + data_off);
            ihdr->height      = be32(buf + data_off + 4);
            ihdr->bit_depth   = buf[data_off + 8];
            ihdr->colour_type = buf[data_off + 9];
            ihdr->compression = buf[data_off + 10];
            ihdr->filter      = buf[data_off + 11];
            ihdr->interlace   = buf[data_off + 12];
            have_ihdr = 1;
        } else if (memcmp(ctype, "IDAT", 4) == 0) {
            uint8_t *grown = (uint8_t *)realloc(idat, idat_len + clen);
            if (!grown) { free(idat); return PNG_ERR_MEMORY; }
            idat = grown;
            memcpy(idat + idat_len, buf + data_off, clen);
            idat_len += clen;
        } else if (memcmp(ctype, "IEND", 4) == 0) {
            break;
        }
        /* Other ancillary chunks are skipped. */
        off = data_off + clen + 4;
    }

    if (!have_ihdr)      { free(idat); return PNG_ERR_TRUNCATED; }
    if (!idat || idat_len == 0) { free(idat); return PNG_ERR_TRUNCATED; }
    *idat_out = idat;
    *idat_len_out = idat_len;
    return PNG_OK;
}

/* ---- Inflate + unfilter -------------------------------------------------- */

/*
 * Inflate the concatenated IDAT payload. The caller owns *out. The
 * expected output size is (height * (1 + stride)) where the leading 1
 * byte per scanline holds the filter selector.
 */
static int inflate_idat(const uint8_t *idat, size_t idat_len,
                        uint8_t **out, size_t *out_len, size_t expected) {
    uLongf dest_len = expected;
    uint8_t *dst = (uint8_t *)malloc(expected);
    if (!dst) return PNG_ERR_MEMORY;
    int rc = uncompress(dst, &dest_len, idat, idat_len);
    if (rc != Z_OK) { free(dst); return PNG_ERR_INFLATE; }
    if (dest_len != expected) { free(dst); return PNG_ERR_TRUNCATED; }
    *out = dst;
    *out_len = dest_len;
    return PNG_OK;
}

/*
 * Reverse the per-scanline PNG filtering. Operates in place on the
 * inflated buffer. stride is the byte length of one scanline excluding
 * the filter byte; bpp is the bytes-per-pixel (3 for RGB8).
 */
static int unfilter(uint8_t *raw, int height, int stride, int bpp) {
    uint8_t *prev = NULL;       /* previous (already reconstructed) scanline */
    uint8_t *cur  = raw;
    for (int y = 0; y < height; y++) {
        uint8_t ftype = cur[0];
        uint8_t *line = cur + 1;
        switch (ftype) {
        case 0: /* None */
            break;
        case 1: /* Sub */
            for (int i = 0; i < stride; i++) {
                int left = (i >= bpp) ? line[i - bpp] : 0;
                line[i] = (uint8_t)(line[i] + left);
            }
            break;
        case 2: /* Up */
            if (prev)
                for (int i = 0; i < stride; i++)
                    line[i] = (uint8_t)(line[i] + prev[i]);
            break;
        case 3: /* Average */
            for (int i = 0; i < stride; i++) {
                int left = (i >= bpp) ? line[i - bpp] : 0;
                int up   = prev ? prev[i] : 0;
                line[i] = (uint8_t)(line[i] + (uint8_t)((left + up) >> 1));
            }
            break;
        case 4: /* Paeth */
            for (int i = 0; i < stride; i++) {
                int left  = (i >= bpp) ? line[i - bpp] : 0;
                int up    = prev ? prev[i] : 0;
                int uplef = (prev && i >= bpp) ? prev[i - bpp] : 0;
                line[i] = (uint8_t)(line[i] + paeth(left, up, uplef));
            }
            break;
        default:
            return PNG_ERR_FILTER;
        }
        prev = line;
        cur  = line + stride;
    }
    return PNG_OK;
}

/* ---- Public API --------------------------------------------------------- */

Image *png_load(const char *path) {
    size_t flen = 0;
    uint8_t *buf = read_file(path, &flen);
    if (!buf) return NULL;

    if (flen < PNG_SIG_LEN || memcmp(buf, PNG_SIG, PNG_SIG_LEN) != 0) {
        free(buf);
        return NULL;
    }

    ihdr_t ihdr;
    memset(&ihdr, 0, sizeof(ihdr));
    uint8_t *idat = NULL;
    size_t idat_len = 0;
    int rc = parse_chunks(buf, flen, &ihdr, &idat, &idat_len);
    free(buf);
    if (rc != PNG_OK) { free(idat); return NULL; }

    /* Validate the subset we support. */
    if (ihdr.bit_depth != 8 || ihdr.colour_type != 2) { free(idat); return NULL; }
    if (ihdr.interlace != 0) { free(idat); return NULL; }
    if (ihdr.compression != 0 || ihdr.filter != 0) { free(idat); return NULL; }
    if (ihdr.width == 0 || ihdr.height == 0) { free(idat); return NULL; }

    int bpp    = 3;                                   /* RGB8 */
    int stride = (int)ihdr.width * bpp;
    size_t raw_len = (size_t)ihdr.height * (size_t)(stride + 1);
    uint8_t *raw = NULL;
    size_t raw_got = 0;
    rc = inflate_idat(idat, idat_len, &raw, &raw_got, raw_len);
    free(idat);
    if (rc != PNG_OK) return NULL;

    rc = unfilter(raw, (int)ihdr.height, stride, bpp);
    if (rc != PNG_OK) { free(raw); return NULL; }

    /* Compact the scanlines: strip the per-line filter byte so the
     * output is a pure interleaved HWC buffer. */
    Image *img = (Image *)malloc(sizeof(Image));
    if (!img) { free(raw); return NULL; }
    img->width    = (int)ihdr.width;
    img->height   = (int)ihdr.height;
    img->channels = 3;
    img->data = (uint8_t *)malloc((size_t)stride * ihdr.height);
    if (!img->data) { free(raw); free(img); return NULL; }
    for (uint32_t y = 0; y < ihdr.height; y++) {
        memcpy(img->data + (size_t)y * stride,
               raw + (size_t)y * (stride + 1) + 1,
               (size_t)stride);
    }
    free(raw);
    return img;
}

void image_free(Image *img) {
    if (!img) return;
    free(img->data);
    free(img);
}
