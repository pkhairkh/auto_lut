/*
 * png.c - Minimal PNG decoder with self-contained DEFLATE/zlib inflate.
 *
 * Pure C11, no external libraries (no libz, no libpng). Implements:
 *
 *   1. PNG signature + chunk reader (IHDR, IDAT, IEND, CRC-32 check)
 *   2. zlib header (CMF/FLG) parsing
 *   3. DEFLATE decompressor
 *      - LSB-first bit reader
 *      - Canonical Huffman decoder (built from code lengths)
 *      - Block types: 0 (stored), 1 (fixed), 2 (dynamic), 3 (error)
 *      - LZ77 backreferences: length 3-258, distance 1-32768
 *   4. PNG scanline unfilter (None/Sub/Up/Average/Paeth)
 *
 * Supported PNG subset:
 *   - colour types 0 (grayscale), 2 (RGB), 6 (RGBA)
 *   - bit depth 8
 *   - non-interlaced (Adam7 not implemented)
 *
 * Reference: RFC 1951 (DEFLATE), RFC 1950 (zlib), PNG ISO/IEC 15948.
 */
#include "png.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Error reporting (thread-local so the API stays clean)              */
/* ------------------------------------------------------------------ */

static _Thread_local int g_last_error = PNG_OK;

int png_last_error(void) { return g_last_error; }

static void png_set_error(int code) { g_last_error = code; }

/* ------------------------------------------------------------------ */
/* CRC-32 (used for PNG chunk integrity checks)                       */
/* ------------------------------------------------------------------ */

static uint32_t crc32_table[256];
static int crc32_table_built = 0;

static void crc32_build_table(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[n] = c;
    }
    crc32_table_built = 1;
}

static uint32_t crc32_compute(const uint8_t *buf, size_t len) {
    if (!crc32_table_built) crc32_build_table();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        c = crc32_table[(c ^ buf[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ */
/* File reader: slurp entire file into a heap buffer                  */
/* ------------------------------------------------------------------ */

static uint8_t *read_file_all(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        png_set_error(PNG_ERR_IO);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        png_set_error(PNG_ERR_IO);
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        png_set_error(PNG_ERR_IO);
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        png_set_error(PNG_ERR_IO);
        fclose(f);
        return NULL;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf) {
        png_set_error(PNG_ERR_MEMORY);
        fclose(f);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        png_set_error(PNG_ERR_IO);
        free(buf);
        return NULL;
    }
    *out_len = (size_t)sz;
    return buf;
}

/* ------------------------------------------------------------------ */
/* PNG signature                                                      */
/* ------------------------------------------------------------------ */

static const uint8_t PNG_SIGNATURE[8] = {
    137, 80, 78, 71, 13, 10, 26, 10
};

static int check_png_signature(const uint8_t *buf, size_t len) {
    if (len < 8) {
        png_set_error(PNG_ERR_SIGNATURE);
        return -1;
    }
    if (memcmp(buf, PNG_SIGNATURE, 8) != 0) {
        png_set_error(PNG_ERR_SIGNATURE);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Chunk reader state                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    size_t total_len;
    size_t offset;          /* current read position */
} ChunkReader;

static uint32_t read_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
}

/* Read the next chunk header. Returns 1 on success, 0 on EOF, -1 on error.
 * On success, fills *out_type (4-byte tag, NOT null-terminated),
 * *out_data_len, *out_data_ptr (pointing into the file buffer), and
 * advances the reader past the chunk (data + 4-byte CRC). */
static int chunk_next(ChunkReader *r,
                      char out_type[4],
                      uint32_t *out_data_len,
                      const uint8_t **out_data) {
    /* Need 4 (length) + 4 (type) bytes minimum */
    if (r->offset + 8 > r->total_len) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    uint32_t dlen = read_u32_be(r->data + r->offset);
    const uint8_t *type_ptr = r->data + r->offset + 4;

    /* data + 4-byte CRC after the type */
    if (r->offset + 8 + (size_t)dlen + 4 > r->total_len) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    const uint8_t *data_ptr = type_ptr + 4;
    uint32_t crc_stored = read_u32_be(data_ptr + dlen);

    /* CRC covers type + data */
    uint32_t crc_calc = crc32_compute(type_ptr, 4 + dlen);
    if (crc_stored != crc_calc) {
        png_set_error(PNG_ERR_CRC);
        return -1;
    }

    memcpy(out_type, type_ptr, 4);
    *out_data_len = dlen;
    *out_data = data_ptr;
    r->offset += 8 + dlen + 4;
    return 1;
}

/* ------------------------------------------------------------------ */
/* IHDR parsing                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t  bit_depth;
    uint8_t  color_type;
    uint8_t  compression;   /* always 0 (DEFLATE) */
    uint8_t  filter;        /* always 0 (adaptive) */
    uint8_t  interlace;     /* 0 = none, 1 = Adam7 */
} IHDR;

static int parse_ihdr(const uint8_t *data, uint32_t len, IHDR *out) {
    if (len != 13) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    out->width       = read_u32_be(data + 0);
    out->height      = read_u32_be(data + 4);
    out->bit_depth   = data[8];
    out->color_type  = data[9];
    out->compression = data[10];
    out->filter      = data[11];
    out->interlace   = data[12];
    return 0;
}

/* ------------------------------------------------------------------ */
/* DEFLATE bit reader (LSB-first within each byte)                    */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    size_t total_len;
    size_t byte_pos;        /* next unread byte index */
    uint32_t bit_buf;       /* accumulated bits, LSB-first */
    int bit_count;          /* number of valid bits in bit_buf */
} BitReader;

static void bit_init(BitReader *br, const uint8_t *data, size_t len) {
    br->data = data;
    br->total_len = len;
    br->byte_pos = 0;
    br->bit_buf = 0;
    br->bit_count = 0;
}

/* Pull n bits (1..16) from the reader, LSB-first. */
static uint32_t bit_read(BitReader *br, int n) {
    while (br->bit_count < n) {
        if (br->byte_pos >= br->total_len) {
            /* Past end: feed zeros (DEFLATE will detect the resulting
             * invalid symbol and error out; caller checks bounds). */
            br->bit_buf |= 0u << br->bit_count;
            br->bit_count += 8;
            br->byte_pos++;   /* keep advancing so EOF is detectable */
        } else {
            br->bit_buf |= (uint32_t)br->data[br->byte_pos++] << br->bit_count;
            br->bit_count += 8;
        }
    }
    uint32_t v = br->bit_buf & ((1u << n) - 1u);
    br->bit_buf >>= n;
    br->bit_count -= n;
    return v;
}

/* Skip remaining bits in the current byte (used for stored blocks). */
static void bit_align_byte(BitReader *br) {
    int drop = br->bit_count & 7;
    br->bit_buf >>= drop;
    br->bit_count -= drop;
}

/* Read one raw byte (after byte alignment). Used for stored blocks. */
static int bit_read_byte(BitReader *br) {
    bit_align_byte(br);
    /* Drain whatever is left in the buffer first */
    if (br->bit_count >= 8) {
        uint8_t v = (uint8_t)(br->bit_buf & 0xFFu);
        br->bit_buf >>= 8;
        br->bit_count -= 8;
        return v;
    }
    if (br->byte_pos >= br->total_len) return -1;
    return br->data[br->byte_pos++];
}

/* ------------------------------------------------------------------ */
/* Canonical Huffman decoder                                          */
/* ------------------------------------------------------------------ */

#define MAX_HUFF_BITS  15
#define MAX_HUFF_SYMBOLS 288   /* lit/len alphabet size */

typedef struct {
    /* canonical Huffman decode table.
     *   count[i]   = number of codes of length i (1..MAX_HUFF_BITS)
     *   symbols[]  = symbols sorted by (length, code value)
     * For decoding, we use the standard RFC 1951 reference algorithm. */
    int count[MAX_HUFF_BITS + 1];
    int symbols[MAX_HUFF_SYMBOLS];
    int num_symbols;
} Huff;

static int huff_build(Huff *h, const uint8_t *code_lengths, int n) {
    memset(h->count, 0, sizeof(h->count));
    h->num_symbols = n;

    /* Count occurrences of each code length */
    for (int i = 0; i < n; i++) {
        if (code_lengths[i] > MAX_HUFF_BITS) {
            png_set_error(PNG_ERR_INFLATE);
            return -1;
        }
        h->count[code_lengths[i]]++;
    }
    h->count[0] = 0;   /* length 0 means "not present" */

    /* Validate: the Kraft inequality must hold. Compute the "left" check
     * from RFC 1951 §3.2.2. */
    int left = 1;
    for (int len = 1; len <= MAX_HUFF_BITS; len++) {
        left <<= 1;
        left -= h->count[len];
        if (left < 0) {
            png_set_error(PNG_ERR_INFLATE);
            return -1;
        }
    }
    /* left > 0 means an incomplete code; allowed only for the special
     * "single symbol with length 1" case. We accept incomplete codes
     * (some valid PNGs use sparse code-length tables) but flag errors
     * when an unreachable symbol is requested. */

    /* Build the offsets into the symbol table */
    int offsets[MAX_HUFF_BITS + 2];
    offsets[1] = 0;
    for (int len = 1; len < MAX_HUFF_BITS; len++) {
        offsets[len + 1] = offsets[len] + h->count[len];
    }

    /* Place symbols into the table sorted by (length, code) */
    for (int sym = 0; sym < n; sym++) {
        int len = code_lengths[sym];
        if (len > 0) {
            h->symbols[offsets[len]++] = sym;
        }
    }
    return 0;
}

/* Decode one symbol using the canonical Huffman table. */
static int huff_decode(BitReader *br, const Huff *h) {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len <= MAX_HUFF_BITS; len++) {
        code |= bit_read(br, 1);
        int count = h->count[len];
        if (code - first < count) {
            return h->symbols[index + (code - first)];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    /* No matching code */
    png_set_error(PNG_ERR_INFLATE);
    return -1;
}

/* ------------------------------------------------------------------ */
/* DEFLATE constants (length and distance tables, RFC 1951 §3.2.5)    */
/* ------------------------------------------------------------------ */

static const int LENGTH_BASE[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};
static const int LENGTH_EXTRA[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};
static const int DIST_BASE[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577
};
static const int DIST_EXTRA[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};

/* Order in which code-length-code lengths appear in dynamic blocks */
static const int CL_ORDER[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

/* ------------------------------------------------------------------ */
/* Output buffer (growable)                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t cap;
    size_t len;
} OutBuf;

static int outbuf_reserve(OutBuf *o, size_t extra) {
    if (o->len + extra <= o->cap) return 0;
    size_t new_cap = o->cap ? o->cap : (1u << 16);
    while (new_cap < o->len + extra) {
        new_cap <<= 1;
        if (new_cap < o->cap) {   /* overflow */
            png_set_error(PNG_ERR_MEMORY);
            return -1;
        }
    }
    uint8_t *nd = (uint8_t *)realloc(o->data, new_cap);
    if (!nd) {
        png_set_error(PNG_ERR_MEMORY);
        return -1;
    }
    o->data = nd;
    o->cap = new_cap;
    return 0;
}

static int outbuf_push(OutBuf *o, uint8_t b) {
    if (outbuf_reserve(o, 1) != 0) return -1;
    o->data[o->len++] = b;
    return 0;
}

static int outbuf_copy_back(OutBuf *o, size_t dist, size_t length) {
    if (dist == 0 || dist > o->len) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }
    if (outbuf_reserve(o, length) != 0) return -1;
    /* Note: ranges may overlap, so byte-by-byte copy is required. */
    size_t src = o->len - dist;
    for (size_t i = 0; i < length; i++) {
        o->data[o->len++] = o->data[src++];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* DEFLATE block decoders                                             */
/* ------------------------------------------------------------------ */

/* Build the fixed Huffman tables (RFC 1951 §3.2.6). */
static void build_fixed_huff(Huff *litlen, Huff *dist) {
    uint8_t lens[288];
    for (int i = 0;   i <= 143; i++) lens[i] = 8;
    for (int i = 144; i <= 255; i++) lens[i] = 9;
    for (int i = 256; i <= 279; i++) lens[i] = 7;
    for (int i = 280; i <= 287; i++) lens[i] = 8;
    huff_build(litlen, lens, 288);

    uint8_t dlens[30];
    for (int i = 0; i < 30; i++) dlens[i] = 5;
    huff_build(dist, dlens, 30);
}

/* Decode a dynamic Huffman block (RFC 1951 §3.2.7). */
static int decode_dynamic_huff(BitReader *br, Huff *litlen, Huff *dist) {
    int hlit  = (int)bit_read(br, 5) + 257;
    int hdist = (int)bit_read(br, 5) + 1;
    int hclen = (int)bit_read(br, 4) + 4;

    if (hlit > 286 || hdist > 30) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }

    uint8_t cl_lens[19] = {0};
    for (int i = 0; i < hclen; i++) {
        cl_lens[CL_ORDER[i]] = (uint8_t)bit_read(br, 3);
    }
    Huff cl_huff;
    if (huff_build(&cl_huff, cl_lens, 19) != 0) return -1;

    /* Decode the combined lit/len + dist code-length sequence */
    uint8_t lens[286 + 30] = {0};
    int total = hlit + hdist;
    int i = 0;
    while (i < total) {
        int sym = huff_decode(br, &cl_huff);
        if (sym < 0) return -1;
        if (sym < 16) {
            lens[i++] = (uint8_t)sym;
        } else if (sym == 16) {
            if (i == 0) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
            int rep = (int)bit_read(br, 2) + 3;
            uint8_t prev = lens[i - 1];
            while (rep-- > 0 && i < total) lens[i++] = prev;
            if (i > total) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
        } else if (sym == 17) {
            int rep = (int)bit_read(br, 3) + 3;
            while (rep-- > 0 && i < total) lens[i++] = 0;
            if (i > total) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
        } else if (sym == 18) {
            int rep = (int)bit_read(br, 7) + 11;
            while (rep-- > 0 && i < total) lens[i++] = 0;
            if (i > total) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
        } else {
            png_set_error(PNG_ERR_INFLATE);
            return -1;
        }
    }

    if (huff_build(litlen, lens, hlit) != 0) return -1;
    if (huff_build(dist, lens + hlit, hdist) != 0) return -1;
    return 0;
}

/* Decode the LZ77 stream of one block (litlen + dist tables already
 * built). Reads symbols until end-of-block (256). */
static int decode_block_body(BitReader *br, const Huff *litlen,
                             const Huff *dist, OutBuf *out) {
    for (;;) {
        int sym = huff_decode(br, litlen);
        if (sym < 0) return -1;
        if (sym == 256) {
            return 0;   /* end of block */
        } else if (sym < 256) {
            if (outbuf_push(out, (uint8_t)sym) != 0) return -1;
        } else {
            int len_idx = sym - 257;
            if (len_idx < 0 || len_idx >= 29) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
            int length = LENGTH_BASE[len_idx] +
                         (int)bit_read(br, LENGTH_EXTRA[len_idx]);
            int dsym = huff_decode(br, dist);
            if (dsym < 0 || dsym >= 30) {
                png_set_error(PNG_ERR_INFLATE);
                return -1;
            }
            int distance = DIST_BASE[dsym] +
                           (int)bit_read(br, DIST_EXTRA[dsym]);
            if (outbuf_copy_back(out, (size_t)distance, (size_t)length) != 0)
                return -1;
        }
    }
}

/* Decode a stored (BTYPE=0) block. */
static int decode_stored_block(BitReader *br, OutBuf *out) {
    bit_align_byte(br);
    /* Read LEN and NLEN (2 bytes each, little-endian, after alignment). */
    if (br->byte_pos + 4 > br->total_len) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    /* After alignment, the bit buffer should be empty or have less than
     * 8 bits; consume any leftover bytes from the buffer first. */
    int b0 = bit_read_byte(br);
    int b1 = bit_read_byte(br);
    int n0 = bit_read_byte(br);
    int n1 = bit_read_byte(br);
    if (b0 < 0 || b1 < 0 || n0 < 0 || n1 < 0) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    int len  = b0 | (b1 << 8);
    int nlen = n0 | (n1 << 8);
    if ((len ^ 0xFFFF) != nlen) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }
    if (outbuf_reserve(out, (size_t)len) != 0) return -1;
    for (int i = 0; i < len; i++) {
        int v = bit_read_byte(br);
        if (v < 0) {
            png_set_error(PNG_ERR_TRUNCATED);
            return -1;
        }
        if (outbuf_push(out, (uint8_t)v) != 0) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* zlib inflate: parse zlib header, then DEFLATE blocks              */
/* ------------------------------------------------------------------ */

static int zlib_inflate(const uint8_t *src, size_t src_len,
                        OutBuf *out) {
    if (src_len < 2) {
        png_set_error(PNG_ERR_TRUNCATED);
        return -1;
    }
    /* zlib header: CMF, FLG. CM = CMF & 0x0F must be 8 (DEFLATE).
     * CINFO = CMF >> 4 must be <= 7 (window size 2^(CINFO+8)). */
    uint8_t cmf = src[0];
    uint8_t flg = src[1];
    if ((cmf & 0x0F) != 8 || ((cmf >> 4) & 0x0F) > 7) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }
    if (((uint32_t)cmf << 8 | flg) % 31 != 0) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }
    /* FDICT bit (0x20) indicates a preset dictionary; we don't support
     * preset dictionaries (PNG never uses them). */
    if (flg & 0x20) {
        png_set_error(PNG_ERR_INFLATE);
        return -1;
    }

    BitReader br;
    bit_init(&br, src + 2, src_len - 2);

    Huff fixed_litlen, fixed_dist;
    int fixed_built = 0;

    for (;;) {
        int bfinal = (int)bit_read(&br, 1);
        int btype  = (int)bit_read(&br, 2);

        if (btype == 0) {
            if (decode_stored_block(&br, out) != 0) return -1;
        } else if (btype == 1) {
            if (!fixed_built) {
                build_fixed_huff(&fixed_litlen, &fixed_dist);
                fixed_built = 1;
            }
            if (decode_block_body(&br, &fixed_litlen, &fixed_dist, out) != 0)
                return -1;
        } else if (btype == 2) {
            Huff litlen, dist;
            if (decode_dynamic_huff(&br, &litlen, &dist) != 0) return -1;
            if (decode_block_body(&br, &litlen, &dist, out) != 0) return -1;
        } else {
            png_set_error(PNG_ERR_INFLATE);
            return -1;
        }

        if (bfinal) break;
    }
    /* Note: zlib appends a 4-byte Adler-32 checksum after the DEFLATE
     * data. We do not verify it (the PNG chunk CRC already guarantees
     * the IDAT bytes are intact). */
    return 0;
}

/* ------------------------------------------------------------------ */
/* PNG scanline unfilter                                              */
/* ------------------------------------------------------------------ */

static int paeth_predictor(int a, int b, int c) {
    int p = a + b - c;
    int pa = p - a; if (pa < 0) pa = -pa;
    int pb = p - b; if (pb < 0) pb = -pb;
    int pc = p - c; if (pc < 0) pc = -pc;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/*bpp = bytes per pixel (1 for grayscale, 3 for RGB, 4 for RGBA). */
static int unfilter(uint8_t *raw, uint32_t width, uint32_t height,
                    int channels) {
    int bpp = channels;
    size_t row_bytes = (size_t)width * (size_t)bpp;
    size_t stride = row_bytes + 1;   /* +1 for the filter-type byte */

    for (uint32_t y = 0; y < height; y++) {
        uint8_t *row = raw + (size_t)y * stride;
        uint8_t filter = row[0];
        uint8_t *px = row + 1;

        uint8_t *up_row = NULL;
        if (y > 0) up_row = (raw + (size_t)(y - 1) * stride) + 1;

        switch (filter) {
        case 0: /* None */
            break;
        case 1: /* Sub */
            for (size_t i = 0; i < row_bytes; i++) {
                int left = (i >= (size_t)bpp) ? px[i - bpp] : 0;
                px[i] = (uint8_t)(px[i] + left);
            }
            break;
        case 2: /* Up */
            for (size_t i = 0; i < row_bytes; i++) {
                int up = up_row ? up_row[i] : 0;
                px[i] = (uint8_t)(px[i] + up);
            }
            break;
        case 3: /* Average */
            for (size_t i = 0; i < row_bytes; i++) {
                int left = (i >= (size_t)bpp) ? px[i - bpp] : 0;
                int up   = up_row ? up_row[i] : 0;
                px[i] = (uint8_t)(px[i] + (uint8_t)((left + up) >> 1));
            }
            break;
        case 4: /* Paeth */
            for (size_t i = 0; i < row_bytes; i++) {
                int left = (i >= (size_t)bpp) ? px[i - bpp] : 0;
                int up   = up_row ? up_row[i] : 0;
                int upleft = (up_row && i >= (size_t)bpp) ? up_row[i - bpp] : 0;
                px[i] = (uint8_t)(px[i] + paeth_predictor(left, up, upleft));
            }
            break;
        default:
            png_set_error(PNG_ERR_FILTER);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

Image *png_load(const char *path) {
    png_set_error(PNG_OK);

    size_t file_len = 0;
    uint8_t *file_buf = read_file_all(path, &file_len);
    if (!file_buf) return NULL;

    if (check_png_signature(file_buf, file_len) != 0) {
        free(file_buf);
        return NULL;
    }

    ChunkReader r = { file_buf, file_len, 8 };

    IHDR ihdr = {0};
    int have_ihdr = 0;

    /* Accumulate IDAT data into one contiguous buffer */
    OutBuf idat = {0};

    int seen_iend = 0;
    int rc = 0;
    while (r.offset < r.total_len) {
        char type[4];
        uint32_t dlen;
        const uint8_t *data;
        rc = chunk_next(&r, type, &dlen, &data);
        if (rc <= 0) break;

        if (memcmp(type, "IHDR", 4) == 0) {
            if (have_ihdr || dlen != 13) {
                png_set_error(PNG_ERR_TRUNCATED);
                rc = -1;
                break;
            }
            if (parse_ihdr(data, dlen, &ihdr) != 0) { rc = -1; break; }
            have_ihdr = 1;
        } else if (memcmp(type, "IDAT", 4) == 0) {
            if (!have_ihdr) {
                png_set_error(PNG_ERR_TRUNCATED);
                rc = -1;
                break;
            }
            if (outbuf_reserve(&idat, dlen) != 0) { rc = -1; break; }
            memcpy(idat.data + idat.len, data, dlen);
            idat.len += dlen;
        } else if (memcmp(type, "IEND", 4) == 0) {
            seen_iend = 1;
            break;
        }
        /* Other chunks (PLTE, tEXt, etc.) are skipped silently. */
    }

    if (rc < 0 || !have_ihdr || !seen_iend || idat.len == 0) {
        if (png_last_error() == PNG_OK) png_set_error(PNG_ERR_TRUNCATED);
        free(file_buf);
        free(idat.data);
        return NULL;
    }

    /* Validate IHDR fields we support */
    if (ihdr.bit_depth != 8) {
        png_set_error(PNG_ERR_UNSUPPORTED);
        free(file_buf);
        free(idat.data);
        return NULL;
    }
    int channels;
    switch (ihdr.color_type) {
        case 0: channels = 1; break;   /* grayscale */
        case 2: channels = 3; break;   /* RGB       */
        case 6: channels = 4; break;   /* RGBA      */
        default:
            png_set_error(PNG_ERR_UNSUPPORTED);
            free(file_buf);
            free(idat.data);
            return NULL;
    }
    if (ihdr.interlace != 0) {
        png_set_error(PNG_ERR_UNSUPPORTED);
        free(file_buf);
        free(idat.data);
        return NULL;
    }
    if (ihdr.compression != 0 || ihdr.filter != 0) {
        png_set_error(PNG_ERR_UNSUPPORTED);
        free(file_buf);
        free(idat.data);
        return NULL;
    }
    if (ihdr.width == 0 || ihdr.height == 0 ||
        ihdr.width > 0x7FFFFFFFu || ihdr.height > 0x7FFFFFFFu) {
        png_set_error(PNG_ERR_UNSUPPORTED);
        free(file_buf);
        free(idat.data);
        return NULL;
    }

    /* Inflate the IDAT data */
    OutBuf raw = {0};
    if (zlib_inflate(idat.data, idat.len, &raw) != 0) {
        free(file_buf);
        free(idat.data);
        free(raw.data);
        return NULL;
    }

    /* Expected size: height * (1 + width*channels) */
    size_t stride = (size_t)ihdr.width * (size_t)channels + 1;
    size_t expected = stride * (size_t)ihdr.height;
    if (raw.len != expected) {
        png_set_error(PNG_ERR_INFLATE);
        free(file_buf);
        free(idat.data);
        free(raw.data);
        return NULL;
    }

    /* Unfilter in place */
    if (unfilter(raw.data, ihdr.width, ihdr.height, channels) != 0) {
        free(file_buf);
        free(idat.data);
        free(raw.data);
        return NULL;
    }

    /* Strip the per-row filter bytes and pack pixels tightly */
    size_t pixel_bytes = (size_t)ihdr.width * (size_t)ihdr.height *
                         (size_t)channels;
    uint8_t *pixels = (uint8_t *)malloc(pixel_bytes ? pixel_bytes : 1);
    if (!pixels) {
        png_set_error(PNG_ERR_MEMORY);
        free(file_buf);
        free(idat.data);
        free(raw.data);
        return NULL;
    }
    for (uint32_t y = 0; y < ihdr.height; y++) {
        const uint8_t *src_row = raw.data + (size_t)y * stride + 1;
        uint8_t *dst_row = pixels + (size_t)y * (size_t)ihdr.width * channels;
        memcpy(dst_row, src_row, (size_t)ihdr.width * channels);
    }

    Image *img = (Image *)malloc(sizeof(Image));
    if (!img) {
        png_set_error(PNG_ERR_MEMORY);
        free(pixels);
        free(file_buf);
        free(idat.data);
        free(raw.data);
        return NULL;
    }
    img->width = (int)ihdr.width;
    img->height = (int)ihdr.height;
    img->channels = channels;
    img->pixels = pixels;

    free(file_buf);
    free(idat.data);
    free(raw.data);
    return img;
}

void image_free(Image *img) {
    if (!img) return;
    free(img->pixels);
    free(img);
}
