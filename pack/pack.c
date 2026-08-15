/* pack.c — ANE-compatible packing primitives for the auto_lut engine.
 *
 * Pure C11. No external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c pack.c -o pack.o
 *
 * See pack.h for the high-level contract. */

#include "pack.h"

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * pack_idx4 — 4-bit LSB-first nibble packing.
 *
 * Two indices per byte. The even-positioned index goes in the low nibble,
 * the odd-positioned index in the high nibble:
 *
 *   byte[k] = idx[2k] | (idx[2k+1] << 4)
 *
 * When (rows * cols) is odd the final byte holds only one index in its low
 * nibble; the high nibble is zero. The low 4 bits of every input index are
 * used; the high 4 bits are masked off so a stray 0xFF doesn't bleed into
 * the neighbouring nibble.
 * ------------------------------------------------------------------------- */
size_t pack_idx4(const uint8_t *indices, int rows, int cols, uint8_t *out)
{
    size_t n = (size_t)rows * (size_t)cols;
    size_t k = 0;
    size_t i = 0;

    /* Bulk: pack pairs of indices. */
    size_t pairs = n >> 1;
    while (i + 2 <= n) {
        uint8_t lo = indices[i]     & 0x0F;
        uint8_t hi = indices[i + 1] & 0x0F;
        out[k++] = (uint8_t)(lo | (hi << 4));
        i += 2;
    }
    (void)pairs; /* silence unused warning when loops get unrolled */

    /* Trailing odd index, if any. */
    if (i < n) {
        out[k++] = (uint8_t)(indices[i] & 0x0F);
    }

    return k;
}

/* -------------------------------------------------------------------------
 * pack_idx6 — 6-bit LSB-first bit packing with a uint32_t accumulator.
 *
 * Each input index contributes exactly 6 bits to the output stream. Bits
 * are emitted LSB-first: bit 0 of index 0 becomes bit 0 of byte 0, bit 1
 * of index 0 becomes bit 1 of byte 0, and so on. The accumulator is a
 * uint32_t; whenever 8 or more bits have accumulated a full byte is
 * drained from the low end. Because the accumulator never holds more
 * than 13 bits (8 + 6 worst case after the previous drain), there is no
 * risk of shifting a 6-bit value past the top of a uint32_t.
 *
 * Output size: ceil(rows*cols * 6 / 8) bytes.
 *
 * Roundtrip property: packing then unpacking (reading 6 bits at a time
 * LSB-first) reproduces every input index modulo 0x3F exactly. Inputs
 * that already fit in 6 bits (0..63) round-trip verbatim.
 * ------------------------------------------------------------------------- */
size_t pack_idx6(const uint8_t *indices, int rows, int cols, uint8_t *out)
{
    size_t n = (size_t)rows * (size_t)cols;
    uint32_t acc = 0;
    int bits = 0;
    size_t k = 0;

    for (size_t i = 0; i < n; i++) {
        uint32_t idx = (uint32_t)(indices[i] & 0x3F); /* keep low 6 bits */
        acc |= (idx << bits);
        bits += 6;

        /* Drain every full byte. After this loop bits ∈ [0, 7]. */
        while (bits >= 8) {
            out[k++] = (uint8_t)(acc & 0xFFu);
            acc >>= 8;
            bits -= 8;
        }
    }

    /* Flush the remaining partial byte (bits ∈ [1, 7]). The high bits of
     * the emitted byte are zero. */
    if (bits > 0) {
        out[k++] = (uint8_t)(acc & 0xFFu);
    }

    return k;
}

/* -------------------------------------------------------------------------
 * pack_idx8 — 8-bit packing (raw bytes).
 *
 * No bit manipulation is required: each index already occupies a full
 * byte. The implementation is a straight memcpy, but we keep it as a
 * real function rather than a macro so the public API is symmetric and
 * so that any future ANE-specific byte ordering (e.g. byte swapping on
 * big-endian hosts, although we only target x86-64) can be centralised
 * here.
 *
 * Output size: rows * cols bytes.
 * ------------------------------------------------------------------------- */
size_t pack_idx8(const uint8_t *indices, int rows, int cols, uint8_t *out)
{
    size_t n = (size_t)rows * (size_t)cols;
    if (n > 0) {
        memcpy(out, indices, n);
    }
    return n;
}
