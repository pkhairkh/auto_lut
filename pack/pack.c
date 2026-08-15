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
