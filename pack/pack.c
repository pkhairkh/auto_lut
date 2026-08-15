/* pack.c — ANE-compatible packing primitives for the auto_lut engine.
 *
 * Pure C11. No external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c pack.c -o pack.o
 *
 * See pack.h for the high-level contract. */

#include "pack.h"

#include <stdio.h>
#include <stdlib.h>
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

/* -------------------------------------------------------------------------
 * sanitize_name — turn a tensor name into a filename component.
 *
 * The safetensors format allows tensor names to contain dots and
 * (rarely) slashes, neither of which are safe in a path component on
 * any host filesystem. We rewrite both '.' and '/' (plus '\\' for
 * Windows-style separators) to '_', then truncate to dstsz-1 chars and
 * NUL-terminate. Other characters (including spaces, dashes, brackets)
 * are left untouched — callers that need stricter sanitisation can
 * post-process.
 *
 * The function is safe to call with a NULL source or a zero-sized
 * destination; in either case it becomes a no-op.
 * ------------------------------------------------------------------------- */
void sanitize_name(const char *src, char *dst, size_t dstsz)
{
    if (dst == NULL || dstsz == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }

    size_t i = 0;
    for (; i + 1 < dstsz; i++) {
        char c = src[i];
        if (c == '\0') {
            break;
        }
        if (c == '.' || c == '/' || c == '\\') {
            c = '_';
        }
        dst[i] = c;
    }
    dst[i] = '\0';
}

/* -------------------------------------------------------------------------
 * pack_f32_to_f16 — IEEE 754 binary32 -> binary16 with round-to-nearest,
 * ties-to-even.
 *
 * Used by write_lut_fp16 and copy_tensor_fp16. Implemented as a pure
 * bit-manipulation on the float's IEEE 754 representation, so it has no
 * dependency on a hardware FP16 conversion instruction (portable to any
 * platform with a 32-bit IEEE 754 float).
 *
 * Behaviour:
 *   - Zero (signed)            -> signed zero
 *   - Float32 denormal         -> signed zero (too small for FP16 denormal
 *                                 range, would round to zero anyway)
 *   - Float32 normal in range  -> FP16 normal, with correct rounding
 *   - Float32 normal too small -> FP16 denormal, with correct rounding
 *   - Float32 normal too large -> FP16 +/- Inf
 *   - Float32 +/- Inf          -> FP16 +/- Inf
 *   - Float32 NaN              -> FP16 NaN (high bits of mantissa kept;
 *                                 if those happen to be zero, force a 1
 *                                 so the result is still a NaN)
 * ------------------------------------------------------------------------- */
static uint16_t pack_f32_to_f16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));

    uint32_t sign = (u >> 31) & 0x1u;
    uint32_t exp  = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x007FFFFFu;

    uint16_t h_sign = (uint16_t)(sign << 15);

    /* +/- Inf or NaN. */
    if (exp == 0xFFu) {
        if (mant == 0u) {
            return (uint16_t)(h_sign | 0x7C00u); /* Inf */
        }
        /* NaN: keep the top 10 bits of mantissa; if those are zero,
         * set the LSB so the value is still a NaN. */
        uint16_t h_mant = (uint16_t)(mant >> 13);
        if (h_mant == 0u) {
            h_mant = 1u;
        }
        return (uint16_t)(h_sign | 0x7C00u | h_mant);
    }

    /* Float32 zero (signed) or denormal — denormals are too small to
     * represent even as an FP16 denormal and round to signed zero. */
    if (exp == 0u) {
        return h_sign;
    }

    /* Float32 normal: rebias the exponent from 127 to 15. */
    int new_exp = (int)exp - 127 + 15;

    /* Overflow to FP16 Inf. */
    if (new_exp >= 0x1F) {
        return (uint16_t)(h_sign | 0x7C00u);
    }

    /* Underflow into FP16 denormal range. */
    if (new_exp <= 0) {
        /* Too small even for a denormal — rounds to zero. */
        if (new_exp < -10) {
            return h_sign;
        }
        /* Add the implicit leading 1 of the float32 mantissa. */
        uint32_t m = mant | 0x00800000u;
        int shift = 14 - new_exp;          /* 1..24 */
        if (shift > 24) {
            return h_sign;
        }
        uint32_t new_mant = m >> shift;
        /* Round half-to-even: look at the bits we just shifted out. */
        uint32_t half_bit = (m >> (shift - 1)) & 1u;
        uint32_t sticky    = (m & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
        if (half_bit && (sticky || (new_mant & 1u))) {
            new_mant += 1u;
        }
        /* If rounding overflowed the 10-bit mantissa, it correctly
         * produces the smallest normal FP16 (exponent = 1). */
        return (uint16_t)(h_sign | (new_mant & 0x03FFu));
    }

    /* Normalised FP16. */
    uint32_t new_mant = mant >> 13;
    uint32_t half_bit = (mant >> 12) & 1u;
    uint32_t sticky    = (mant & 0x00000FFFu) ? 1u : 0u;
    uint16_t result =
        (uint16_t)(h_sign | (uint16_t)((new_exp << 10)) | (uint16_t)new_mant);
    /* Round half-to-even. Adding 1 here is safe: if new_mant was 0x3FF,
     * rounding up produces 0x400 which is exactly exp=1, mant=0 — the
     * smallest normal FP16. */
    if (half_bit && (sticky || (new_mant & 1u))) {
        result = (uint16_t)(result + 1);
    }
    return result;
}

/* -------------------------------------------------------------------------
 * write_lut_fp16 — write a flat (groups * palette) float LUT as FP16.
 *
 * Each float is converted to IEEE 754 binary16 via pack_f32_to_f16 and
 * written little-endian. On a little-endian host (x86-64) we just write
 * the uint16_t buffer verbatim, which already produces little-endian
 * bytes. The output file is exactly `groups * palette * 2` bytes.
 *
 * The function returns void; on failure (NULL inputs, non-positive
 * dimensions, malloc failure, or file open failure) it silently does
 * nothing. Callers can detect failure by stat()-ing the output file or
 * by checking errno after the call.
 * ------------------------------------------------------------------------- */
void write_lut_fp16(const float *lut, int groups, int palette,
                    const char *path)
{
    if (lut == NULL || path == NULL || groups <= 0 || palette <= 0) {
        return;
    }

    size_t n = (size_t)groups * (size_t)palette;
    uint16_t *buf = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (buf == NULL) {
        return;
    }

    for (size_t i = 0; i < n; i++) {
        buf[i] = pack_f32_to_f16(lut[i]);
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(buf);
        return;
    }

    /* Host is x86-64 (little-endian); writing the uint16_t array raw
     * produces the correct little-endian FP16 bytes on disk. */
    size_t written = fwrite(buf, sizeof(uint16_t), n, f);
    (void)written;

    fclose(f);
    free(buf);
}
