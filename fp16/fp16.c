/* fp16.c — IEEE 754 binary16 conversion (pure C, no external libs).
 *
 * Bit-exact with the standard "round to nearest even" algorithm used by
 * numpy.float16 / torch.float16 / CoreML FP16 scalars. */
#include "fp16.h"
#include <string.h>

/* union for type-punning via memcpy (UB-safe) */
typedef union { uint32_t u; float f; } bits32;

float fp16_to_f32(uint16_t h) {
    /* expand half to float using bit manipulation */
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (uint32_t)((h & 0x7C00u) >> 10);
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    uint32_t f;

    if (exp == 0) {
        /* subnormal or zero */
        if (mant == 0) {
            f = sign;  /* signed zero */
        } else {
            /* normalize subnormal */
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while ((mant & 0x0400u) == 0);
            mant &= 0x03FFu;
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        /* inf or nan */
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        /* normal */
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    bits32 b;
    b.u = f;
    return b.f;
}

uint16_t f32_to_fp16(float f) {
    bits32 b;
    b.f = f;
    uint32_t x = b.u;
    uint32_t sign = (x & 0x80000000u) >> 16;
    int32_t  exp  = (int32_t)((x & 0x7F800000u) >> 23);
    uint32_t mant = x & 0x007FFFFFu;

    if (exp == 0xFF) {
        /* inf or nan */
        if (mant) {
            /* nan: preserve top bit, clear others to make it a quiet nan */
            mant = (mant & 0x007FF000u) >> 13;
            if (mant == 0) mant = 1;  /* ensure nan */
            return (uint16_t)(sign | 0x7C00u | mant);
        }
        return (uint16_t)(sign | 0x7C00u);  /* inf */
    }

    exp = exp - 127 + 15;
    /* mant is the 23-bit fractional mantissa (no implicit 1).
     * We need to reduce it to 10 bits with round-half-to-even.
     * DO NOT shift left first (that would overflow uint32 and lose bits). */

    if (exp >= 0x1F) {
        /* overflow -> inf */
        return (uint16_t)(sign | 0x7C00u);
    } else if (exp <= 0) {
        if (exp < -10) {
            /* underflow -> zero */
            return (uint16_t)sign;
        }
        /* subnormal: shift mantissa right, round to nearest even */
        mant |= 0x00800000u;  /* implicit 1 (bit 23) */
        int sh = 14 - exp;    /* shift amount to get 10-bit subnormal mantissa */
        /* round half to even */
        uint32_t lsb    = (mant >> sh) & 1;
        uint32_t round  = (mant >> (sh - 1)) & 1;
        uint32_t dropped = (mant & ((1u << (sh - 1)) - 1)) ? 1 : 0;
        mant = mant >> sh;
        if (round && (lsb || dropped)) mant++;
        if (mant & 0x0400u) {
            /* rounded up to smallest normal */
            return (uint16_t)(sign | 0x0400u | (mant & 0x03FFu));
        }
        return (uint16_t)(sign | (mant & 0x03FFu));
    } else {
        /* normal: reduce 23-bit mantissa to 10 bits with round-half-to-even.
         * bit 13 is the new LSB, bit 12 is the rounding bit, bits 11-0 are
         * the dropped tail. */
        uint32_t lsb    = (mant >> 13) & 1;
        uint32_t round  = (mant >> 12) & 1;
        uint32_t dropped = (mant & 0xFFFu) ? 1 : 0;
        mant = mant >> 13;
        if (round && (lsb || dropped)) {
            mant++;
            if (mant & 0x0400u) {
                mant = 0;
                exp++;
                if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
            }
        }
        return (uint16_t)(sign | (exp << 10) | (mant & 0x03FFu));
    }
}

void fp16_to_f32_array(const void *in, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)in;
    for (size_t i = 0; i < n; i++) {
        uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
        out[i] = fp16_to_f32(h);
        p += 2;
    }
}

void f32_to_fp16_array(const float *in, void *out, size_t n) {
    uint8_t *p = (uint8_t *)out;
    for (size_t i = 0; i < n; i++) {
        uint16_t h = f32_to_fp16(in[i]);
        p[0] = (uint8_t)(h & 0xFF);
        p[1] = (uint8_t)((h >> 8) & 0xFF);
        p += 2;
    }
}
