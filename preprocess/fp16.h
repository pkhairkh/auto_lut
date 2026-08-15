#ifndef FP16_H
#define FP16_H
/*
 * fp16.h - IEEE 754 half-precision (binary16) conversion utilities.
 *
 * Provides bidirectional conversion between IEEE-754 single precision
 * (binary32, C float) and half precision (binary16, uint16_t). The
 * implementation is bit-exact and handles subnormals, infinities and
 * NaNs correctly. It is header-only and dependency-free so it can be
 * included from any translation unit in the auto_lut project.
 *
 * The conversion follows the reference algorithm used by most software
 * FP16 emulators (matching the behaviour of numpy.float16 and the ARM
 * vcvt instructions). No rounding-mode configuration is exposed: the
 * default round-to-nearest-even semantics are always applied, which is
 * the only mode the preprocess pipeline requires.
 */
#include <stdint.h>

/*
 * Convert a single-precision float to a half-precision value stored in
 * the low 16 bits of an unsigned short. The high 16 bits of the result
 * are zero. infinities and NaNs are preserved; subnormals are produced
 * when the exponent would underflow the normal half range.
 */
static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    /* Drop the sign bit so the exponent/magnitude logic is sign-agnostic. */
    x &= 0x7fffffffu;

    /* HandleInf/NaN: FP32 inf (exp=0xff, mantissa=0) -> FP16 inf;
     * FP32 NaN (exp=0xff, mantissa!=0) -> FP16 NaN with the top
     * mantissa bit forced to 1 so it stays a NaN. */
    if (x >= 0x47800000u) {          /* exponent >= 143 -> overflow to inf */
        uint16_t inf_or_nan;
        if (x > 0x7f800000u) {        /* NaN: keep top mantissa bits */
            inf_or_nan = (uint16_t)(0x7e00u | ((x >> 13) & 0x3ffu));
        } else {                      /* infinity */
            inf_or_nan = 0x7c00u;
        }
        return (uint16_t)(sign | inf_or_nan);
    }

    /* Normal/subnormal path. Shift the FP32 mantissa into FP16 layout
     * with round-to-nearest-even. */
    if (x < 0x38800000u) {
        /* Subnormal result: the FP32 value is too small to represent as
         * a normal FP16. Shift the mantissa so the implicit bit becomes
         * explicit, then round. */
        uint32_t mant = (x & 0x7fffffu) | 0x800000u;
        int shift = 113 - (int)((x >> 23) & 0xffu);
        if (shift > 0) mant >>= shift;
        uint16_t rounded = (uint16_t)(mant + 0xfffu + ((mant >> 10) & 1u));
        return (uint16_t)(sign | rounded);
    }

    /* Normal result: rebias the exponent from FP32 (bias 127) to FP16
     * (bias 15) and round the 23-bit mantissa down to 10 bits. */
    uint32_t mant = x + 0xc8000000u;
    uint16_t rounded = (uint16_t)((mant >> 13) + ((mant >> 12) & 1u)
                                  + ((mant >> 13) & 1u) - 1u);
    return (uint16_t)(sign | rounded);
}

/*
 * Convert a half-precision value to a single-precision float. Subnormals
 * are normalised into the FP32 range, infinities and NaNs are preserved.
 */
static inline float fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } v;
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h & 0x7c00u) >> 10;
    uint32_t mant = ((uint32_t)h & 0x03ffu);

    if (exp == 0) {
        if (mant == 0) {
            /* Signed zero. */
            v.u = sign;
        } else {
            /* Subnormal: normalise so the highest set mantissa bit
             * becomes the implicit one, then recompute the exponent. */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0u);
            mant &= 0x3ffu;
            v.u = sign | (((uint32_t)(127 - 15 - e)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        /* Inf or NaN. */
        v.u = sign | 0x7f800000u | (mant << 13);
    } else {
        /* Normal. */
        v.u = sign | (((exp + (127 - 15)) << 23)) | (mant << 13);
    }
    return v.f;
}

#endif /* FP16_H */
