/* fp16.c - IEEE 754 binary16 <-> binary32 conversion (pure C11, no external libs).
 *
 * Bit-exact with the standard "round to nearest even" algorithm used by
 * numpy.float16 / torch.float16 / CoreML FP16 scalars.
 *
 * Layout:
 *   binary16: 1 sign | 5 exponent (bias 15)  | 10 mantissa
 *   binary32: 1 sign | 8 exponent (bias 127) | 23 mantissa
 */
#include "fp16.h"
#include <string.h>

/* Union for safe type-punning between float and uint32_t (no strict-aliasing UB). */
typedef union { uint32_t u; float f; } bits32_t;

/* ===========================================================================
 * fp16_to_fp32_scalar
 *
 * Expand a binary16 bit pattern into a binary32. This conversion is always
 * exact: every binary16 value has a unique binary32 representation.
 *
 *   sign:   bit 15 -> bit 31
 *   exp:    bits 10-14, bias 15  ->  bits 23-30, bias 127  (rebias +112)
 *   mant:   bits 0-9  ->  bits 13-22 (left-shift by 13)
 *
 * Special cases:
 *   - exp == 0, mant == 0  : signed zero
 *   - exp == 0, mant != 0  : subnormal -> normalize (shift until implicit-1
 *                            reaches bit 10, decrement would-be exp each step)
 *   - exp == 0x1F           : Inf (mant == 0) or NaN (mant != 0)
 * =========================================================================== */
float fp16_to_fp32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (uint32_t)((h & 0x7C00u) >> 10);
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            /* Signed zero. */
            bits = sign;
        } else {
            /* Subnormal: shift left until the implicit-1 bit (position 10) is
             * set. Each left-shift corresponds to a -1 in the would-be FP16
             * biased exponent (which starts at 1 = smallest normal). */
            uint32_t m = mant;
            int32_t  e = 1;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                e--;
            }
            m &= 0x03FFu;            /* drop the now-set implicit-1 */
            /* FP32 biased exp = e + (127 - 15) = e + 112 */
            bits = sign | ((uint32_t)(e + 112) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        /* Inf or NaN: exp -> 0xFF, mantissa payload preserved (shifted). */
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        /* Normal: rebias 15 -> 127 (+112), shift mantissa 10 -> 23 (+13). */
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    bits32_t b;
    b.u = bits;
    return b.f;
}

/* ===========================================================================
 * fp32_to_fp16_scalar
 *
 * Round a binary32 to the nearest binary16 using round-to-nearest-even
 * (the default IEEE 754 rounding mode). Ties are broken to the value with
 * an even LSB of the resulting mantissa.
 *
 * Overflow: any finite |x| larger than 65504.0 (the largest finite binary16)
 * saturates to +/-Inf. Underflow: any |x| smaller than 0.5 * 2^-24 (half
 * of the smallest subnormal) rounds to +/-0.
 *
 * NaN handling: the top 10 bits of the FP32 payload are propagated into the
 * FP16 mantissa; if all those bits are zero, a sentinel payload of 1 is set
 * to guarantee the result is still a NaN (not Inf).
 * =========================================================================== */
uint16_t fp32_to_fp16_scalar(float f) {
    bits32_t b;
    b.f = f;
    uint32_t x     = b.u;
    uint32_t sign  = (x & 0x80000000u) >> 16;
    int32_t  exp32 = (int32_t)((x & 0x7F800000u) >> 23);   /* FP32 biased exp */
    uint32_t mant  = x & 0x007FFFFFu;

    /* ---- Inf / NaN ---- */
    if (exp32 == 0xFFu) {
        if (mant == 0u) {
            return (uint16_t)(sign | 0x7C00u);             /* Inf */
        }
        /* NaN: drop top 10 bits of payload; force non-zero to stay NaN. */
        uint32_t m = mant >> 13;
        if (m == 0u) m = 1u;
        return (uint16_t)(sign | 0x7C00u | (m & 0x03FFu));
    }

    /* ---- FP32 zero or subnormal -> always FP16 zero (sign preserved) ---- */
    if (exp32 == 0u) {
        return (uint16_t)sign;
    }

    /* ---- Compute FP16 biased exponent ---- */
    int32_t e16 = exp32 - 127 + 15;

    /* ---- Overflow: |x| > 65504 -> +/-Inf ---- */
    if (e16 >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }

    /* ---- Underflow to zero: |x| < 0.5 * 2^-24 ---- */
    if (e16 < -10) {
        return (uint16_t)sign;
    }

    /* ---- Subnormal result range (e16 in [-10, 0]) ----
     *
     * FP32 significand = (1.mant) * 2^(exp32-127), 24 bits with implicit 1.
     * FP16 subnormal value = m * 2^-24,  m in [0, 1023].
     * Setting them equal: m = (1.mant) << 1 >> (e16_offset).
     * Concretely: take 24-bit significand `m_ext`, right-shift by
     * `shift = 14 - e16` (in [14, 24] for e16 in [-10, 0]).
     */
    if (e16 <= 0) {
        uint32_t m_ext = mant | 0x00800000u;                 /* add implicit 1 */
        int32_t  shift = 14 - e16;                            /* [14, 24] */
        uint32_t round_bit = (m_ext >> (shift - 1)) & 1u;
        uint32_t sticky_mask = (1u << (shift - 1)) - 1u;
        uint32_t sticky = (m_ext & sticky_mask) ? 1u : 0u;
        uint32_t result = m_ext >> shift;
        if (round_bit && (sticky || (result & 1u))) {
            result += 1u;
        }
        /* result may carry into the normal range (0x400 = smallest normal),
         * which is exactly the right encoding: bit 10 set = exp=1, mant=0. */
        return (uint16_t)(sign | (result & 0x07FFu));
    }

    /* ---- Normal result range (e16 in [1, 30]) ----
     *
     * Drop 13 bits of mantissa (23 -> 10) with round-to-nearest-even.
     */
    uint32_t shift = 13u;
    uint32_t round_bit = (mant >> (shift - 1)) & 1u;
    uint32_t sticky_mask = (1u << (shift - 1)) - 1u;
    uint32_t sticky = (mant & sticky_mask) ? 1u : 0u;
    uint32_t result_m = mant >> shift;
    if (round_bit && (sticky || (result_m & 1u))) {
        result_m += 1u;
    }
    /* Mantissa can overflow into bit 10 (0x400): carry into exponent. */
    if (result_m & 0x0400u) {
        result_m = 0u;
        e16 += 1;
        if (e16 >= 31) {
            return (uint16_t)(sign | 0x7C00u);               /* carry -> Inf */
        }
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (result_m & 0x03FFu));
}
