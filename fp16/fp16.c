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
