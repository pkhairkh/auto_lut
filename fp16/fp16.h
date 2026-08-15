#ifndef FP16_H
#define FP16_H

#include <stdint.h>
#include <stddef.h>

/* ===========================================================================
 * fp16.h - IEEE 754 binary16 (half precision) <-> binary32 (single) conversion.
 *
 * Pure C11, no external libraries. Used by the auto_lut engine to read FP16
 * weights from safetensors files and to write FP16 LUT / non-palettizable
 * tensor outputs.
 *
 * Bit layout (IEEE 754):
 *   binary16: 1 sign | 5 exponent (bias 15) | 10 mantissa
 *   binary32: 1 sign | 8 exponent (bias 127) | 23 mantissa
 *
 * Conversion semantics:
 *   - fp16_to_fp32: always exact (every binary16 has a unique binary32 value).
 *   - fp32_to_fp16: round-to-nearest-even, ties-to-even; overflow saturates
 *     to +/-Inf; NaN payload best-effort preserved (quiet NaN ensured).
 * =========================================================================== */

/* Convert one IEEE 754 binary16 (uint16 bit pattern) to float32. */
float fp16_to_fp32_scalar(uint16_t h);

/* Convert an array of `n` binary16 values to float32.
 * `dst` must hold `n` floats; `src` and `dst` must not overlap. */
void fp16_to_fp32_array(const uint16_t *src, float *dst, size_t n);

/* Convert one float32 to IEEE 754 binary16 (returns the uint16 bit pattern). */
uint16_t fp32_to_fp16_scalar(float f);

/* Convert an array of `n` float32 values to binary16.
 * `dst` must hold `n` uint16 values; `src` and `dst` must not overlap. */
void fp32_to_fp16_array(const float *src, uint16_t *dst, size_t n);

#endif /* FP16_H */
