#ifndef FP16_H
#define FP16_H

#include <stdint.h>
#include <stddef.h>

/* ===========================================================================
 * fp16.h — IEEE 754 binary16 (half precision) conversion utilities.
 *
 * Used by the auto_lut engine to read FP16 weights from safetensors files
 * and to write FP16 LUT / non-palettizable tensor outputs.
 * =========================================================================== */

/* Convert one IEEE 754 binary16 (little-endian uint16) to float32. */
float fp16_to_f32(uint16_t h);

/* Convert one float32 to IEEE 754 binary16 (returns uint16). */
uint16_t f32_to_fp16(float f);

/* Convert an array of `n` FP16 values (little-endian uint16) to float32.
 * `out` must hold `n` floats. `in` and `out` must not overlap. */
void fp16_to_f32_array(const void *in, float *out, size_t n);

/* Convert an array of `n` float32 values to FP16 (little-endian uint16).
 * `out` must hold `n * 2` bytes. `in` and `out` must not overlap. */
void f32_to_fp16_array(const float *in, void *out, size_t n);

#endif /* FP16_H */
