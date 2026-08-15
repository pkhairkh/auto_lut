#ifndef PACK_H
#define PACK_H

#include <stdint.h>
#include <stddef.h>

/* ===========================================================================
 * pack.h — ANE-compatible packing primitives for the auto_lut engine.
 *
 * This module converts uint8 palette indices into the on-disk byte layout
 * that Apple's Neural Engine (ANE) expects when loading palettised weights.
 *
 * Three bit widths are supported, all strictly LSB-first:
 *
 *   idx4 : 4 bits per index, 2 indices per byte.
 *          byte[k] = idx[2k] | (idx[2k+1] << 4)
 *          -> low nibble  = even-positioned index
 *          -> high nibble = odd-positioned  index
 *
 *   idx6 : 6 bits per index, packed bit-stream, LSB-first.
 *          A 32-bit accumulator is filled with the next 32 indices worth of
 *          bits (32 * 6 = 192 bits = 24 bytes) and flushed in little-endian
 *          byte order. The final partial group is flushed as a whole uint32
 *          with the unused high bits being zero.
 *
 *   idx8 : 8 bits per index, raw byte stream (effectively memcpy).
 *
 * Pre-transpose convention
 * ------------------------
 * Palettised weight tensors live in memory as (out_dim, in_dim) row-major.
 * The ANE, however, wants the lookup to be (in_dim, out_dim). Instead of
 * physically transposing the index tensor before packing, callers pass the
 * indices already laid out as (in_dim, out_dim) — i.e. already transposed —
 * and the pack_* functions simply walk them row by row. The function
 * signatures therefore take `rows` and `cols` describing the transposed
 * layout, not the original weight layout.
 *
 * Return value
 * ------------
 * Each pack_* function returns the number of bytes written to `out`. The
 * caller must size `out` according to:
 *
 *   idx4 : ceil(rows * cols / 2)
 *   idx6 : ceil(rows * cols * 6 / 8)
 *   idx8 : rows * cols
 *
 * All functions assume `out` has been allocated with at least the above
 * number of bytes; no bounds checking is performed.
 * =========================================================================== */

/* Pack (rows, cols) uint8 indices into packed bytes — 4-bit LSB-first.
 *
 * Two indices are packed into each output byte:
 *   byte[k] = idx[2k] | (idx[2k+1] << 4)
 *
 * If (rows * cols) is odd the trailing nibble holds the final index in its
 * low bits and the high nibble is zero.
 *
 * Pre-transpose: indices are (in_dim, out_dim) — transposed from
 * weight (out_dim, in_dim).
 *
 * Returns: number of bytes written to `out`.
 */
size_t pack_idx4(const uint8_t *indices, int rows, int cols, uint8_t *out);

/* Pack (rows, cols) uint8 indices into packed bytes — 6-bit LSB-first.
 *
 * A 32-bit accumulator is filled with bits in LSB-first order: index 0
 * occupies bits 0..5, index 1 occupies bits 6..11, and so on. When the
 * accumulator holds a full 32 bits worth of bits (which happens after 5
 * indices, leaving 2 bits unused in the uint32, but flushing every 5
 * indices is wasteful) — instead, every 32 indices fill exactly 24 bytes
 * (32 * 6 = 192 = 24 * 8), so the accumulator is drained as a 24-byte
 * little-endian block once 32 indices have been pushed. The trailing
 * partial block is flushed as a final 24-byte little-endian block whose
 * unused high bits are zero.
 *
 * Returns: number of bytes written to `out`.
 */
size_t pack_idx6(const uint8_t *indices, int rows, int cols, uint8_t *out);

/* Pack (rows, cols) uint8 indices into packed bytes — 8-bit (raw bytes).
 *
 * Equivalent to memcpy(out, indices, rows * cols). Provided for API
 * symmetry and to centralise any future ANE-specific byte ordering.
 *
 * Returns: number of bytes written to `out`.
 */
size_t pack_idx8(const uint8_t *indices, int rows, int cols, uint8_t *out);

/* Sanitize a tensor name into a filesystem-safe filename component.
 *
 * Replaces '.' and '/' (and '\\') with '_', truncates to dstsz-1 chars and
 * NUL-terminates. If `src` is NULL or `dstsz` is 0 the function is a no-op.
 * Output is always NUL-terminated when dstsz > 0.
 *
 * Examples:
 *   "model.layers.0.weight" -> "model_layers_0_weight"
 *   "a/b\\c.d"               -> "a_b_c_d"
 */
void sanitize_name(const char *src, char *dst, size_t dstsz);

/* Write a flat (groups * palette) float LUT as an FP16 binary file.
 *
 * `lut` points to `groups * palette` float values laid out row-major as
 * lut[g * palette + p]. Each float is converted to IEEE 754 binary16
 * (FP16) with round-to-nearest, ties-to-even, and written little-endian.
 * The resulting file is exactly `groups * palette * 2` bytes long.
 *
 * The function returns void. On failure (NULL inputs, non-positive
 * dimensions, malloc failure, or file open failure) it silently does
 * nothing; callers can detect failure by stat()-ing the output file
 * or checking errno.
 */
void write_lut_fp16(const float *lut, int groups, int palette,
                    const char *path);

/* Copy a non-palettisable tensor to a file, converting to FP16 on the way.
 *
 * `data`     : pointer to the raw tensor data.
 * `n_elements`: number of scalar elements (NOT bytes).
 * `is_fp32`  : non-zero if `data` is float32, zero if already float16.
 * `path`     : output file path. The file will be `n_elements * 2` bytes.
 *
 * When is_fp32 is non-zero each element is converted to IEEE 754 binary16.
 * When is_fp32 is zero the bytes are copied verbatim (no endianness swap —
 * the source is assumed to already be little-endian FP16, which matches
 * the safetensors convention).
 */
void copy_tensor_fp16(const void *data, size_t n_elements, int is_fp32,
                       const char *path);

#endif /* PACK_H */
