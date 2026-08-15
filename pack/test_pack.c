/* test_pack.c — roundtrip & edge-case tests for the pack module.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra \
 *       test_pack.c pack.c -o test_pack
 *
 * Run:
 *   ./test_pack
 *
 * Exit code: 0 if all tests passed, 1 otherwise. The number of tests
 * run and the number of failures are also printed at the end.
 */

#include "pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Tiny test framework                                                       */
/* ------------------------------------------------------------------------- */

static int g_tests    = 0;
static int g_failures = 0;

/* CHECK(cond, fmt, ...) — counts one test, records a failure with a
 * printf-style message if cond is false. */
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        g_tests++;                                                          \
        if (!(cond)) {                                                     \
            g_failures++;                                                  \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                   \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
        }                                                                   \
    } while (0)

/* ------------------------------------------------------------------------- */
/* Unpack helpers — only used by the tests, mirroring the pack logic so we    */
/* can verify roundtrip behaviour.                                           */
/* ------------------------------------------------------------------------- */

/* Unpack 4-bit indices: byte[k] -> idx[2k] (low), idx[2k+1] (high). */
static void unpack_idx4(const uint8_t *packed, size_t n_bytes, uint8_t *out)
{
    for (size_t k = 0; k < n_bytes; k++) {
        out[2 * k]     = (uint8_t)(packed[k] & 0x0F);
        out[2 * k + 1] = (uint8_t)((packed[k] >> 4) & 0x0F);
    }
}

/* Unpack 6-bit indices: read 6 bits at a time, LSB-first.
 * Reads exactly ceil(n_indices*6/8) bytes from `packed`. */
static void unpack_idx6(const uint8_t *packed, size_t n_indices, uint8_t *out)
{
    uint32_t acc  = 0;
    int     bits = 0;
    size_t  pi   = 0;
    for (size_t i = 0; i < n_indices; i++) {
        while (bits < 6) {
            acc  |= ((uint32_t)packed[pi++]) << bits;
            bits += 8;
        }
        out[i] = (uint8_t)(acc & 0x3Fu);
        acc  >>= 6;
        bits  -= 6;
    }
}

/* ------------------------------------------------------------------------- */
/* Known-pattern tests                                                       */
/* ------------------------------------------------------------------------- */

/* pack_idx4: input [0,1,2,3,4,5,6,7] must produce [0x10,0x32,0x54,0x76]. */
static void test_pack_idx4_known(void)
{
    uint8_t in[8]   = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t out[4];
    uint8_t back[8];

    size_t n = pack_idx4(in, 1, 8, out);
    CHECK(n == 4, "pack_idx4 wrote %zu bytes, expected 4", n);
    CHECK(out[0] == 0x10, "out[0]=0x%02X, expected 0x10", out[0]);
    CHECK(out[1] == 0x32, "out[1]=0x%02X, expected 0x32", out[1]);
    CHECK(out[2] == 0x54, "out[2]=0x%02X, expected 0x54", out[2]);
    CHECK(out[3] == 0x76, "out[3]=0x%02X, expected 0x76", out[3]);

    unpack_idx4(out, n, back);
    CHECK(memcmp(in, back, 8) == 0, "idx4 roundtrip mismatch");
}

/* pack_idx6: input [0,1,2,3,4,5,6,7] must produce
 * [0x40,0x20,0x0C,0x44,0x61,0x1C]. */
static void test_pack_idx6_known(void)
{
    uint8_t in[8]   = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t out[6];
    uint8_t back[8];

    size_t n = pack_idx6(in, 1, 8, out);
    CHECK(n == 6, "pack_idx6 wrote %zu bytes, expected 6", n);

    uint8_t expected[6] = {0x40, 0x20, 0x0C, 0x44, 0x61, 0x1C};
    for (int i = 0; i < 6; i++) {
        CHECK(out[i] == expected[i],
              "out[%d]=0x%02X, expected 0x%02X",
              i, out[i], expected[i]);
    }

    unpack_idx6(out, 8, back);
    CHECK(memcmp(in, back, 8) == 0, "idx6 roundtrip mismatch");
}

/* pack_idx8: input is identity-copied. */
static void test_pack_idx8_known(void)
{
    uint8_t in[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t out[8];

    size_t n = pack_idx8(in, 1, 8, out);
    CHECK(n == 8, "pack_idx8 wrote %zu bytes, expected 8", n);
    CHECK(memcmp(in, out, 8) == 0, "idx8 roundtrip mismatch");
}

/* ------------------------------------------------------------------------- */
/* Edge-case tests                                                          */
/* ------------------------------------------------------------------------- */

/* pack_idx4: high bits must be masked off; odd count yields a final byte   */
/* with only the low nibble populated.                                      */
static void test_pack_idx4_edge(void)
{
    /* High bits set: 0xFF input becomes 0xFF output (both nibbles 0xF). */
    uint8_t in_hi[2] = {0xFF, 0xFF};
    uint8_t out[4];
    size_t  n = pack_idx4(in_hi, 1, 2, out);
    CHECK(n == 1, "wrote %zu, expected 1", n);
    CHECK(out[0] == 0xFF,
          "out=0x%02X, expected 0xFF (both nibbles 0xF after masking)",
          out[0]);

    /* Odd count: trailing byte holds only the last index. */
    uint8_t in_odd[3] = {0x0A, 0x0B, 0x0C};
    n = pack_idx4(in_odd, 1, 3, out);
    CHECK(n == 2, "wrote %zu, expected 2", n);
    CHECK(out[0] == 0xBA,
          "out[0]=0x%02X, expected 0xBA (0x0A | 0x0B<<4)",
          out[0]);
    CHECK(out[1] == 0x0C,
          "out[1]=0x%02X, expected 0x0C (lone trailing index)",
          out[1]);
}

/* pack_idx6: high bits must be masked off (input values > 0x3F should    */
/* contribute only their low 6 bits).                                      */
static void test_pack_idx6_edge(void)
{
    /* 0xFF input should be masked to 0x3F. */
    uint8_t in[1] = {0xFF};
    uint8_t out[1];
    size_t  n = pack_idx6(in, 1, 1, out);
    CHECK(n == 1, "wrote %zu, expected 1", n);
    /* 0x3F in binary is 00111111, packed LSB-first into bit 0 of byte 0. */
    CHECK(out[0] == 0x3F,
          "out=0x%02X, expected 0x3F (low 6 bits of 0xFF)",
          out[0]);

    /* Two 0xFF inputs: bits 0..5 = 0x3F, bits 6..11 = 0x3F.
     * byte 0 = bits 0..7  = 0x3F | (0x3F << 6) & 0xFF = 0x3F | 0xC0 = 0xFF
     * byte 1 = bits 8..11 = 0x3F >> 2 = 0x0F (with high 4 bits zero). */
    uint8_t in2[2] = {0xFF, 0xFF};
    n = pack_idx6(in2, 1, 2, out);
    CHECK(n == 2, "wrote %zu, expected 2", n);
    CHECK(out[0] == 0xFF,
          "out[0]=0x%02X, expected 0xFF (6+2 bits of two 0x3F values)",
          out[0]);
    CHECK(out[1] == 0x0F,
          "out[1]=0x%02X, expected 0x0F (top 4 bits of second 0x3F)",
          out[1]);
}

/* pack_idx8: empty input must produce zero bytes and not crash. */
static void test_pack_idx8_edge(void)
{
    uint8_t out[1] = {0xAB};
    size_t  n = pack_idx8(out, 0, 0, out);
    CHECK(n == 0, "wrote %zu, expected 0", n);
}

/* ------------------------------------------------------------------------- */
/* Randomised roundtrip tests                                               */
/* ------------------------------------------------------------------------- */

/* Deterministic LCG so test runs are reproducible. */
static uint32_t lcg_next(uint32_t *state)
{
    *state = *state * 1103515245u + 12345u;
    return *state;
}

static void test_pack_idx4_random(void)
{
    uint32_t state = 12345;
    for (int trial = 0; trial < 200; trial++) {
        int rows = 1 + (lcg_next(&state) % 32);
        int cols = 1 + (lcg_next(&state) % 32);
        int n    = rows * cols;

        uint8_t *in   = (uint8_t *)malloc((size_t)n);
        uint8_t *back = (uint8_t *)malloc((size_t)n);
        size_t   obytes = ((size_t)n + 1u) / 2u;
        uint8_t *out  = (uint8_t *)malloc(obytes);
        CHECK(in != NULL && back != NULL && out != NULL,
              "alloc failure trial=%d n=%d", trial, n);

        for (int i = 0; i < n; i++) {
            in[i] = (uint8_t)(lcg_next(&state) & 0x0F);
        }

        size_t written = pack_idx4(in, rows, cols, out);
        CHECK(written == obytes,
              "trial %d: wrote %zu, expected %zu",
              trial, written, obytes);

        unpack_idx4(out, written, back);
        CHECK(memcmp(in, back, (size_t)n) == 0,
              "trial %d: idx4 roundtrip mismatch (n=%d)", trial, n);

        free(in);
        free(back);
        free(out);
    }
}

static void test_pack_idx6_random(void)
{
    uint32_t state = 98765;
    for (int trial = 0; trial < 200; trial++) {
        int rows = 1 + (lcg_next(&state) % 32);
        int cols = 1 + (lcg_next(&state) % 32);
        int n    = rows * cols;

        uint8_t *in   = (uint8_t *)malloc((size_t)n);
        uint8_t *back = (uint8_t *)malloc((size_t)n);
        size_t   obytes = ((size_t)n * 6u + 7u) / 8u;
        uint8_t *out  = (uint8_t *)malloc(obytes);
        CHECK(in != NULL && back != NULL && out != NULL,
              "alloc failure trial=%d n=%d", trial, n);

        for (int i = 0; i < n; i++) {
            in[i] = (uint8_t)(lcg_next(&state) & 0x3F);
        }

        size_t written = pack_idx6(in, rows, cols, out);
        CHECK(written == obytes,
              "trial %d: wrote %zu, expected %zu",
              trial, written, obytes);

        unpack_idx6(out, (size_t)n, back);
        CHECK(memcmp(in, back, (size_t)n) == 0,
              "trial %d: idx6 roundtrip mismatch (n=%d)", trial, n);

        free(in);
        free(back);
        free(out);
    }
}

static void test_pack_idx8_random(void)
{
    uint32_t state = 11111;
    for (int trial = 0; trial < 200; trial++) {
        int rows = 1 + (lcg_next(&state) % 32);
        int cols = 1 + (lcg_next(&state) % 32);
        int n    = rows * cols;

        uint8_t *in   = (uint8_t *)malloc((size_t)n);
        uint8_t *back = (uint8_t *)malloc((size_t)n);
        uint8_t *out  = (uint8_t *)malloc((size_t)n);
        CHECK(in != NULL && back != NULL && out != NULL,
              "alloc failure trial=%d n=%d", trial, n);

        for (int i = 0; i < n; i++) {
            in[i] = (uint8_t)(lcg_next(&state) & 0xFF);
        }

        size_t written = pack_idx8(in, rows, cols, out);
        CHECK(written == (size_t)n,
              "trial %d: wrote %zu, expected %d", trial, written, n);

        memcpy(back, out, (size_t)n);
        CHECK(memcmp(in, back, (size_t)n) == 0,
              "trial %d: idx8 roundtrip mismatch", trial);

        free(in);
        free(back);
        free(out);
    }
}

/* ------------------------------------------------------------------------- */
/* sanitize_name tests                                                      */
/* ------------------------------------------------------------------------- */

static void test_sanitize_name(void)
{
    char buf[128];

    /* Dots -> underscores. */
    sanitize_name("model.layers.0.weight", buf, sizeof(buf));
    CHECK(strcmp(buf, "model_layers_0_weight") == 0,
          "got '%s', expected 'model_layers_0_weight'", buf);

    /* Slashes and backslashes -> underscores. */
    /* In C source, "\\\\" is one backslash. */
    sanitize_name("a/b\\\\c.d", buf, sizeof(buf));
    CHECK(strcmp(buf, "a_b_c_d") == 0,
          "got '%s', expected 'a_b_c_d'", buf);

    /* No-op when no special chars. */
    sanitize_name("nochange", buf, sizeof(buf));
    CHECK(strcmp(buf, "nochange") == 0,
          "got '%s', expected 'nochange'", buf);

    /* Empty string. */
    sanitize_name("", buf, sizeof(buf));
    CHECK(strcmp(buf, "") == 0,
          "got '%s', expected ''", buf);

    /* Truncation: dstsz-1 chars are written, then NUL. */
    sanitize_name("abcdefghijklmnopqrstuvwxyz", buf, 10);
    CHECK(strlen(buf) == 9,
          "len=%zu, expected 9", strlen(buf));
    CHECK(strcmp(buf, "abcdefghi") == 0,
          "got '%s', expected 'abcdefghi'", buf);

    /* dstsz = 1 means only a NUL byte can be written. */
    sanitize_name("hello", buf, 1);
    CHECK(strlen(buf) == 0,
          "len=%zu, expected 0", strlen(buf));

    /* NULL src -> empty string. */
    sanitize_name(NULL, buf, sizeof(buf));
    CHECK(strcmp(buf, "") == 0,
          "got '%s', expected ''", buf);

    /* dstsz = 0 -> no-op (buf unchanged). We set buf to a sentinel and
     * verify it is not touched. */
    memset(buf, 0x7E, sizeof(buf));
    sanitize_name("whatever", buf, 0);
    CHECK((unsigned char)buf[0] == 0x7E,
          "buf[0]=0x%02X, expected unchanged 0x7E", (unsigned char)buf[0]);
}

/* ------------------------------------------------------------------------- */
/* write_lut_fp16 tests                                                     */
/* ------------------------------------------------------------------------- */

static void test_write_lut_fp16(void)
{
    /* FP16 reference values (little-endian on x86-64):
     *   1.0  -> 0x3C00
     *   0.5  -> 0x3800
     *   2.0  -> 0x4000
     *  -1.0  -> 0xBC00
     *   0.0  -> 0x0000
     *  -0.0  -> 0x8000
     */
    float    lut[6] = {1.0f, 0.5f, 2.0f, -1.0f, 0.0f, -0.0f};
    int      groups  = 2;
    int      palette = 3;
    const char *path = "/tmp/test_pack_lut_fp16.bin";

    write_lut_fp16(lut, groups, palette, path);

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "could not open %s", path);

    uint16_t buf[6];
    size_t   r = fread(buf, sizeof(uint16_t), 6, f);
    fclose(f);
    CHECK(r == 6, "read %zu, expected 6", r);

    CHECK(buf[0] == 0x3C00, "buf[0]=0x%04X, expected 0x3C00 (1.0)",  buf[0]);
    CHECK(buf[1] == 0x3800, "buf[1]=0x%04X, expected 0x3800 (0.5)",  buf[1]);
    CHECK(buf[2] == 0x4000, "buf[2]=0x%04X, expected 0x4000 (2.0)",  buf[2]);
    CHECK(buf[3] == 0xBC00, "buf[3]=0x%04X, expected 0xBC00 (-1.0)", buf[3]);
    CHECK(buf[4] == 0x0000, "buf[4]=0x%04X, expected 0x0000 (0.0)",  buf[4]);
    CHECK(buf[5] == 0x8000, "buf[5]=0x%04X, expected 0x8000 (-0.0)", buf[5]);

    /* Verify file size = groups * palette * 2 = 12 bytes. */
    FILE *f2 = fopen(path, "rb");
    fseek(f2, 0, SEEK_END);
    long sz = ftell(f2);
    fclose(f2);
    CHECK(sz == 12, "file size=%ld, expected 12", sz);

    remove(path);
}

/* ------------------------------------------------------------------------- */
/* copy_tensor_fp16 tests                                                  */
/* ------------------------------------------------------------------------- */

static void test_copy_tensor_fp16_fp32(void)
{
    /* FP32 input -> FP16 output.
     *   1.0 -> 0x3C00
     *   2.0 -> 0x4000
     *   3.0 -> 0x4200  (1.5 * 2^1)
     *   4.0 -> 0x4400
     */
    float     data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const char *path = "/tmp/test_pack_copy_fp32.bin";

    copy_tensor_fp16(data, 4, /*is_fp32=*/1, path);

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "could not open %s", path);

    uint16_t buf[4];
    size_t   r = fread(buf, sizeof(uint16_t), 4, f);
    fclose(f);
    CHECK(r == 4, "read %zu, expected 4", r);

    CHECK(buf[0] == 0x3C00, "buf[0]=0x%04X, expected 0x3C00 (1.0)", buf[0]);
    CHECK(buf[1] == 0x4000, "buf[1]=0x%04X, expected 0x4000 (2.0)", buf[1]);
    CHECK(buf[2] == 0x4200, "buf[2]=0x%04X, expected 0x4200 (3.0)", buf[2]);
    CHECK(buf[3] == 0x4400, "buf[3]=0x%04X, expected 0x4400 (4.0)", buf[3]);

    remove(path);
}

static void test_copy_tensor_fp16_passthrough(void)
{
    /* FP16 input -> byte-identical output. */
    uint16_t   in[4] = {0x3C00, 0x4000, 0x4200, 0x4400};
    const char *path = "/tmp/test_pack_copy_fp16.bin";

    copy_tensor_fp16(in, 4, /*is_fp32=*/0, path);

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "could not open %s", path);

    uint16_t buf[4];
    size_t   r = fread(buf, sizeof(uint16_t), 4, f);
    fclose(f);
    CHECK(r == 4, "read %zu, expected 4", r);

    CHECK(memcmp(in, buf, sizeof(in)) == 0,
          "fp16 passthrough mismatch");

    remove(path);
}

/* ------------------------------------------------------------------------- */
/* Main                                                                      */
/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("=== pack module tests ===\n");

    test_pack_idx4_known();
    test_pack_idx6_known();
    test_pack_idx8_known();

    test_pack_idx4_edge();
    test_pack_idx6_edge();
    test_pack_idx8_edge();

    test_pack_idx4_random();
    test_pack_idx6_random();
    test_pack_idx8_random();

    test_sanitize_name();

    test_write_lut_fp16();

    test_copy_tensor_fp16_fp32();
    test_copy_tensor_fp16_passthrough();

    printf("\n");
    printf("=== Summary: %d tests, %d failures ===\n",
           g_tests, g_failures);

    return g_failures == 0 ? 0 : 1;
}
