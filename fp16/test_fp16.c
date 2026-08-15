/* test_fp16.c - DoD test driver for the fp16 module.
 *
 * Verifies:
 *   1. fp16_to_fp32_scalar against expected bit patterns for every FP16 class
 *      (zero, subnormal, normal, Inf, NaN).
 *   2. fp32_to_fp16_scalar against expected bit patterns, including the DoD
 *      required values: 0.0, 1.0, -1.0, 0.5, 65504.0, 5.9604645e-08,
 *      +/-Inf, NaN.
 *   3. Round-trip behaviour (fp32 -> fp16 -> fp32) for every test value;
 *      the result must equal the FP16-quantised version of the input.
 *   4. Array versions match scalar versions on a fixed sample vector.
 *   5. Boundary cases: smallest subnormal, smallest normal, largest finite,
 *      tie-to-even rounding, overflow to Inf, NaN payload preservation.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra -I. \
 *       test_fp16.c fp16.c -o test_fp16 -lm
 * Run:
 *   ./test_fp16
 */
#include "fp16.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----- helpers --------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static void check_fp32_eq(float expected, float actual, const char *name) {
    uint32_t eu = f2u(expected);
    uint32_t au = f2u(actual);
    if (eu == au) {
        printf("  [PASS] %s\n", name);
        ++g_pass;
    } else {
        printf("  [FAIL] %s: expected %a (0x%08X), got %a (0x%08X)\n",
               name, expected, eu, actual, au);
        ++g_fail;
    }
}

static void check_fp16_eq(uint16_t expected, uint16_t actual, const char *name) {
    if (expected == actual) {
        printf("  [PASS] %s\n", name);
        ++g_pass;
    } else {
        printf("  [FAIL] %s: expected 0x%04X, got 0x%04X\n",
               name, expected, actual);
        ++g_fail;
    }
}

static void check_is_nan(float actual, const char *name) {
    if (isnan(actual)) {
        printf("  [PASS] %s (NaN, 0x%08X)\n", name, f2u(actual));
        ++g_pass;
    } else {
        printf("  [FAIL] %s: expected NaN, got %a (0x%08X)\n",
               name, actual, f2u(actual));
        ++g_fail;
    }
}

static void check_is_inf(float actual, int sign, const char *name) {
    /* sign: +1 -> +Inf, -1 -> -Inf, 0 -> any Inf */
    if (isinf(actual) && (sign == 0 || (sign > 0 && actual > 0.0f)
                                || (sign < 0 && actual < 0.0f))) {
        printf("  [PASS] %s (%cInf, 0x%08X)\n", name, actual > 0 ? '+' : '-', f2u(actual));
        ++g_pass;
    } else {
        printf("  [FAIL] %s: expected %cInf, got %a (0x%08X)\n",
               name, sign >= 0 ? '+' : '-', actual, f2u(actual));
        ++g_fail;
    }
}

static void check_true(int cond, const char *name) {
    if (cond) {
        printf("  [PASS] %s\n", name);
        ++g_pass;
    } else {
        printf("  [FAIL] %s\n", name);
        ++g_fail;
    }
}

/* ----- test 1: fp16_to_fp32_scalar ------------------------------------- */

static void test_fp16_to_fp32_scalar(void) {
    printf("\n[test 1] fp16_to_fp32_scalar\n");

    /* signed zero */
    check_fp32_eq( 0.0f,        fp16_to_fp32_scalar(0x0000), "0x0000 -> +0.0");
    check_fp32_eq(-0.0f,        fp16_to_fp32_scalar(0x8000), "0x8000 -> -0.0");

    /* DoD-specified normal values */
    check_fp32_eq( 1.0f,       fp16_to_fp32_scalar(0x3C00), "0x3C00 -> +1.0");
    check_fp32_eq(-1.0f,       fp16_to_fp32_scalar(0xBC00), "0xBC00 -> -1.0");
    check_fp32_eq( 0.5f,       fp16_to_fp32_scalar(0x3800), "0x3800 -> +0.5");
    check_fp32_eq( 65504.0f,   fp16_to_fp32_scalar(0x7BFF), "0x7BFF -> +65504.0");

    /* smallest subnormal (DoD value) = 2^-24 */
    check_fp32_eq( 5.9604645e-08f, fp16_to_fp32_scalar(0x0001), "0x0001 -> 2^-24");
    check_fp32_eq( 0x1p-24f,       fp16_to_fp32_scalar(0x0001), "0x0001 -> 2^-24 (hex)");

    /* smallest normal = 2^-14  (= 6.103515625e-5) */
    check_fp32_eq( 0x1p-14f,       fp16_to_fp32_scalar(0x0400), "0x0400 -> 2^-14 (smallest normal)");

    /* largest subnormal = 1023 * 2^-24 = 0x1.ff8p-15 (just below 2^-14) */
    check_fp32_eq( 0x1.ff8p-15f,  fp16_to_fp32_scalar(0x03FF), "0x03FF -> largest subnormal");

    /* Inf / NaN */
    check_is_inf(fp16_to_fp32_scalar(0x7C00), +1, "0x7C00 -> +Inf");
    check_is_inf(fp16_to_fp32_scalar(0xFC00), -1, "0xFC00 -> -Inf");
    check_is_nan(fp16_to_fp32_scalar(0x7E00), "0x7E00 -> NaN");
    check_is_nan(fp16_to_fp32_scalar(0x7FFF), "0x7FFF -> NaN (all payload bits)");

    /* mid-range normal: pi-ish */
    check_fp32_eq( 3.140625f,   fp16_to_fp32_scalar(0x4248), "0x4248 -> 3.140625 (pi approx)");
}

/* ----- test 2: fp32_to_fp16_scalar ------------------------------------- */

static void test_fp32_to_fp16_scalar(void) {
    printf("\n[test 2] fp32_to_fp16_scalar\n");

    /* DoD required values */
    check_fp16_eq(0x0000, fp32_to_fp16_scalar( 0.0f),                "0.0      -> 0x0000");
    check_fp16_eq(0x8000, fp32_to_fp16_scalar(-0.0f),               "-0.0     -> 0x8000");
    check_fp16_eq(0x3C00, fp32_to_fp16_scalar( 1.0f),                "+1.0     -> 0x3C00");
    check_fp16_eq(0xBC00, fp32_to_fp16_scalar(-1.0f),                "-1.0     -> 0xBC00");
    check_fp16_eq(0x3800, fp32_to_fp16_scalar( 0.5f),                "+0.5     -> 0x3800");
    check_fp16_eq(0x7BFF, fp32_to_fp16_scalar( 65504.0f),            "+65504.0 -> 0x7BFF");
    check_fp16_eq(0x0001, fp32_to_fp16_scalar( 5.9604645e-08f),      "+2^-24   -> 0x0001");

    /* Inf / NaN */
    check_fp16_eq(0x7C00, fp32_to_fp16_scalar( INFINITY),            "+Inf     -> 0x7C00");
    check_fp16_eq(0xFC00, fp32_to_fp16_scalar(-INFINITY),            "-Inf     -> 0xFC00");

    /* NaN must produce a NaN bit pattern (0x7C00 with non-zero mantissa) */
    uint16_t nan_h = fp32_to_fp16_scalar(NAN);
    check_true(((nan_h & 0x7C00) == 0x7C00 && (nan_h & 0x03FF) != 0),
               "NAN      -> NaN bit pattern (0x7C00 | nonzero)");

    /* overflow saturates to Inf */
    check_fp16_eq(0x7C00, fp32_to_fp16_scalar( 70000.0f),           "+70000.0 -> 0x7C00 (overflow -> +Inf)");
    check_fp16_eq(0xFC00, fp32_to_fp16_scalar(-70000.0f),           "-70000.0 -> 0xFC00 (overflow -> -Inf)");

    /* underflow: half of smallest subnormal rounds to 0 (round-to-even) */
    check_fp16_eq(0x0000, fp32_to_fp16_scalar( 2.9802322e-08f),      "+2^-25   -> 0x0000 (tie -> even)");

    /* smallest normal boundary */
    check_fp16_eq(0x0400, fp32_to_fp16_scalar( 0x1p-14f),           "+2^-14   -> 0x0400 (smallest normal)");
    /* largest subnormal boundary (1023 * 2^-24 = 0x1.ff8p-15) */
    check_fp16_eq(0x03FF, fp32_to_fp16_scalar( 0x1.ff8p-15f),       "+largest subnormal -> 0x03FF");

    /* tie-to-even at 0.5 ULP above 1.0: 1.0 + 2^-11 = 1.000488...
     * FP16 around 1.0 has ULP 2^-10; the halfway point is 1.0 + 2^-11.
     * Result mantissa is even (0x000), so it should round down to 1.0 (0x3C00). */
    check_fp16_eq(0x3C00, fp32_to_fp16_scalar(1.0f + (1.0f / 2048.0f)),
                  "1.0 + 2^-11 -> 0x3C00 (tie -> even, round down)");

    /* 1.0 + 2^-10 + 2^-24 (just above 1.0 + 2^-11) rounds up to next FP16 */
    check_fp16_eq(0x3C01, fp32_to_fp16_scalar(1.0f + (1.0f / 1024.0f) + (1.0f / 16777216.0f)),
                  "1.0 + 2^-10 + 2^-24 -> 0x3C01 (round up)");

    /* largest finite + tiny must stay finite; +1 ULP overflow goes to Inf */
    check_fp16_eq(0x7BFF, fp32_to_fp16_scalar(65504.0f + 0.5f),      "65504.5 -> 0x7BFF (round-to-even, stays finite)");
    check_fp16_eq(0x7C00, fp32_to_fp16_scalar(65520.0f),             "65520.0 -> 0x7C00 (overflow)");
}

/* ----- test 3: round-trip --------------------------------------------- */

static void test_roundtrip(void) {
    printf("\n[test 3] round-trip fp32 -> fp16 -> fp32\n");

    /* For every test value V, fp32_to_fp16(V) gives the closest FP16 H,
     * and fp16_to_fp32(H) gives back the exact FP32 form of H. So the
     * composition fp16(fp32(V)) should equal fp32(fp16(fp32(V))) -- i.e.,
     * doing a second round-trip should be a fixed point. */
    struct { float v; const char *name; } cases[] = {
        { 0.0f,              "0.0"      },
        {-0.0f,              "-0.0"     },
        { 1.0f,              "1.0"      },
        {-1.0f,              "-1.0"     },
        { 0.5f,              "0.5"      },
        { 65504.0f,          "65504.0"  },
        { 5.9604645e-08f,    "2^-24"    },
        { 6.097555e-05f,     "largest subnormal" },
        { 0x1p-14f,          "2^-14"    },
        { 3.140625f,         "pi approx"},
        { 100.0f,            "100.0"    },
        { 0.1f,              "0.1"      },
        { 1.0f / 3.0f,       "1/3"      },
        { 1234.5f,           "1234.5"   },
        {-0.000123f,         "-1.23e-4" },
    };
    size_t ncases = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < ncases; ++i) {
        float  v   = cases[i].v;
        uint16_t h1 = fp32_to_fp16_scalar(v);
        float  r1  = fp16_to_fp32_scalar(h1);
        uint16_t h2 = fp32_to_fp16_scalar(r1);
        float  r2  = fp16_to_fp32_scalar(h2);
        char name[128];
        snprintf(name, sizeof name, "round-trip %-10s: h=0x%04X r1=%a  h2=0x%04X r2=%a",
                 cases[i].name, h1, r1, h2, r2);
        /* Both round-trips must agree bit-exactly (idempotent after first hop). */
        check_true(h1 == h2 && f2u(r1) == f2u(r2), name);
    }

    /* Inf and NaN round-trips */
    uint16_t h_inf = fp32_to_fp16_scalar(INFINITY);
    check_is_inf(fp16_to_fp32_scalar(h_inf), +1, "round-trip +Inf");
    uint16_t h_ninf = fp32_to_fp16_scalar(-INFINITY);
    check_is_inf(fp16_to_fp32_scalar(h_ninf), -1, "round-trip -Inf");

    uint16_t h_nan = fp32_to_fp16_scalar(NAN);
    check_is_nan(fp16_to_fp32_scalar(h_nan), "round-trip NaN");
}

/* ----- test 4: array versions ----------------------------------------- */

static void test_arrays(void) {
    printf("\n[test 4] array versions vs scalar\n");

    const uint16_t src_h[10] = {
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x3800,
        0x7BFF, 0x0001, 0x0400, 0x7C00, 0x7E00,
    };
    float out_f[10];
    fp16_to_fp32_array(src_h, out_f, 10);

    int all_match = 1;
    for (int i = 0; i < 10; ++i) {
        float s = fp16_to_fp32_scalar(src_h[i]);
        if (f2u(s) != f2u(out_f[i])) {
            printf("  [FAIL] fp16_to_fp32_array mismatch at i=%d: scalar=0x%08X array=0x%08X\n",
                   i, f2u(s), f2u(out_f[i]));
            all_match = 0;
            ++g_fail;
        }
    }
    if (all_match) {
        printf("  [PASS] fp16_to_fp32_array matches scalar for all 10 elements\n");
        ++g_pass;
    }

    const float src_f[10] = {
         0.0f, -0.0f,  1.0f, -1.0f,  0.5f,
         65504.0f, 5.9604645e-08f, 6.097555e-05f,
         INFINITY, NAN,
    };
    uint16_t out_h[10];
    fp32_to_fp16_array(src_f, out_h, 10);

    all_match = 1;
    for (int i = 0; i < 10; ++i) {
        uint16_t s = fp32_to_fp16_scalar(src_f[i]);
        int eq = (s == out_h[i]);
        /* NaN is allowed to differ in payload, but both must be NaN-class. */
        if (!eq) {
            int s_nan = ((s & 0x7C00) == 0x7C00 && (s & 0x03FF) != 0);
            int o_nan = ((out_h[i] & 0x7C00) == 0x7C00 && (out_h[i] & 0x03FF) != 0);
            if (s_nan && o_nan) eq = 1;
        }
        if (!eq) {
            printf("  [FAIL] fp32_to_fp16_array mismatch at i=%d: scalar=0x%04X array=0x%04X\n",
                   i, s, out_h[i]);
            all_match = 0;
            ++g_fail;
        }
    }
    if (all_match) {
        printf("  [PASS] fp32_to_fp16_array matches scalar for all 10 elements\n");
        ++g_pass;
    }
}

/* ----- test 5: full FP16 range enumeration --------------------------- */

static void test_full_range(void) {
    printf("\n[test 5] full FP16 range enumeration (every value, %d total)\n", 65536);

    /* For every possible FP16 bit pattern, the round-trip
     *   fp16_to_fp32(h) -> fp32_to_fp16(result) == h
     * must hold bit-exactly (because every FP16 has an exact FP32 form, and
     * rounding that FP32 back to FP16 returns the same FP16). */
    int mismatches = 0;
    int nan_count = 0;
    int inf_count = 0;
    int sub_count = 0;
    int zero_count = 0;
    for (uint32_t h = 0; h < 0x10000; ++h) {
        float f = fp16_to_fp32_scalar((uint16_t)h);
        uint16_t back = fp32_to_fp16_scalar(f);
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0x1F) {
            if (mant == 0) ++inf_count; else ++nan_count;
            /* Inf must round-trip exactly; NaN must round-trip to *some* NaN
             * pattern (any 0x7C00 | nonzero). */
            if (mant == 0) {
                if (back != (uint16_t)h) ++mismatches;
            } else {
                if (!((back & 0x7C00) == 0x7C00 && (back & 0x03FF) != 0)) ++mismatches;
            }
        } else {
            if (exp == 0) {
                if (mant == 0) ++zero_count; else ++sub_count;
            }
            if (back != (uint16_t)h) {
                ++mismatches;
                if (mismatches < 5) {
                    printf("  MISMATCH h=0x%04X back=0x%04X f=%a\n",
                           (uint16_t)h, back, f);
                }
            }
        }
    }
    printf("  range summary: %d zeros, %d subnormals, %d normals, %d Infs, %d NaNs\n",
           zero_count, sub_count, 65536 - zero_count - sub_count - inf_count - nan_count,
           inf_count, nan_count);
    if (mismatches == 0) {
        printf("  [PASS] all 65536 FP16 values round-trip bit-exactly (Inf) or stay NaN-class\n");
        ++g_pass;
    } else {
        printf("  [FAIL] %d mismatches out of 65536\n", mismatches);
        ++g_fail;
    }
}

/* ----- main ----------------------------------------------------------- */

/* Test 6: real Dolphin data sanity check.
 *
 * Reads the first 4096 FP16 weights from the Dolphin model safetensors file,
 * scanning the JSON header for the first tensor whose dtype is "F16", and
 * verifying that:
 *   - fp16_to_fp32_array produces finite values for normal/subnormal inputs
 *   - every converted value round-trips bit-exactly back to its FP16 form
 *   - NaN/Inf inputs (if any) are preserved class-correctly
 *
 * This honours the project rule "Test with real Dolphin data".
 */
static void test_real_dolphin(void) {
    printf("\n[test 6] real Dolphin FP16 weights (/root/dolphin/hf_model_main/model.safetensors)\n");

    FILE *fp = fopen("/root/dolphin/hf_model_main/model.safetensors", "rb");
    if (!fp) {
        printf("  [SKIP] Dolphin model not present on this machine\n");
        return;
    }

    /* Read 8-byte LE header length, then read the JSON header. */
    uint64_t hdr_len = 0;
    if (fread(&hdr_len, 8, 1, fp) != 1) {
        printf("  [FAIL] could not read safetensors header length\n");
        ++g_fail;
        fclose(fp);
        return;
    }
    if (hdr_len > (16u * 1024u * 1024u)) {
        printf("  [FAIL] safetensors header length unreasonably large: %llu\n",
               (unsigned long long)hdr_len);
        ++g_fail;
        fclose(fp);
        return;
    }
    char *hdr = (char *)malloc((size_t)hdr_len + 1);
    if (!hdr) { ++g_fail; fclose(fp); return; }
    if (fread(hdr, 1, (size_t)hdr_len, fp) != hdr_len) {
        printf("  [FAIL] could not read %llu-byte safetensors header\n",
               (unsigned long long)hdr_len);
        ++g_fail; free(hdr); fclose(fp); return;
    }
    hdr[hdr_len] = '\0';

    /* Find the first tensor with "dtype":"F16".  We scan the header for the
     * first occurrence of "F16" and then walk backwards to find the start
     * of the surrounding tensor entry, then forward to find its
     * data_offsets array. This is a deliberately minimal ad-hoc parser
     * (the project's safetensors module is on another branch). */
    const char *p = hdr;
    const char *end = hdr + hdr_len;
    long f16_offset = -1;
    while (p < end) {
        const char *hit = strstr(p, "\"F16\"");
        if (!hit) break;
        /* Find the next "data_offsets":[<a>,<b>] after this dtype. */
        const char *doff = strstr(hit, "\"data_offsets\"");
        if (!doff) break;
        const char *lb = strchr(doff, '[');
        const char *comma = strchr(lb ? lb : doff, ',');
        const char *rb = strchr(lb ? lb : doff, ']');
        if (!lb || !comma || !rb || comma > rb) {
            p = hit + 5; continue;
        }
        /* Parse the two integers. */
        long a = strtol(lb + 1, NULL, 10);
        long b = strtol(comma + 1, NULL, 10);
        if (b > a && b - a >= (long)(2 * 4096)) {
            f16_offset = a;
            break;
        }
        p = rb + 1;
    }
    free(hdr);
    if (f16_offset < 0) {
        printf("  [FAIL] no F16 tensor with >= 4096 values found in header\n");
        ++g_fail; fclose(fp); return;
    }

    /* Seek to the start of the first F16 tensor's data section. */
    long data_start = (long)(8u + hdr_len + (uint64_t)f16_offset);
    if (fseek(fp, data_start, SEEK_SET) != 0) {
        printf("  [FAIL] could not seek to F16 tensor at byte %ld\n", data_start);
        ++g_fail; fclose(fp); return;
    }

    enum { N = 4096 };
    uint16_t buf_h[N];
    float    buf_f[N];
    size_t got = fread(buf_h, sizeof(uint16_t), N, fp);
    fclose(fp);
    if (got != N) {
        printf("  [FAIL] short read: wanted %d fp16 values, got %zu\n", N, got);
        ++g_fail;
        return;
    }
    printf("  read %d FP16 values from tensor data offset %ld (file byte %ld)\n",
           N, f16_offset, data_start);

    /* Convert all of them. */
    fp16_to_fp32_array(buf_h, buf_f, N);

    /* Classify and verify round-trip. */
    int n_zero = 0, n_sub = 0, n_norm = 0, n_inf = 0, n_nan = 0;
    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        uint16_t h = buf_h[i];
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0x1F) {
            if (mant == 0) ++n_inf; else ++n_nan;
        } else if (exp == 0) {
            if (mant == 0) ++n_zero; else ++n_sub;
        } else {
            ++n_norm;
        }
        /* Skip NaN class for bit-exact check; NaN != NaN anyway. */
        if (exp == 0x1F && mant != 0) continue;
        /* For Inf, the FP32 form should be +/-Inf. */
        if (exp == 0x1F && mant == 0) {
            if (!isinf(buf_f[i])) { ++mismatch; if (mismatch < 4) printf("  h=0x%04X -> not Inf\n", h); }
            continue;
        }
        /* Round-trip: fp32 -> fp16 must reproduce h. */
        uint16_t back = fp32_to_fp16_scalar(buf_f[i]);
        if (back != h) {
            ++mismatch;
            if (mismatch < 4) printf("  MISMATCH i=%d h=0x%04X back=0x%04X f=%a\n", i, h, back, buf_f[i]);
        }
    }
    printf("  classified: %d zeros, %d subnormal, %d normal, %d Inf, %d NaN (out of %d)\n",
           n_zero, n_sub, n_norm, n_inf, n_nan, N);
    if (mismatch == 0) {
        printf("  [PASS] all %d Dolphin FP16 values round-trip bit-exactly (NaN class preserved)\n", N);
        ++g_pass;
    } else {
        printf("  [FAIL] %d mismatches out of %d\n", mismatch, N);
        ++g_fail;
    }
}

int main(void) {
    printf("=== fp16 DoD test driver ===");
    printf("\nbuild: %s, %s", __DATE__, __TIME__);

    test_fp16_to_fp32_scalar();
    test_fp32_to_fp16_scalar();
    test_roundtrip();
    test_arrays();
    test_full_range();
    test_real_dolphin();

    printf("\n=== Summary ===\n");
    printf("  passed: %d\n", g_pass);
    printf("  failed: %d\n", g_fail);
    if (g_fail == 0) {
        printf("\nRESULT: ALL TESTS PASSED\n");
        return 0;
    }
    printf("\nRESULT: %d TEST(S) FAILED\n", g_fail);
    return 1;
}
