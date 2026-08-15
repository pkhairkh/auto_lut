/* test_safetensors.c -- load a .safetensors file and print summary info.
 *
 * Usage:
 *     ./test_safetensors <model.safetensors> [N]
 *
 * Default model path: /root/dolphin/hf_model_main/model.safetensors
 * Default N (tensors to print): 5
 *
 * DoD check: prints tensor count and the first N tensors'
 * name / dtype / shape / byte_size.
 */

#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_tensor(const TensorInfo *ti, int idx)
{
    printf("[%4d] %-64s %6s  shape=[", idx, ti->name, ti->dtype);
    for (int i = 0; i < ti->ndim; i++) {
        printf("%d%s", ti->shape[i], (i + 1 < ti->ndim) ? "," : "");
    }
    printf("]  bytes=%zu  n_elem=%zu\n", ti->byte_size, ti->n_elements);
}

int main(int argc, char **argv)
{
    const char *path = "/root/dolphin/hf_model_main/model.safetensors";
    int n_print = 5;
    if (argc >= 2) path    = argv[1];
    if (argc >= 3) n_print = atoi(argv[2]);
    if (n_print < 0) n_print = 0;

    fprintf(stderr, "Loading %s ...\n", path);
    SafeTensors *st = safetensors_load(path);
    if (!st) {
        fprintf(stderr, "FAIL: safetensors_load returned NULL\n");
        return 1;
    }

    printf("=== safetensors summary ===\n");
    printf("file_size   : %zu bytes\n", st->file_size);
    printf("data_start  : %zu\n", st->data_start);
    printf("tensor_count: %d\n", st->count);
    printf("\n");

    /* First N tensors in declaration order. */
    int limit = (n_print < st->count) ? n_print : st->count;
    printf("--- first %d tensors ---\n", limit);
    for (int i = 0; i < limit; i++) {
        print_tensor(&st->tensors[i], i);
    }
    printf("\n");

    /* Quick lookup sanity-check: probe a few names actually present in
     * the Dolphin model (so find() exercises a hit, not just a miss). */
    const char *names_to_try[] = {
        "encoder.encoder.layers.0.blocks.0.attention.self.relative_position_index",
        "decoder.model.decoder.layers.1.encoder_attn.q_proj.weight",
        "encoder.encoder.layers.3.blocks.1.output.dense.weight",
        "lm_head.weight",     /* expected: NOTFOUND (dolphin doesn't use it) */
        NULL
    };
    printf("--- find() probes ---\n");
    for (int i = 0; names_to_try[i]; i++) {
        TensorInfo *ti = safetensors_find(st, names_to_try[i]);
        if (ti) {
            printf("FOUND  %s  dtype=%s  ndim=%d  bytes=%zu  ptr_ok=%d\n",
                   names_to_try[i], ti->dtype, ti->ndim, ti->byte_size,
                   safetensors_get_ptr(st, ti) ? 1 : 0);
        } else {
            printf("NOTFOUND  %s\n", names_to_try[i]);
        }
    }

    safetensors_free(st);
    return 0;
}
