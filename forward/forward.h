#ifndef FORWARD_H
#define FORWARD_H

#include "safetensors.h"
#include "json.h"

/* ===========================================================================
 * forward.h — Generic transformer forward pass with activation capture.
 *
 * Loads a VisionEncoderDecoderModel (e.g. Dolphin: DonutSwin encoder +
 * mBART decoder) from a .safetensors file plus its config.json, runs the
 * forward pass over a batch of preprocessed images + a fixed decoder
 * prompt, and captures the input activation X (seq_len, in_dim) for every
 * 2D weight tensor just before the corresponding matmul.
 *
 * Captured activations are used downstream by the GPTQ palettizer
 * to compute per-output-channel Hessians for Hessian-weighted quantization.
 *
 * The implementation is model-agnostic at the weight-resolution layer:
 * it scans all 2D tensors, groups them by the layer index encoded in
 * their name, and identifies q/k/v/o/fc1/fc2/etc by their shape.
 * =========================================================================== */

/* Maximum number of samples (rows) captured per tensor. The Hessian only
 * needs a few thousand samples to be representative; capping keeps memory
 * bounded. */
#define FORWARD_ACT_SAMPLE_CAP 2048

/* Maximum number of distinct activations (one per 2D weight tensor) tracked
 * across the whole forward pass. 640 tensors in the Dolphin model, so 1024
 * is comfortable headroom. */
#define FORWARD_ACT_CAP 1024

/* Maximum tensor name length (matches SAFETENSORS_NAME_MAX). */
#define FORWARD_NAME_MAX 512

/* A captured activation for a single weight tensor.
 *
 *   name      : the weight tensor's name (e.g. "decoder.model.decoder.layers.0.fc1.weight")
 *   in_dim    : the width of the matmul (weight's input dimension)
 *   n_samples : number of captured rows (<= FORWARD_ACT_SAMPLE_CAP)
 *   data      : (n_samples, in_dim) row-major float32. Each row is a single
 *               activation vector X observed just before the matmul Y = X @ W^T.
 *   hessian   : (in_dim,) float32. Per-channel sum of squares of X,
 *               i.e. hessian[j] = sum_i X[i,j]^2. Computed incrementally as
 *               samples are appended. */
typedef struct {
    char    name[FORWARD_NAME_MAX];
    int     in_dim;
    int     n_samples;
    float  *data;            /* n_samples * in_dim floats */
    float  *hessian;         /* in_dim floats (diagonal) */
    float  *hessian_matrix;  /* in_dim * in_dim floats (full H = X^T X), NULL if not computed */
    int     has_hessian_matrix;
} Activation;

/* A collection of captured activations plus the model state needed to
 * keep capturing more. Returned by forward_run(); caller releases with
 * activation_capture_free(). */
typedef struct {
    SafeTensors  *st;       /* borrowed (caller owns) */
    JsonValue    *cfg;      /* borrowed (caller owns) */
    Activation   *acts;     /* owned array */
    int           n_acts;
    int           act_cap;
    char          output_dir[512];  /* where .bin / manifest.json were written */
} ActivationCapture;

/* Run the forward pass over `n_images` preprocessed images and a fixed
 * decoder prompt of `n_prompt_tokens` input_ids.
 *
 *   st               : loaded SafeTensors (caller owns; must outlive the capture)
 *   cfg              : parsed config.json (caller owns)
 *   pixel_values     : array of `n_images` float* pointers, each pointing to
 *                      a CHW float32 buffer of size 3 * 896 * 896 (Donut)
 *   n_images         : number of images
 *   input_ids        : array of `n_prompt_tokens` ints (decoder prompt)
 *   n_prompt_tokens  : prompt length (typically 5 for Dolphin: <s> <task> <s_docvqa> </s> </s>)
 *   output_dir       : directory to write activation .bin files + manifest.json
 *
 * Returns an ActivationCapture handle, or NULL on any error.
 * The captured activations are also persisted to output_dir as:
 *   <output_dir>/activations/<sanitized_name>.bin  (raw float32, n_samples*in_dim)
 *   <output_dir>/activations/manifest.json         (index of all captured tensors)
 *
 * Each .bin file is consumed by the GPTQ palettizer to compute
 * Hessian weights for the corresponding tensor. */
ActivationCapture *forward_run(
    SafeTensors *st, JsonValue *cfg,
    float **pixel_values, int n_images,
    int *input_ids, int n_prompt_tokens,
    const char *output_dir
);

/* Release an ActivationCapture and all owned memory. Safe on NULL. */
void activation_capture_free(ActivationCapture *ac);

#endif /* FORWARD_H */
