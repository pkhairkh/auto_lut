#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * tokenizer.h - BPE Tokenizer for the auto_lut project
 *
 * Loads a HuggingFace `tokenizer.json` (BPE model with ByteLevel
 * pre-tokenizer and decoder) and exposes encode/decode in pure C.
 *
 * Design notes:
 *   - Depends only on the in-tree json.h/json.c parser.
 *   - No external libraries; C11 only.
 *   - Special tokens (added_tokens) are matched greedily by longest
 *     content string before BPE pre-tokenization runs.
 *   - ByteLevel pre-tokenization follows the GPT-2 recipe: regex
 *     split + byte-to-unicode map (so a leading space becomes U+0120 'Ġ').
 *   - BPE merges are applied greedily by lowest rank.
 *   - The HuggingFace post-processor (template wrapping with <s>/</s>)
 *     is NOT applied; callers that need wrapping must add it themselves.
 *     This matches the Dolphin tokenizer test expectation that the
 *     input text is encoded verbatim without synthetic bos/eos injection.
 *
 * Memory model:
 *   - The returned int array from tokenizer_encode() is malloc'd and
 *     owned by the caller; free with free().
 *   - The returned char* from tokenizer_decode() is malloc'd and owned
 *     by the caller; free with free().
 *   - Tokenizer itself is an opaque pointer; free with tokenizer_free().
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tokenizer Tokenizer;

/*
 * Load a HuggingFace tokenizer.json file.
 *
 * Returns NULL on failure (file not found, parse error, OOM).
 * The returned Tokenizer must be freed with tokenizer_free().
 */
Tokenizer *tokenizer_load(const char *path);

/*
 * Encode a NUL-terminated UTF-8 string into a sequence of token IDs.
 *
 * - Special tokens embedded in `text` (e.g. "<s>", " <Answer/>") are
 *   matched directly against the tokenizer's added_tokens list and
 *   emitted as their registered IDs (NOT split into BPE pieces).
 * - All other text is split by the ByteLevel pre-tokenizer and then
 *   BPE-merged.
 *
 * On success, returns a malloc'd int array of length *out_len.
 * The caller owns the array and must free() it.
 *
 * On failure, returns NULL and sets *out_len = 0.
 */
int *tokenizer_encode(Tokenizer *tok, const char *text, int *out_len);

/*
 * Decode a sequence of token IDs back to a NUL-terminated UTF-8 string.
 *
 * - Applies the ByteLevel decoder: each token's vocab string is
 *   converted from byte-level unicode (e.g. 'Ġ' -> space) back to
 *   raw bytes, then concatenated.
 * - Special tokens are emitted literally (their content strings are
 *   already in their final form, e.g. " <Answer/>").
 *
 * On success, returns a malloc'd char* the caller must free().
 * On failure, returns NULL.
 */
char *tokenizer_decode(Tokenizer *tok, const int *ids, int len);

/*
 * Free a Tokenizer previously returned by tokenizer_load().
 * Safe to call on NULL.
 */
void tokenizer_free(Tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
