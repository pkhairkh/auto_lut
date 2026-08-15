#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

/* ===========================================================================
 * tokenizer.h — Minimal HuggingFace tokenizer loader for auto_lut.
 *
 * Loads a HF tokenizer.json and exposes only what the forward-pass driver
 * needs: encode a string into a list of input_ids using the "added_tokens"
 * list (special tokens) and a basic BPE/WordPiece fallback that maps
 * characters to their unicode codepoint id (used only for the prompt
 * "<s>" -> [0] for the Dolphin decoder prefill).
 *
 * This is NOT a full BPE implementation. The auto_lut forward pass needs
 * only the fixed 5-token Dolphin prompt: "<s>", "<s_docvqa>", "<question>",
 * "<answer>", "</s>" — all of which appear in added_tokens with explicit
 * ids. The driver constructs the prompt directly; this module exists to
 * look up token ids by content for any future prompt variants.
 * =========================================================================== */

typedef struct {
    int    id;
    char  *content;   /* owned, NUL-terminated */
} AddedToken;

typedef struct {
    AddedToken *tokens;
    int         count;
    int         cap;
} Tokenizer;

/* Load tokenizer.json. Returns NULL on error. */
Tokenizer *tokenizer_load(const char *path);

/* Look up a token by its exact content string. Returns id or -1. */
int tokenizer_lookup(const Tokenizer *tk, const char *content);

/* Free. Safe on NULL. */
void tokenizer_free(Tokenizer *tk);

#endif /* TOKENIZER_H */
