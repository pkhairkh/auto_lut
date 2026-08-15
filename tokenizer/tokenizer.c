/*
 * tokenizer.c - BPE Tokenizer implementation for the auto_lut project
 *
 * Pure C11, no external libraries beyond the in-tree json.h/json.c.
 *
 * Pipeline (encode):
 *   1. Read whole tokenizer.json via json_parse().
 *   2. Build vocab hash map (string -> int id).
 *   3. Build reverse-vocab array (int id -> string), sized by max id.
 *   4. Build merge-rank hash map ("tokenA tokenB" -> rank).
 *   5. Build added_tokens list (special tokens, longest-first match).
 *   6. For encode(text):
 *      a. Walk text, at each position try to match a special token
 *         (longest content first). On match, emit its id verbatim.
 *      b. For non-special segments, run the pre-tokenizer Sequence:
 *           - Split on literal "SPL1T-TH1S-Pl3A5E" (drop).
 *           - Split digit runs into individual digits.
 *           - Isolate bracket chars and runs of ASCII punctuation.
 *           - Isolate newlines.
 *           - Apply ByteLevel: GPT-2 regex + byte-to-unicode map.
 *      c. For each ByteLevel piece, run BPE merges greedily by rank,
 *         then look up final tokens in vocab.
 *   7. For decode(ids): concatenate vocab[id] strings, then reverse
 *      the byte-level unicode map back to raw bytes.
 *
 * Limitations (documented):
 *   - The GPT-2 regex matcher is ASCII-aware: it treats [a-zA-Z] as
 *     \p{L}, [0-9] as \p{N}, and standard ASCII whitespace as \s.
 *     Multi-byte UTF-8 letters are handled byte-by-byte via the
 *     byte-level map (they become sequences of U+0100+ chars) but
 *     are NOT grouped as single \p{L} code points. This is sufficient
 *     for the Dolphin English test prompt and most Latin-script text.
 *   - The NFKC normalizer is declared in tokenizer.json but skipped
 *     here (the Dolphin prompt is already NFKC-stable). Adding full
 *     NFKC requires Unicode tables; deferred to a later module.
 *   - The post_processor (TemplateProcessing with <s>/</s>) is NOT
 *     applied, matching the DoD test expectation.
 */

#include "tokenizer.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/* Hash map: char* key -> int value                                    */
/* ------------------------------------------------------------------ */

#define HASH_INIT_CAP 1024
#define HASH_MAX_LOAD 0.70

typedef struct HashNode {
    char             *key;     /* owned, NUL-terminated */
    int               value;
    struct HashNode  *next;
} HashNode;

typedef struct {
    HashNode **buckets;
    size_t     cap;
    size_t     count;
} HashMap;

static uint64_t fnv1a(const char *s, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int hashmap_init(HashMap *m, size_t cap) {
    m->buckets = calloc(cap, sizeof(HashNode *));
    if (!m->buckets) return -1;
    m->cap = cap;
    m->count = 0;
    return 0;
}

static void hashmap_free(HashMap *m) {
    if (!m->buckets) return;
    for (size_t i = 0; i < m->cap; i++) {
        HashNode *n = m->buckets[i];
        while (n) {
            HashNode *next = n->next;
            free(n->key);
            free(n);
            n = next;
        }
    }
    free(m->buckets);
    m->buckets = NULL;
    m->cap = m->count = 0;
}

/* Returns 1 if found (writes *out), 0 if not found. */
static int hashmap_get(const HashMap *m, const char *key, size_t key_len, int *out) {
    if (!m->buckets) return 0;
    uint64_t h = fnv1a(key, key_len) % m->cap;
    for (HashNode *n = m->buckets[h]; n; n = n->next) {
        /* Compare lengths first, then bytes */
        size_t nk = strlen(n->key);
        if (nk == key_len && memcmp(n->key, key, key_len) == 0) {
            *out = n->value;
            return 1;
        }
    }
    return 0;
}

/* Insert (key, value). Duplicates are not checked (caller's responsibility).
 * Returns 0 on success, -1 on OOM. */
static int hashmap_put(HashMap *m, const char *key, size_t key_len, int value) {
    if ((double)m->count / m->cap > HASH_MAX_LOAD) {
        size_t new_cap = m->cap * 2;
        HashNode **new_buckets = calloc(new_cap, sizeof(HashNode *));
        if (!new_buckets) return -1;
        for (size_t i = 0; i < m->cap; i++) {
            HashNode *n = m->buckets[i];
            while (n) {
                HashNode *next = n->next;
                size_t nk = strlen(n->key);
                uint64_t h = fnv1a(n->key, nk) % new_cap;
                n->next = new_buckets[h];
                new_buckets[h] = n;
                n = next;
            }
        }
        free(m->buckets);
        m->buckets = new_buckets;
        m->cap = new_cap;
    }
    HashNode *node = malloc(sizeof(HashNode));
    if (!node) return -1;
    node->key = malloc(key_len + 1);
    if (!node->key) { free(node); return -1; }
    memcpy(node->key, key, key_len);
    node->key[key_len] = '\0';
    node->value = value;
    uint64_t h = fnv1a(key, key_len) % m->cap;
    node->next = m->buckets[h];
    m->buckets[h] = node;
    m->count++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Tokenizer state                                                     */
/* ------------------------------------------------------------------ */

struct Tokenizer {
    /* vocab: token-string -> id  (built from model.vocab + added_tokens) */
    HashMap     vocab;
    /* reverse vocab: id -> token-string (malloc'd, size = rev_cap) */
    char      **rev_vocab;
    size_t      rev_cap;        /* allocated length of rev_vocab */
    int         max_id;         /* largest id seen during load  */
    /* merge ranks: "tokA tokB" -> rank (lower = applied earlier) */
    HashMap     merges;
    int         merge_count;
    /* special tokens (added_tokens with special=true), longest-first */
    char      **special_str;
    int        *special_id;
    size_t      special_count;
    /* id of <unk>, or -1 if not present */
    int         unk_id;
};

/* ------------------------------------------------------------------ */
/* Special tokens: load and longest-first match                       */
/* ------------------------------------------------------------------ */

/* We need to sort special_str and special_id together by length-desc.
 * Build an index array, sort that, then compact. */
static void sort_specials(Tokenizer *t) {
    if (t->special_count < 2) return;
    /* simple insertion sort by length-desc, then alpha */
    for (size_t i = 1; i < t->special_count; i++) {
        char *si = t->special_str[i];
        int   ii = t->special_id[i];
        size_t li = strlen(si);
        size_t j = i;
        while (j > 0) {
            size_t lj = strlen(t->special_str[j-1]);
            if (lj < li || (lj == li && strcmp(t->special_str[j-1], si) > 0)) {
                t->special_str[j] = t->special_str[j-1];
                t->special_id[j]  = t->special_id[j-1];
                j--;
            } else break;
        }
        t->special_str[j] = si;
        t->special_id[j]  = ii;
    }
}

/* ------------------------------------------------------------------ */
/* GPT-2 byte-to-unicode mapping                                       */
/* ------------------------------------------------------------------ */

/* The standard GPT-2 byte-to-unicode table. For each byte 0..255,
 * returns a unicode code point that is safe to store in a UTF-8
 * string. The reverse map converts back to the original byte. */
static int byte_to_cp[256];
static int cp_to_byte[256 * 4 + 64]; /* sparse: index by cp; for cp<1024 */
static int cp_to_byte_inited = 0;

static void init_byte_maps(void) {
    if (cp_to_byte_inited) return;
    /* Build byte -> code point */
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp;
        if ( (b >= 33 && b <= 126) ||
             (b >= 161 && b <= 172) ||
             (b >= 174 && b <= 255) ) {
            cp = b;
        } else {
            cp = 256 + n;
            n++;
        }
        byte_to_cp[b] = cp;
    }
    /* Build reverse: cp -> byte. Mark unused as -1. */
    for (int i = 0; i < (int)(sizeof(cp_to_byte)/sizeof(int)); i++)
        cp_to_byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        int cp = byte_to_cp[b];
        if (cp < (int)(sizeof(cp_to_byte)/sizeof(int)))
            cp_to_byte[cp] = b;
    }
    cp_to_byte_inited = 1;
}

/* Encode a sequence of bytes (typically a pre-token piece) to a UTF-8
 * string using the GPT-2 byte-to-unicode map. Returns malloc'd NUL-term
 * string. Caller frees. */
static char *bytes_to_bytestr(const unsigned char *bytes, size_t len) {
    /* Each byte maps to one code point in range 33..255, 256..323.
     * Code points <= 127 are 1 byte UTF-8; <= 2047 are 2 bytes UTF-8.
     * Worst case: 2 UTF-8 bytes per input byte + NUL. */
    size_t cap = len * 2 + 1;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = byte_to_cp[bytes[i]];
        if (cp < 0x80) {
            out[pos++] = (char)cp;
        } else {
            /* 2-byte UTF-8: 110xxxxx 10xxxxxx */
            out[pos++] = (char)(0xC0 | (cp >> 6));
            out[pos++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[pos] = '\0';
    return out;
}

/* ------------------------------------------------------------------ */
/* Pre-tokenizer Sequence                                              */
/* ------------------------------------------------------------------ */

/* A dynamic array of (start,len) byte ranges into the input segment. */
typedef struct {
    size_t *starts;
    size_t *lens;
    size_t  count;
    size_t  cap;
} PieceList;

static int plist_init(PieceList *p, size_t cap) {
    p->starts = malloc(cap * sizeof(size_t));
    p->lens   = malloc(cap * sizeof(size_t));
    if (!p->starts || !p->lens) { free(p->starts); free(p->lens); p->starts=NULL; p->lens=NULL; return -1; }
    p->count = 0;
    p->cap = cap;
    return 0;
}

static void plist_free(PieceList *p) {
    free(p->starts); free(p->lens);
    p->starts = p->lens = NULL;
    p->count = p->cap = 0;
}

static int plist_push(PieceList *p, size_t start, size_t len) {
    if (p->count == p->cap) {
        size_t new_cap = p->cap * 2;
        size_t *ns = realloc(p->starts, new_cap * sizeof(size_t));
        size_t *nl = realloc(p->lens,   new_cap * sizeof(size_t));
        if (!ns || !nl) { free(ns); free(nl); return -1; }
        p->starts = ns; p->lens = nl; p->cap = new_cap;
    }
    p->starts[p->count] = start;
    p->lens[p->count]   = len;
    p->count++;
    return 0;
}

/* Apply GPT-2 ByteLevel regex on a byte range. Splits into pieces
 * per the pattern:
 *   's | 't | 're | 've | 'm | 'll | 'd
 *   | ' ?\p{L}+ | ' ?\p{N}+ | ' ?[^\s\p{L}\p{N}]+
 *   | \s+(?!\S) | \s+
 * (ASCII-only: \p{L} = [a-zA-Z], \p{N} = [0-9], \s = [ \t\n\r\v\f])
 *
 * Each piece's bytes are converted via bytes_to_bytestr() by the caller.
 * The pieces pushed here are byte ranges into `text` of length `len`.
 *
 * add_prefix_space=false (matches the Dolphin config): no leading space
 * is added; the regex naturally preserves existing leading spaces.
 */
static int bytlevel_split(const char *text, size_t len, PieceList *out) {
    size_t i = 0;
    while (i < len) {
        /* Try contractions: 's 't 're 've 'm 'll 'd (case-insensitive s/t/re/ve/m/ll/d) */
        if (text[i] == '\'' && i + 1 < len) {
            size_t consumed = 0;
            char c1 = tolower((unsigned char)text[i+1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                consumed = 2;
            } else if (i + 2 < len) {
                char c2 = tolower((unsigned char)text[i+2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'l' && c2 == 'l')) {
                    consumed = 3;
                }
            }
            if (consumed > 0) {
                if (plist_push(out, i, consumed) != 0) return -1;
                i += consumed;
                continue;
            }
        }

        /* ' ?\p{L}+ : optional leading space then 1+ letters */
        if (i + 1 < len && text[i] == ' ' && isalpha((unsigned char)text[i+1])) {
            size_t start = i;
            i += 2;
            while (i < len && isalpha((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }
        if (isalpha((unsigned char)text[i])) {
            size_t start = i;
            while (i < len && isalpha((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }

        /* ' ?\p{N}+ : optional leading space then 1+ digits */
        if (i + 1 < len && text[i] == ' ' && isdigit((unsigned char)text[i+1])) {
            size_t start = i;
            i += 2;
            while (i < len && isdigit((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }
        if (isdigit((unsigned char)text[i])) {
            size_t start = i;
            while (i < len && isdigit((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }

        /* ' ?[^\s\p{L}\p{N}]+ : optional leading space then 1+ other */
        if (i + 1 < len && text[i] == ' ' && !isspace((unsigned char)text[i+1]) &&
            !isalpha((unsigned char)text[i+1]) && !isdigit((unsigned char)text[i+1])) {
            size_t start = i;
            i += 2;
            while (i < len && !isspace((unsigned char)text[i]) &&
                   !isalpha((unsigned char)text[i]) && !isdigit((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }
        if (!isspace((unsigned char)text[i]) &&
            !isalpha((unsigned char)text[i]) && !isdigit((unsigned char)text[i])) {
            size_t start = i;
            while (i < len && !isspace((unsigned char)text[i]) &&
                   !isalpha((unsigned char)text[i]) && !isdigit((unsigned char)text[i])) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }

        /* \s+(?!\S) : run of whitespace not followed by non-whitespace
         * (i.e. trailing whitespace at EOS) */
        if (isspace((unsigned char)text[i])) {
            size_t start = i;
            while (i < len && isspace((unsigned char)text[i])) i++;
            /* If i == len, all the trailing whitespace is one piece (matched \s+(?!\S)).
             * If i < len, the regex backtracks one char so that \s+ matches all-but-last
             * and the last space becomes the leading space of the next piece.
             * Implement that: if there's at least 1 non-ws char left, undo the last ws. */
            if (i < len && i - 1 > start) {
                i--;
            }
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }

        /* Fallback (shouldn't happen with the above guards): one byte */
        if (plist_push(out, i, 1) != 0) return -1;
        i++;
    }
    return 0;
}

/* Apply the Dolphin pre-tokenizer Sequence on a regular (non-special)
 * byte range. Output is a list of byte ranges that should then be
 * byte-level-encoded by the caller.
 *
 * Stages applied (in order):
 *   1. Split on literal "SPL1T-TH1S-Pl3A5E" (Removed: drop).
 *   2. Digits: individual_digits=true (split runs into single chars).
 *   3. Split on regex [\(\)\[\]\{\}]|([!...])\1* (Isolated).
 *   4. Split on \n (Isolated).
 *   5. ByteLevel (use_regex=true, add_prefix_space=false).
 *
 * For simplicity we merge stages 2-4: scan the input left-to-right,
 * emitting pieces, with each isolated bracket/punct/newline/digit
 * emitted as its own piece. Other characters accumulate into a "run"
 * piece that is then fed to the ByteLevel splitter.
 */
static int pretokenize_segment(const char *text, size_t len, PieceList *out) {
    size_t i = 0;
    size_t run_start = 0;
    int has_run = 0;

    /* flush the accumulated "run" through ByteLevel */
    #define FLUSH_RUN() do {                                  \
        if (has_run) {                                        \
            if (bytlevel_split(text + run_start, i - run_start, out) != 0) \
                return -1;                                    \
            has_run = 0;                                      \
        }                                                     \
    } while (0)

    while (i < len) {
        /* Stage 1: literal SPL1T-TH1S-Pl3A5E (15 chars), Removed */
        static const char SPL[] = "SPL1T-TH1S-Pl3A5E";
        static const size_t SPL_LEN = 17;
        if (i + SPL_LEN <= len && memcmp(text + i, SPL, SPL_LEN) == 0) {
            FLUSH_RUN();
            i += SPL_LEN;
            continue;
        }

        unsigned char c = (unsigned char)text[i];

        /* Stage 2: digits -> individual pieces */
        if (isdigit(c)) {
            FLUSH_RUN();
            if (plist_push(out, i, 1) != 0) return -1;
            i++;
            continue;
        }

        /* Stage 3: brackets or runs of punctuation */
        if (c == '(' || c == ')' || c == '[' || c == ']' ||
            c == '{' || c == '}') {
            FLUSH_RUN();
            if (plist_push(out, i, 1) != 0) return -1;
            i++;
            continue;
        }
        /* Punctuation run: chars from this set, consecutive same char.
         * Set: !"#$%&'*+,-./:;<=>?\^_`{|}~  (note: ' and ` and " included) */
        if (c=='!' || c=='"' || c=='#' || c=='$' || c=='%' || c=='&' ||
            c=='\'' || c=='*' || c=='+' || c==',' || c=='-' || c=='.' ||
            c=='/' || c==':' || c==';' || c=='<' || c=='=' || c=='>' ||
            c=='?' || c=='\\' || c=='^' || c=='_' || c=='`' || c=='{' ||
            c=='|' || c=='}' || c=='~') {
            FLUSH_RUN();
            size_t start = i;
            char match = text[i];
            while (i < len && text[i] == match) i++;
            if (plist_push(out, start, i - start) != 0) return -1;
            continue;
        }

        /* Stage 4: newline isolated */
        if (c == '\n') {
            FLUSH_RUN();
            if (plist_push(out, i, 1) != 0) return -1;
            i++;
            continue;
        }

        /* Otherwise: accumulate into run */
        if (!has_run) { run_start = i; has_run = 1; }
        i++;
    }
    FLUSH_RUN();
    #undef FLUSH_RUN
    return 0;
}

/* ------------------------------------------------------------------ */
/* BPE merge application                                               */
/* ------------------------------------------------------------------ */

/* Given a byte-level-encoded token string (e.g. "Ġthe"), split it into
 * "characters" (where each char is a UTF-8 code point corresponding to
 * one original byte), apply BPE merges greedily by lowest rank, and
 * append the resulting token IDs to *out_ids.
 *
 * Returns 0 on success, -1 on failure. */
static int bpe_apply(Tokenizer *t, const char *piece, int **out_ids,
                     size_t *out_count, size_t *out_cap) {
    /* Decode the byte-level string into "logical chars": each char is
     * the UTF-8 encoding of one code point that came from one byte.
     * Code points in our table are at most 323, so UTF-8 length is
     * either 1 byte (cp <= 127) or 2 bytes (cp <= 2047). We can simply
     * walk UTF-8 code points. */
    size_t plen = strlen(piece);
    if (plen == 0) return 0;

    /* Allocate array of char* (each is a sub-string of `piece`, but
     * to keep memory simple we'll allocate small buffers). */
    size_t max_chars = plen;  /* at most one char per byte (1-byte cps) */
    char **parts = calloc(max_chars, sizeof(char *));
    size_t  nparts = 0;
    if (!parts) return -1;

    for (size_t i = 0; i < plen; ) {
        unsigned char c = (unsigned char)piece[i];
        size_t charlen = (c < 0x80) ? 1 : 2;  /* our code points fit in 1 or 2 bytes */
        if (i + charlen > plen) charlen = 1;
        char *s = malloc(charlen + 1);
        if (!s) { for (size_t k=0;k<nparts;k++) free(parts[k]); free(parts); return -1; }
        memcpy(s, piece + i, charlen);
        s[charlen] = '\0';
        parts[nparts++] = s;
        i += charlen;
    }

    if (nparts == 0) { free(parts); return 0; }

    /* If only one part, look it up directly. */
    while (nparts > 1) {
        /* Find adjacent pair with lowest rank. */
        int    best_rank = -1;
        size_t best_idx  = 0;
        for (size_t k = 0; k + 1 < nparts; k++) {
            /* Build the merge key: "tokA tokB" */
            size_t la = strlen(parts[k]);
            size_t lb = strlen(parts[k+1]);
            size_t keylen = la + 1 + lb;
            char *key = malloc(keylen + 1);
            if (!key) goto fail;
            memcpy(key, parts[k], la);
            key[la] = ' ';
            memcpy(key + la + 1, parts[k+1], lb);
            key[keylen] = '\0';
            int rank;
            int found = hashmap_get(&t->merges, key, keylen, &rank);
            free(key);
            if (found && (best_rank == -1 || rank < best_rank)) {
                best_rank = rank;
                best_idx  = k;
            }
        }
        if (best_rank == -1) break;  /* no more merges */

        /* Merge parts[best_idx] and parts[best_idx+1]. */
        size_t la = strlen(parts[best_idx]);
        size_t lb = strlen(parts[best_idx+1]);
        char *merged = malloc(la + lb + 1);
        if (!merged) goto fail;
        memcpy(merged, parts[best_idx], la);
        memcpy(merged + la, parts[best_idx+1], lb);
        merged[la+lb] = '\0';
        free(parts[best_idx]);
        free(parts[best_idx+1]);
        parts[best_idx] = merged;
        /* Shift the tail down. */
        for (size_t k = best_idx + 1; k + 1 < nparts; k++) {
            parts[k] = parts[k+1];
        }
        nparts--;
    }

    /* Look up each part in vocab; emit id (or unk_id if not found). */
    for (size_t k = 0; k < nparts; k++) {
        size_t lk = strlen(parts[k]);
        int id;
        if (!hashmap_get(&t->vocab, parts[k], lk, &id)) {
            id = (t->unk_id >= 0) ? t->unk_id : 0;
        }
        /* grow out_ids if needed */
        if (*out_count == *out_cap) {
            size_t new_cap = (*out_cap == 0) ? 16 : (*out_cap * 2);
            int *ni = realloc(*out_ids, new_cap * sizeof(int));
            if (!ni) goto fail;
            *out_ids = ni;
            *out_cap = new_cap;
        }
        (*out_ids)[(*out_count)++] = id;
    }

    for (size_t k = 0; k < nparts; k++) free(parts[k]);
    free(parts);
    return 0;

fail:
    for (size_t k = 0; k < nparts; k++) free(parts[k]);
    free(parts);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

static int load_vocab(Tokenizer *t, JsonValue *vocab_obj) {
    if (!vocab_obj || vocab_obj->type != JSON_OBJ) return -1;
    int n = vocab_obj->obj_len;
    if (hashmap_init(&t->vocab, n > (int)HASH_INIT_CAP ? (size_t)n : HASH_INIT_CAP) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        const char *k  = vocab_obj->keys[i];
        JsonValue   *vv = vocab_obj->vals[i];
        if (!vv || vv->type != JSON_NUM) continue;
        int id = (int)vv->num;
        size_t klen = strlen(k);
        if (hashmap_put(&t->vocab, k, klen, id) != 0) return -1;
        if (id > t->max_id) t->max_id = id;
    }
    return 0;
}

static int load_merges(Tokenizer *t, JsonValue *merges_arr) {
    if (!merges_arr || merges_arr->type != JSON_ARR) return -1;
    int n = merges_arr->arr_len;
    if (hashmap_init(&t->merges, n > (int)HASH_INIT_CAP ? (size_t)n : HASH_INIT_CAP) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        JsonValue *m = merges_arr->arr[i];
        if (!m || m->type != JSON_STR) continue;
        /* Each merge is "tokA tokB" - store as-is with rank = i (lower first). */
        size_t mlen = strlen(m->str);
        if (hashmap_put(&t->merges, m->str, mlen, (int)i) != 0) return -1;
    }
    t->merge_count = n;
    return 0;
}

static int load_added_tokens(Tokenizer *t, JsonValue *added_arr) {
    if (!added_arr || added_arr->type != JSON_ARR) return 0;
    int n = added_arr->arr_len;
    /* Allocate generous upper bound; we'll filter to special==true. */
    t->special_str = calloc((size_t)n, sizeof(char *));
    t->special_id  = calloc((size_t)n, sizeof(int));
    if (!t->special_str || !t->special_id) {
        free(t->special_str); free(t->special_id);
        t->special_str = NULL; t->special_id = NULL;
        return -1;
    }
    t->special_count = 0;

    for (int i = 0; i < n; i++) {
        JsonValue *e = added_arr->arr[i];
        if (!e || e->type != JSON_OBJ) continue;
        JsonValue *jid   = json_obj_get(e, "id");
        JsonValue *jcont = json_obj_get(e, "content");
        JsonValue *jspec = json_obj_get(e, "special");
        if (!jid || jid->type != JSON_NUM) continue;
        if (!jcont || jcont->type != JSON_STR) continue;
        if (!jspec || jspec->type != JSON_BOOL || !((int)jspec->num)) continue;

        int id = (int)jid->num;
        const char *content = jcont->str;

        /* Add to vocab if not already there. */
        size_t clen = strlen(content);
        int existing;
        if (!hashmap_get(&t->vocab, content, clen, &existing)) {
            if (hashmap_put(&t->vocab, content, clen, id) != 0) return -1;
        }

        /* Track <unk>. */
        if (strcmp(content, "<unk>") == 0) t->unk_id = id;

        /* Add to special list. */
        char *copy = malloc(clen + 1);
        if (!copy) return -1;
        memcpy(copy, content, clen);
        copy[clen] = '\0';
        t->special_str[t->special_count] = copy;
        t->special_id [t->special_count] = id;
        t->special_count++;

        if (id > t->max_id) t->max_id = id;
    }
    sort_specials(t);
    return 0;
}

static int build_reverse_vocab(Tokenizer *t) {
    /* Walk the vocab hashmap and build an array indexed by id. */
    t->rev_cap = (t->max_id >= 0) ? (size_t)(t->max_id + 1) : 0;
    if (t->rev_cap == 0) {
        t->rev_vocab = NULL;
        return 0;
    }
    t->rev_vocab = calloc(t->rev_cap, sizeof(char *));
    if (!t->rev_vocab) return -1;
    for (size_t i = 0; i < t->vocab.cap; i++) {
        for (HashNode *n = t->vocab.buckets[i]; n; n = n->next) {
            if (n->value >= 0 && (size_t)n->value < t->rev_cap) {
                /* last write wins (no duplicates expected) */
                t->rev_vocab[n->value] = n->key;
            }
        }
    }
    return 0;
}

Tokenizer *tokenizer_load(const char *path) {
    if (!path) return NULL;
    init_byte_maps();

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';

    JsonValue *root = json_parse(buf);
    free(buf);
    if (!root) return NULL;

    Tokenizer *t = calloc(1, sizeof(Tokenizer));
    if (!t) { json_free(root); return NULL; }
    t->unk_id = -1;
    t->max_id = -1;

    JsonValue *model = json_obj_get(root, "model");
    if (!model || model->type != JSON_OBJ) goto fail;

    JsonValue *vocab_obj  = json_obj_get(model, "vocab");
    JsonValue *merges_arr = json_obj_get(model, "merges");

    if (load_vocab(t, vocab_obj) != 0) goto fail;
    if (load_merges(t, merges_arr) != 0) goto fail;

    JsonValue *added = json_obj_get(root, "added_tokens");
    if (load_added_tokens(t, added) != 0) goto fail;

    if (build_reverse_vocab(t) != 0) goto fail;

    json_free(root);
    return t;

fail:
    json_free(root);
    tokenizer_free(t);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Encode                                                              */
/* ------------------------------------------------------------------ */

/* At position `pos` in `text` (length `len`), try to match a special
 * token. Returns the index into t->special_str[] on match, or -1. */
static int match_special_at(Tokenizer *t, const char *text, size_t len, size_t pos) {
    for (size_t i = 0; i < t->special_count; i++) {
        const char *s = t->special_str[i];
        size_t sl = strlen(s);
        if (pos + sl > len) continue;
        if (memcmp(text + pos, s, sl) == 0) return (int)i;
    }
    return -1;
}

int *tokenizer_encode(Tokenizer *tok, const char *text, int *out_len) {
    if (!tok || !text || !out_len) return NULL;
    *out_len = 0;
    size_t len = strlen(text);
    if (len == 0) {
        int *ids = malloc(sizeof(int));
        if (!ids) return NULL;
        return ids;  /* empty */
    }

    int   *ids   = NULL;
    size_t n_ids = 0, cap_ids = 0;

    /* Walk text, emitting special tokens directly, routing non-special
     * runs through the pre-tokenizer + BPE. */
    size_t i = 0;
    size_t seg_start = 0;
    int    in_seg = 0;

    #define FLUSH_SEG(end_pos) do {                                    \
        if (in_seg) {                                                  \
            size_t seg_len = (end_pos) - seg_start;                    \
            if (seg_len > 0) {                                         \
                PieceList pl;                                          \
                if (plist_init(&pl, 16) != 0) { free(ids); return NULL; } \
                if (pretokenize_segment(text + seg_start, seg_len, &pl) != 0) { \
                    plist_free(&pl); free(ids); return NULL;           \
                }                                                     \
                for (size_t p = 0; p < pl.count; p++) {               \
                    char *bs = bytes_to_bytestr(                      \
                        (const unsigned char *)(text + seg_start + pl.starts[p]), \
                        pl.lens[p]);                                  \
                    if (!bs) { plist_free(&pl); free(ids); return NULL; } \
                    if (bpe_apply(tok, bs, &ids, &n_ids, &cap_ids) != 0) { \
                        free(bs); plist_free(&pl); free(ids); return NULL; \
                    }                                                  \
                    free(bs);                                          \
                }                                                     \
                plist_free(&pl);                                      \
            }                                                         \
            in_seg = 0;                                               \
        }                                                             \
    } while (0)

    while (i < len) {
        int sp_idx = match_special_at(tok, text, len, i);
        if (sp_idx >= 0) {
            FLUSH_SEG(i);
            /* emit the special id */
            if (n_ids == cap_ids) {
                size_t new_cap = (cap_ids == 0) ? 16 : cap_ids * 2;
                int *ni = realloc(ids, new_cap * sizeof(int));
                if (!ni) { free(ids); return NULL; }
                ids = ni; cap_ids = new_cap;
            }
            ids[n_ids++] = tok->special_id[sp_idx];
            i += strlen(tok->special_str[sp_idx]);
            continue;
        }
        if (!in_seg) { seg_start = i; in_seg = 1; }
        i++;
    }
    FLUSH_SEG(len);
    #undef FLUSH_SEG

    *out_len = (int)n_ids;
    return ids;
}

/* ------------------------------------------------------------------ */
/* Decode                                                              */
/* ------------------------------------------------------------------ */

/* Reverse the byte-level encoding: walk code points in `s`, look up
 * the byte for each, append the byte to the output buffer. */
static char *bytestr_to_bytes(const char *s) {
    size_t slen = strlen(s);
    char *out = malloc(slen + 1);
    if (!out) return NULL;
    size_t pos = 0;
    for (size_t i = 0; i < slen; ) {
        unsigned char c = (unsigned char)s[i];
        int cp;
        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= slen) { cp = c; i += 1; }
            else {
                cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F);
                i += 2;
            }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= slen) { cp = c; i += 1; }
            else {
                cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6)
                                       | ((unsigned char)s[i+2] & 0x3F);
                i += 3;
            }
        } else {
            /* 4-byte UTF-8 - shouldn't happen for our byte-level map */
            cp = c; i += 1;
        }
        if (cp < (int)(sizeof(cp_to_byte)/sizeof(int)) && cp_to_byte[cp] >= 0) {
            out[pos++] = (char)cp_to_byte[cp];
        } else {
            /* Unknown: emit '?' as a fallback. */
            out[pos++] = '?';
        }
    }
    out[pos] = '\0';
    return out;
}

char *tokenizer_decode(Tokenizer *tok, const int *ids, int len) {
    if (!tok || (!ids && len > 0)) return NULL;
    /* Compute total length first to allocate a single buffer. */
    size_t total = 1;  /* NUL */
    for (int i = 0; i < len; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= tok->rev_cap) continue;
        const char *s = tok->rev_vocab[ids[i]];
        if (s) total += strlen(s);
    }
    char *out = malloc(total);
    if (!out) return NULL;
    size_t pos = 0;
    for (int i = 0; i < len; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= tok->rev_cap) continue;
        const char *s = tok->rev_vocab[ids[i]];
        if (!s) continue;
        size_t sl = strlen(s);
        memcpy(out + pos, s, sl);
        pos += sl;
    }
    out[pos] = '\0';

    /* Reverse byte-level encoding on the concatenated string. */
    char *rev = bytestr_to_bytes(out);
    free(out);
    return rev;
}

/* ------------------------------------------------------------------ */
/* Free                                                                */
/* ------------------------------------------------------------------ */

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    hashmap_free(&t->vocab);
    hashmap_free(&t->merges);
    free(t->rev_vocab);
    for (size_t i = 0; i < t->special_count; i++) free(t->special_str[i]);
    free(t->special_str);
    free(t->special_id);
    free(t);
}
