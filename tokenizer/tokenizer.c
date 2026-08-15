/* tokenizer.c — Minimal HF tokenizer loader (added_tokens only).
 *
 * Parses tokenizer.json just enough to extract the "added_tokens" array
 * (id + content). The forward-pass driver uses this to resolve special
 * tokens like "<s>", "</s>", "<s_docvqa>". */
#include "tokenizer.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
/* C11: strdup is POSIX, not ISO C — declare it ourselves */
extern char *strdup(const char *);
#endif

Tokenizer *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = 0;
    fclose(f);

    JsonValue *root = json_parse(buf);
    free(buf);
    if (!root) return NULL;

    const JsonValue *added = json_object_get(root, "added_tokens");
    if (!added || added->type != JSON_ARRAY) { json_free(root); return NULL; }

    Tokenizer *tk = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    if (!tk) { json_free(root); return NULL; }
    tk->count = (int)added->v.array.count;
    tk->cap   = tk->count;
    tk->tokens = (AddedToken *)calloc(tk->cap, sizeof(AddedToken));
    if (!tk->tokens) { free(tk); json_free(root); return NULL; }

    for (int i = 0; i < tk->count; i++) {
        const JsonValue *e = json_array_get(added, i);
        if (!e) continue;
        const JsonValue *jid = json_object_get(e, "id");
        const JsonValue *jc  = json_object_get(e, "content");
        tk->tokens[i].id = (int)json_as_long(jid);
        char tmp[256];
        int n = json_as_string(jc, tmp, sizeof(tmp));
        if (n < 0) { tmp[0] = 0; n = 0; }
        tk->tokens[i].content = strdup(tmp);
    }
    json_free(root);
    return tk;
}

int tokenizer_lookup(const Tokenizer *tk, const char *content) {
    if (!tk || !content) return -1;
    for (int i = 0; i < tk->count; i++) {
        if (tk->tokens[i].content && strcmp(tk->tokens[i].content, content) == 0)
            return tk->tokens[i].id;
    }
    return -1;
}

void tokenizer_free(Tokenizer *tk) {
    if (!tk) return;
    for (int i = 0; i < tk->count; i++) free(tk->tokens[i].content);
    free(tk->tokens);
    free(tk);
}
