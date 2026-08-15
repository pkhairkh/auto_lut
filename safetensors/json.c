/* json.c -- minimal recursive-descent JSON parser for auto_lut.
 *
 * Design goals:
 *   - Single C11 translation unit, no external deps.
 *   - Strict enough for safetensors headers and small model configs.
 *   - Tolerant of trailing whitespace; rejects malformed input.
 *
 * Memory: all allocations use malloc()/realloc(). The caller frees the
 * whole tree with json_free(root).
 */

#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* parser state                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *p;        /* current cursor */
    const char *end;       /* one past last byte */
    int         error;    /* set on parse failure */
} Parser;

static void skip_ws(Parser *s)
{
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            s->p++;
        } else {
            break;
        }
    }
}

static JsonValue *parse_value(Parser *s);

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static JsonValue *new_value(JsonType t)
{
    JsonValue *v = (JsonValue *)calloc(1, sizeof(*v));
    if (v) v->type = t;
    return v;
}

static char *strndup_raw(const char *src, size_t n)
{
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, src, n);
    p[n] = '\0';
    return p;
}

/* Decode a JSON string literal beginning at *s->p == '"'. Returns a
 * malloc'd NUL-terminated string, or NULL on error. Advances s->p
 * past the closing quote. */
static char *parse_string_raw(Parser *s)
{
    if (s->p >= s->end || *s->p != '"') { s->error = 1; return NULL; }
    s->p++; /* skip opening quote */

    /* worst case: every char becomes 1 output byte */
    size_t cap = 16, len = 0;
    char  *out = (char *)malloc(cap);
    if (!out) { s->error = 1; return NULL; }

    while (s->p < s->end) {
        unsigned char c = (unsigned char)*s->p++;
        if (c == '"') {
            if (len + 1 > cap) {
                char *t = (char *)realloc(out, len + 1);
                if (!t) { free(out); s->error = 1; return NULL; }
                out = t;
            }
            out[len] = '\0';
            return out;
        }
        if (c == '\\') {
            if (s->p >= s->end) { free(out); s->error = 1; return NULL; }
            char esc = *s->p++;
            switch (esc) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    /* \uXXXX -> UTF-8. Decode 4 hex digits. We support
                     * basic BMP chars (no surrogate-pair merging, since
                     * safetensors headers don't need it). */
                    if (s->p + 4 > s->end) { free(out); s->error = 1; return NULL; }
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = s->p[i];
                        unsigned int d;
                        if (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                        else { free(out); s->error = 1; return NULL; }
                        cp = (cp << 4) | d;
                    }
                    s->p += 4;
                    /* Encode cp as UTF-8 */
                    if (len + 5 > cap) {
                        while (len + 5 > cap) cap *= 2;
                        char *t = (char *)realloc(out, cap);
                        if (!t) { free(out); s->error = 1; return NULL; }
                        out = t;
                    }
                    if (cp < 0x80) {
                        out[len++] = (char)cp;
                    } else if (cp < 0x800) {
                        out[len++] = (char)(0xC0 | (cp >> 6));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[len++] = (char)(0xE0 | (cp >> 12));
                        out[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue;
                }
                default:
                    free(out); s->error = 1; return NULL;
            }
        } else if (c < 0x20) {
            /* control char not allowed */
            free(out); s->error = 1; return NULL;
        }
        if (len + 2 > cap) {
            cap *= 2;
            char *t = (char *)realloc(out, cap);
            if (!t) { free(out); s->error = 1; return NULL; }
            out = t;
        }
        out[len++] = (char)c;
    }
    free(out);
    s->error = 1;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* number, literal, array, object                                     */
/* ------------------------------------------------------------------ */

static JsonValue *parse_number(Parser *s)
{
    const char *start = s->p;
    if (s->p < s->end && (*s->p == '-')) s->p++;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    if (s->p < s->end && *s->p == '.') {
        s->p++;
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    }
    if (s->p < s->end && (*s->p == 'e' || *s->p == 'E')) {
        s->p++;
        if (s->p < s->end && (*s->p == '+' || *s->p == '-')) s->p++;
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    }
    if (s->p == start) { s->error = 1; return NULL; }

    char buf[64];
    size_t n = (size_t)(s->p - start);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, start, n);
    buf[n] = '\0';
    JsonValue *v = new_value(JSON_NUMBER);
    if (!v) { s->error = 1; return NULL; }
    v->v.number = strtod(buf, NULL);
    return v;
}

static JsonValue *parse_literal(Parser *s, const char *kw, JsonType t, int boolval)
{
    size_t n = strlen(kw);
    if (s->p + n > s->end || memcmp(s->p, kw, n) != 0) {
        s->error = 1; return NULL;
    }
    s->p += n;
    JsonValue *v = new_value(t);
    if (!v) { s->error = 1; return NULL; }
    v->v.boolean = boolval;
    return v;
}

static JsonValue *parse_array(Parser *s)
{
    s->p++; /* skip '[' */
    JsonValue *v = new_value(JSON_ARRAY);
    if (!v) { s->error = 1; return NULL; }
    v->v.array.items = NULL;
    v->v.array.count = 0;
    size_t cap = 0;

    skip_ws(s);
    if (s->p < s->end && *s->p == ']') { s->p++; return v; }

    for (;;) {
        JsonValue *item = parse_value(s);
        if (!item) { json_free(v); return NULL; }
        if (v->v.array.count == cap) {
            size_t newcap = cap ? cap * 2 : 8;
            JsonValue **t = (JsonValue **)realloc(v->v.array.items, newcap * sizeof(*t));
            if (!t) { json_free(item); json_free(v); s->error = 1; return NULL; }
            v->v.array.items = t;
            cap = newcap;
        }
        v->v.array.items[v->v.array.count++] = item;
        skip_ws(s);
        if (s->p >= s->end) { json_free(v); s->error = 1; return NULL; }
        if (*s->p == ',') { s->p++; skip_ws(s); continue; }
        if (*s->p == ']') { s->p++; return v; }
        json_free(v); s->error = 1; return NULL;
    }
}

static JsonValue *parse_object(Parser *s)
{
    s->p++; /* skip '{' */
    JsonValue *v = new_value(JSON_OBJECT);
    if (!v) { s->error = 1; return NULL; }
    v->v.object.entries = NULL;
    v->v.object.count = 0;
    size_t cap = 0;

    skip_ws(s);
    if (s->p < s->end && *s->p == '}') { s->p++; return v; }

    for (;;) {
        skip_ws(s);
        if (s->p >= s->end || *s->p != '"') { json_free(v); s->error = 1; return NULL; }
        char *key = parse_string_raw(s);
        if (!key) { json_free(v); return NULL; }
        skip_ws(s);
        if (s->p >= s->end || *s->p != ':') { free(key); json_free(v); s->error = 1; return NULL; }
        s->p++;
        skip_ws(s);
        JsonValue *val = parse_value(s);
        if (!val) { free(key); json_free(v); return NULL; }
        if (v->v.object.count == cap) {
            size_t newcap = cap ? cap * 2 : 8;
            JsonObjectEntry *t = (JsonObjectEntry *)realloc(v->v.object.entries,
                newcap * sizeof(*t));
            if (!t) { free(key); json_free(val); json_free(v); s->error = 1; return NULL; }
            v->v.object.entries = t;
            cap = newcap;
        }
        v->v.object.entries[v->v.object.count].key   = key;
        v->v.object.entries[v->v.object.count].value  = val;
        v->v.object.count++;
        skip_ws(s);
        if (s->p >= s->end) { json_free(v); s->error = 1; return NULL; }
        if (*s->p == ',') { s->p++; continue; }
        if (*s->p == '}') { s->p++; return v; }
        json_free(v); s->error = 1; return NULL;
    }
}

static JsonValue *parse_value(Parser *s)
{
    skip_ws(s);
    if (s->p >= s->end) { s->error = 1; return NULL; }
    char c = *s->p;
    switch (c) {
        case '{': return parse_object(s);
        case '[': return parse_array(s);
        case '"': {
            JsonValue *v = new_value(JSON_STRING);
            if (!v) { s->error = 1; return NULL; }
            v->v.string = parse_string_raw(s);
            if (!v->v.string) { free(v); return NULL; }
            return v;
        }
        case 't': return parse_literal(s, "true",  JSON_BOOL, 1);
        case 'f': return parse_literal(s, "false", JSON_BOOL, 0);
        case 'n': return parse_literal(s, "null",  JSON_NULL, 0);
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number(s);
            s->error = 1; return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

JsonValue *json_parse_n(const char *text, size_t len)
{
    if (!text || len == 0) return NULL;
    Parser s = { text, text + len, 0 };
    JsonValue *v = parse_value(&s);
    if (!v) return NULL;
    skip_ws(&s);
    /* allow trailing whitespace but nothing else */
    if (s.error || s.p != s.end) {
        json_free(v);
        return NULL;
    }
    return v;
}

JsonValue *json_parse(const char *text)
{
    if (!text) return NULL;
    return json_parse_n(text, strlen(text));
}

void json_free(JsonValue *v)
{
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            free(v->v.string);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->v.array.count; i++)
                json_free(v->v.array.items[i]);
            free(v->v.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->v.object.count; i++) {
                free(v->v.object.entries[i].key);
                json_free(v->v.object.entries[i].value);
            }
            free(v->v.object.entries);
            break;
        default: break;
    }
    free(v);
}

const JsonValue *json_object_get(const JsonValue *v, const char *key)
{
    if (!v || v->type != JSON_OBJECT || !key) return NULL;
    for (size_t i = 0; i < v->v.object.count; i++) {
        if (strcmp(v->v.object.entries[i].key, key) == 0)
            return v->v.object.entries[i].value;
    }
    return NULL;
}

const JsonValue *json_array_get(const JsonValue *v, size_t index)
{
    if (!v || v->type != JSON_ARRAY) return NULL;
    if (index >= v->v.array.count) return NULL;
    return v->v.array.items[index];
}

long json_as_long(const JsonValue *v)
{
    if (!v) return 0;
    if (v->type == JSON_NUMBER) return (long)v->v.number;
    if (v->type == JSON_BOOL)  return v->v.boolean ? 1L : 0L;
    return 0;
}

int json_as_string(const JsonValue *v, char *out, size_t cap)
{
    if (!v || v->type != JSON_STRING) return -1;
    if (!out || cap == 0) return -1;
    size_t n = strlen(v->v.string);
    if (n >= cap) n = cap - 1;
    memcpy(out, v->v.string, n);
    out[n] = '\0';
    return (int)n;
}
